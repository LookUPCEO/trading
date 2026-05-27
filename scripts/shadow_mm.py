"""SHADOW LIVE MM: Bybit WebSocket reader → simulated fills → real maker rate measurement.

NO REAL ORDERS. No capital risk. Reads OB + trades, simulates passive limit fill behavior,
records: maker fill rate, queue dynamics, adverse selection. Run for 7 days.
"""
import asyncio, json, logging, os, time, sys
from datetime import datetime, timezone
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets

# ---- Config ----
SYMBOL = os.environ.get("SHADOW_SYMBOL", "ETHUSDT")
WS_URL = "wss://stream.bybit.com/v5/public/linear"
ORDER_SIZE = 0.01           # ETH (virtual)
SPREAD_BP = 0               # 0 = at best
MAX_INVENTORY = 0.05        # virtual
QUEUE_DEPLETION_FACTOR = 0.5
ADVERSE_DRIFT_THRESHOLD = 0.0005
ADVERSE_LOOKAHEAD_SEC = 60
LOG_INTERVAL_SEC = 300      # log summary every 5 min
PERSIST_INTERVAL_SEC = 600  # persist state every 10 min
LATENCY_SEC = 1
DEPTH = 50

# ---- Logging ----
log_dir = Path("/Users/dohun/Desktop/Mark/mark19/logs")
log_dir.mkdir(exist_ok=True)
out_file = log_dir / f"shadow_mm_{SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
state_file = Path(f"/Users/dohun/Desktop/Mark/mark19/data/analysis_results/shadow_mm_state_{SYMBOL}.json")
state_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(out_file), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ---- State ----
class OrderBook:
    def __init__(self):
        self.bids = {}  # price → size
        self.asks = {}

    def apply_snapshot(self, b, a):
        self.bids = {float(p): float(s) for p, s in b}
        self.asks = {float(p): float(s) for p, s in a}

    def apply_delta(self, b, a):
        for p, s in b:
            p = float(p); s = float(s)
            if s == 0: self.bids.pop(p, None)
            else: self.bids[p] = s
        for p, s in a:
            p = float(p); s = float(s)
            if s == 0: self.asks.pop(p, None)
            else: self.asks[p] = s

    def best(self):
        if not self.bids or not self.asks: return None, None, None, None
        bb = max(self.bids); ba = min(self.asks)
        return bb, self.bids[bb], ba, self.asks[ba]


class ShadowState:
    def __init__(self):
        self.start_ts = datetime.now(timezone.utc)
        self.bid_state = None  # {price, queue, place_ts}
        self.ask_state = None
        self.inventory = 0.0
        self.fills = []  # {ts, side, price, mid, queue_at_fill}
        self.cancels = 0
        self.places = 0
        self.cooldown_bid_until = None
        self.cooldown_ask_until = None
        self.last_log_ts = self.start_ts
        self.last_persist_ts = self.start_ts
        # Adverse: pending fills awaiting drift check
        self.pending_adverse = []  # {fill_ts, fill_mid, side, fill_price}
        self.n_toxic = 0
        self.n_favorable = 0

    def to_dict(self):
        return {
            "symbol": SYMBOL,
            "start_ts": str(self.start_ts),
            "current_ts": str(datetime.now(timezone.utc)),
            "uptime_sec": (datetime.now(timezone.utc) - self.start_ts).total_seconds(),
            "inventory": self.inventory,
            "n_fills": len(self.fills),
            "n_cancels": self.cancels,
            "n_places": self.places,
            "n_toxic": self.n_toxic,
            "n_favorable": self.n_favorable,
            "toxic_rate": self.n_toxic / max(self.n_toxic + self.n_favorable, 1),
            "pending_adverse": len(self.pending_adverse),
            "params": {
                "size_eth": ORDER_SIZE, "spread_bp": SPREAD_BP,
                "max_inventory": MAX_INVENTORY, "queue_depletion": QUEUE_DEPLETION_FACTOR,
            },
            "recent_fills": [
                {"ts": str(f["ts"]), "side": f["side"], "price": f["price"], "mid": f["mid"]}
                for f in self.fills[-10:]
            ],
        }


def maybe_log_summary(state):
    now = datetime.now(timezone.utc)
    if (now - state.last_log_ts).total_seconds() >= LOG_INTERVAL_SEC:
        d = state.to_dict()
        log.info(f"=== SHADOW SUMMARY ({d['uptime_sec']/60:.1f}min) ===")
        log.info(f"  Fills: {d['n_fills']}  Cancels: {d['n_cancels']}  Places: {d['n_places']}")
        log.info(f"  Inventory: {d['inventory']:.4f}  Toxic: {d['n_toxic']}  Favorable: {d['n_favorable']}  Toxic rate: {d['toxic_rate']*100:.1f}%")
        if d['n_fills'] > 0:
            recent = state.fills[-1]
            log.info(f"  Last fill: {recent['side']} @ ${recent['price']:.2f}")
        state.last_log_ts = now

    if (now - state.last_persist_ts).total_seconds() >= PERSIST_INTERVAL_SEC:
        try:
            with open(state_file, "w") as f:
                json.dump(state.to_dict(), f, indent=2, default=str)
        except Exception as e:
            log.warning(f"State persist failed: {e}")
        state.last_persist_ts = now


def process_adverse(state, current_mid, now_ts):
    """Check pending fills for adverse drift after ADVERSE_LOOKAHEAD_SEC."""
    still_pending = []
    for adv in state.pending_adverse:
        elapsed = (now_ts - adv["fill_ts"]).total_seconds()
        if elapsed >= ADVERSE_LOOKAHEAD_SEC:
            drift = (current_mid - adv["fill_mid"]) / adv["fill_mid"]
            if adv["side"] == "bid":
                # Bought; toxic if drift < -threshold (price dropped after buy)
                if drift < -ADVERSE_DRIFT_THRESHOLD: state.n_toxic += 1
                elif drift > ADVERSE_DRIFT_THRESHOLD: state.n_favorable += 1
            else:
                if drift > ADVERSE_DRIFT_THRESHOLD: state.n_toxic += 1
                elif drift < -ADVERSE_DRIFT_THRESHOLD: state.n_favorable += 1
        else:
            still_pending.append(adv)
    state.pending_adverse = still_pending


def update_orders(state, ob, now_ts):
    """Cancel/replace shadow orders based on current OB state."""
    bb, bsz, ba, asz = ob.best()
    if bb is None: return
    mid = (bb + ba) / 2
    target_bid = bb * (1 - SPREAD_BP / 10000)
    target_ask = ba * (1 + SPREAD_BP / 10000)

    # Cancel if our level too far from target
    if state.bid_state is not None:
        if abs(state.bid_state["price"] - target_bid) / target_bid > 0.0001:
            state.cancels += 1
            state.bid_state = None
            state.cooldown_bid_until = now_ts.timestamp() + LATENCY_SEC
    if state.ask_state is not None:
        if abs(state.ask_state["price"] - target_ask) / target_ask > 0.0001:
            state.cancels += 1
            state.ask_state = None
            state.cooldown_ask_until = now_ts.timestamp() + LATENCY_SEC

    # Place if missing & cooldown OK & inv allows
    now_ts_f = now_ts.timestamp()
    if (state.bid_state is None and state.inventory < MAX_INVENTORY
        and (state.cooldown_bid_until is None or now_ts_f >= state.cooldown_bid_until)):
        # Initial queue = bid_0_size if at best, halved if outside
        initial_q = bsz if SPREAD_BP <= 0 else bsz * 0.5
        state.bid_state = {"price": target_bid, "queue": initial_q, "place_ts": now_ts, "place_mid": mid}
        state.places += 1
    if (state.ask_state is None and state.inventory > -MAX_INVENTORY
        and (state.cooldown_ask_until is None or now_ts_f >= state.cooldown_ask_until)):
        initial_q = asz if SPREAD_BP <= 0 else asz * 0.5
        state.ask_state = {"price": target_ask, "queue": initial_q, "place_ts": now_ts, "place_mid": mid}
        state.places += 1


def process_trade(state, ob, side, price, size, now_ts):
    """Trade arrived — check if fills any of our shadow orders."""
    bb, bsz, ba, asz = ob.best()
    if bb is None: return
    mid = (bb + ba) / 2

    if state.bid_state is not None and side == "Sell" and price <= state.bid_state["price"] + 1e-9:
        state.bid_state["queue"] -= size * QUEUE_DEPLETION_FACTOR
        if state.bid_state["queue"] <= 0:
            fp = state.bid_state["price"]
            state.fills.append({"ts": now_ts, "side": "bid", "price": fp,
                                "mid": mid, "queue_at_fill": state.bid_state["queue"]})
            state.inventory += ORDER_SIZE
            state.pending_adverse.append({"fill_ts": now_ts, "fill_mid": mid,
                                           "side": "bid", "fill_price": fp})
            state.bid_state = None
            state.cooldown_bid_until = now_ts.timestamp() + LATENCY_SEC

    if state.ask_state is not None and side == "Buy" and price >= state.ask_state["price"] - 1e-9:
        state.ask_state["queue"] -= size * QUEUE_DEPLETION_FACTOR
        if state.ask_state["queue"] <= 0:
            fp = state.ask_state["price"]
            state.fills.append({"ts": now_ts, "side": "ask", "price": fp,
                                "mid": mid, "queue_at_fill": state.ask_state["queue"]})
            state.inventory -= ORDER_SIZE
            state.pending_adverse.append({"fill_ts": now_ts, "fill_mid": mid,
                                           "side": "ask", "fill_price": fp})
            state.ask_state = None
            state.cooldown_ask_until = now_ts.timestamp() + LATENCY_SEC


async def run():
    log.info("=" * 70)
    log.info(f"SHADOW LIVE MM — symbol={SYMBOL}, size={ORDER_SIZE}, spread={SPREAD_BP}bp")
    log.info("=" * 70)
    log.info("READ-ONLY WebSocket. NO real orders. Simulated fills only.")

    ob = OrderBook()
    state = ShadowState()
    ob_topic = f"orderbook.{DEPTH}.{SYMBOL}"
    tr_topic = f"publicTrade.{SYMBOL}"

    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20) as ws:
                sub = {"op": "subscribe", "args": [ob_topic, tr_topic]}
                await ws.send(json.dumps(sub))
                log.info(f"Subscribed: {ob_topic}, {tr_topic}")

                async for raw in ws:
                    msg = json.loads(raw)
                    topic = msg.get("topic", "")
                    now_ts = datetime.now(timezone.utc)

                    if topic == ob_topic:
                        data = msg.get("data", {})
                        msg_type = msg.get("type")
                        b = data.get("b", [])
                        a = data.get("a", [])
                        if msg_type == "snapshot":
                            ob.apply_snapshot(b, a)
                        elif msg_type == "delta":
                            ob.apply_delta(b, a)
                        update_orders(state, ob, now_ts)
                        # Check adverse periodically
                        bb, _, ba, _ = ob.best()
                        if bb is not None:
                            mid = (bb + ba) / 2
                            process_adverse(state, mid, now_ts)
                        maybe_log_summary(state)

                    elif topic == tr_topic:
                        for trade in msg.get("data", []):
                            try:
                                side = trade.get("S")
                                price = float(trade.get("p"))
                                size = float(trade.get("v"))
                                process_trade(state, ob, side, price, size, now_ts)
                            except Exception as e:
                                log.warning(f"trade parse: {e}")

        except websockets.exceptions.ConnectionClosed as e:
            log.warning(f"WS closed: {e}; reconnecting in 5s")
            await asyncio.sleep(5)
        except Exception as e:
            log.error(f"WS error: {e}; reconnecting in 5s", exc_info=True)
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("\nstopped by user")
