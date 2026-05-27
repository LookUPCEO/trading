"""
Bybit V5 public WebSocket — orderbook.50.ETHUSDT + publicTrade.ETHUSDT.

Maintains:
  - Latest 50-level orderbook via snapshot + delta updates
  - Rolling 1-second trade aggregation
  - 5-minute bar builder (same schema as backtest v3)
  - Disconnect/reconnect counter, stale detection

NO trading logic here. Feeds BarBuffer + persists bars to disk.

URL: wss://stream.bybit.com/v5/public/linear
Docs: https://bybit-exchange.github.io/docs/v5/ws/connect

Reconnect: every disconnect or stale (>60s no message) → reconnect with exponential backoff.
Persist: every completed 5-min bar written to ~/mark19_data/bars_5min_v3_live/ETHUSDT/{date}.parquet
         (append-friendly; restart safely resumes)
"""
from __future__ import annotations
import json, logging, os, queue, signal, sys, threading, time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import websocket  # websocket-client

# Optional Discord notifier (silent if missing)
try:
    import discord_notify as dn
except Exception:
    dn = None


WS_URL = "wss://stream.bybit.com/v5/public/linear"
SYMBOL = "ETHUSDT"
BAR_SECONDS = 300
STALE_THRESHOLD_SEC = 60.0
RECONNECT_BACKOFF_MAX_SEC = 60.0
PERSIST_DIR = Path(os.environ.get("MARK19_LIVE_BARS_DIR",
                                    "/Users/mark/mark19_data/bars_5min_v3_live")) / SYMBOL
PERSIST_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/Users/mark/mark19_data/ws_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============== Orderbook state ==============
class OrderBookState:
    """Maintains a 50-level orderbook via snapshot + delta apply."""
    def __init__(self):
        self.bids: dict[float, float] = {}  # price → size
        self.asks: dict[float, float] = {}
        self.last_update_id: int = 0
        self.last_update_ts: float = 0.0
        self.lock = threading.Lock()

    def apply_snapshot(self, bids: list, asks: list, u: int, ts: int):
        with self.lock:
            self.bids = {float(p): float(s) for p, s in bids if float(s) > 0}
            self.asks = {float(p): float(s) for p, s in asks if float(s) > 0}
            self.last_update_id = u
            self.last_update_ts = ts / 1000.0

    def apply_delta(self, bids: list, asks: list, u: int, ts: int):
        with self.lock:
            for p, s in bids:
                p, s = float(p), float(s)
                if s == 0: self.bids.pop(p, None)
                else: self.bids[p] = s
            for p, s in asks:
                p, s = float(p), float(s)
                if s == 0: self.asks.pop(p, None)
                else: self.asks[p] = s
            self.last_update_id = u
            self.last_update_ts = ts / 1000.0

    def snapshot_top(self, n: int = 50) -> dict:
        """Return frozen snapshot of top-n bids/asks (sorted)."""
        with self.lock:
            top_bids = sorted(self.bids.items(), key=lambda kv: -kv[0])[:n]
            top_asks = sorted(self.asks.items(), key=lambda kv: kv[0])[:n]
            return {
                "ts": self.last_update_ts,
                "u": self.last_update_id,
                "bids": top_bids,   # [(price, size), ...]
                "asks": top_asks,
            }


# ============== Intra-bar 1Hz cache (for accurate OHLC + vel + obi_std) ==============
class IntraBarCache:
    """Stores per-second mid/obi5 snapshots so bar OHLC matches backtest formula exactly.

    Backtest pipeline (build_intraday_bars_v3.py): for each 1Hz orderbook row, compute
    mid, obi5, log_ret_step (= log(mid).diff()), obi5_step (= obi5.diff()). Then groupby
    bar_idx and aggregate: first/last/max/min on mid; std/mean on log_ret_step;
    std on obi5 + obi5_step. Live must mirror this.

    Sampling: pop the latest OB snapshot once per wall-clock second. Anything between
    seconds is averaged out (negligible — Bybit OB usually updates every 100ms; per-sec
    sample matches backtest input granularity).
    """
    def __init__(self):
        # Each entry: (ts_sec_int, mid, obi5)
        self.entries: list = []
        self.lock = threading.Lock()
        self.last_sampled_sec = None
        self._prev_obi5 = None

    def sample(self, ts_sec_int: int, mid: float, obi5: float):
        """Append one sample if it's a new second (deduplicate)."""
        with self.lock:
            if self.last_sampled_sec is not None and ts_sec_int == self.last_sampled_sec:
                # Update with most recent within-second value (matches Bybit raw 'last' aggregation)
                self.entries[-1] = (ts_sec_int, mid, obi5)
                return
            self.entries.append((ts_sec_int, mid, obi5))
            self.last_sampled_sec = ts_sec_int

    def drain_window(self, start_sec: int, end_sec: int) -> list:
        """Return entries with start_sec <= ts < end_sec, removing older."""
        with self.lock:
            window = [e for e in self.entries if start_sec <= e[0] < end_sec]
            # Keep entries at or after end_sec
            self.entries = [e for e in self.entries if e[0] >= end_sec]
            return window


# ============== Trade buffer ==============
class TradeBuffer:
    """Append-only 5-min trade buffer; consumed at bar boundary."""
    def __init__(self):
        self.trades: list[dict] = []
        self.lock = threading.Lock()
        self.last_trade_ts: float = 0.0

    def add(self, ts_ms: int, side: str, price: float, size: float):
        with self.lock:
            self.trades.append({"ts": ts_ms / 1000.0, "side": side, "price": price, "size": size})
            self.last_trade_ts = ts_ms / 1000.0

    def consume_window(self, start_ts: float, end_ts: float) -> list[dict]:
        """Return trades with start_ts <= ts < end_ts, removing older trades."""
        with self.lock:
            window = [t for t in self.trades if start_ts <= t["ts"] < end_ts]
            # Keep only future trades after end_ts (memory)
            self.trades = [t for t in self.trades if t["ts"] >= end_ts]
            return window


# ============== Bar Builder (5-min) — backtest-identical formulas ==============
def build_bar(ob_snap: dict, trades: list[dict], intra_cache: list,
              bar_start: datetime, bar_end: datetime) -> dict:
    """Match v3 schema EXACTLY (53 cols).

    intra_cache: list of (ts_sec, mid, obi5) tuples during the bar (1Hz samples).
    Used to compute mid_open/high/low + rv_bar_bp + vel + obi5_std + obi5_step_std
    in the same way backtest does (pandas groupby on 1Hz dataframe).

    Falls back to OB-only approximation if intra_cache too short (<5 samples).
    """
    bids = ob_snap["bids"]; asks = ob_snap["asks"]
    if not bids or not asks:
        return None
    bid_top = bids[0]; ask_top = asks[0]
    mid_close = (bid_top[0] + ask_top[0]) / 2
    spread = ask_top[0] - bid_top[0]
    spread_bp = spread / mid_close * 10000

    def depth_at(side, k):
        return sum(s for _, s in side[:k])
    def obi_at(k):
        b = depth_at(bids, k); a = depth_at(asks, k)
        t = b + a
        return (b - a) / t if t > 0 else 0.0
    obi1 = obi_at(1); obi5 = obi_at(5); obi10 = obi_at(10); obi20 = obi_at(20); obi50 = obi_at(50)
    tot_d5 = depth_at(bids, 5) + depth_at(asks, 5)
    tot_d10 = depth_at(bids, 10) + depth_at(asks, 10)
    tot_d20 = depth_at(bids, 20) + depth_at(asks, 20)
    tot_d50 = depth_at(bids, 50) + depth_at(asks, 50)
    def slope(side, span):
        if len(side) < span: return 0.0
        if side[0][0] == 0: return 0.0
        return abs(side[0][0] - side[span-1][0]) / span
    bid_sl5 = slope(bids, 5); ask_sl5 = slope(asks, 5)
    bid_sl10 = slope(bids, 10); ask_sl10 = slope(asks, 10)
    bid_sl20 = slope(bids, 20); ask_sl20 = slope(asks, 20)
    b0_size = bids[0][1]; a0_size = asks[0][1]
    mt = b0_size + a0_size
    micro_dev_bp_last = ((ask_top[0]*b0_size + bid_top[0]*a0_size)/mt - mid_close) / mid_close * 10000 if mt > 0 else 0
    concentration_5_50 = tot_d5 / tot_d50 if tot_d50 > 0 else 0
    depth_asym_50 = depth_at(bids, 50) / depth_at(asks, 50) if depth_at(asks, 50) > 0 else 1
    bid_wall = max(s for _, s in bids[:10]) / np.mean([s for _, s in bids[:10]]) if bids[:10] else 1
    ask_wall = max(s for _, s in asks[:10]) / np.mean([s for _, s in asks[:10]]) if asks[:10] else 1

    # === Intra-bar OHLC + rv + vel + obi std (backtest-identical) ===
    if intra_cache and len(intra_cache) >= 2:
        mids = np.array([m for _, m, _ in intra_cache], dtype=np.float64)
        obi5_series = np.array([o for _, _, o in intra_cache], dtype=np.float64)
        mid_open = float(mids[0])
        mid_close_intra = float(mids[-1])
        mid_high = float(mids.max())
        mid_low = float(mids.min())
        log_mid = np.log(mids)
        log_ret_step = np.diff(log_mid, prepend=log_mid[0])  # match pandas .diff().fillna(0)
        log_ret_step[0] = 0.0
        rv_bar_bp = float(np.std(log_ret_step, ddof=1) * 10000) if len(log_ret_step) > 1 else 0.0
        vel_mean_bp = float(log_ret_step.mean() * 10000)
        vel_abs_mean_bp = float(np.abs(log_ret_step).mean() * 10000)
        obi5_std = float(np.std(obi5_series, ddof=1)) if len(obi5_series) > 1 else 0.0
        obi5_step = np.diff(obi5_series, prepend=obi5_series[0])
        obi5_step[0] = 0.0
        obi5_step_std = float(np.std(obi5_step, ddof=1)) if len(obi5_step) > 1 else 0.0
        obi5_mean = float(obi5_series.mean())
        # use intra mid_close (more recent than OB top)
        mid_close = mid_close_intra
        intra_used = len(intra_cache)
    else:
        # Fallback: degenerate bar — log a warning at caller. OHLC collapses to close.
        mid_open = mid_high = mid_low = mid_close
        rv_bar_bp = vel_mean_bp = vel_abs_mean_bp = 0.0
        obi5_std = obi5_step_std = 0.0
        obi5_mean = obi5
        intra_used = 0

    # === Trade aggregates (per-bar) ===
    n_trades = len(trades)
    buys = [t for t in trades if t["side"] == "Buy"]
    sells = [t for t in trades if t["side"] == "Sell"]
    buy_size = sum(t["size"] for t in buys); sell_size = sum(t["size"] for t in sells)
    buy_notional = sum(t["size"] * t["price"] for t in buys)
    sell_notional = sum(t["size"] * t["price"] for t in sells)
    total_notional = buy_notional + sell_notional
    sizes = [t["size"] for t in trades]
    p95 = float(np.percentile(sizes, 95)) if sizes else 0
    tr_large_count = sum(1 for s in sizes if s > p95) if sizes else 0
    tr_max_size = max(sizes) if sizes else 0
    sz_price = sum(t["size"] * t["price"] for t in trades)
    sz_sum = sum(sizes) if sizes else 0
    tr_vwap = sz_price / sz_sum if sz_sum > 0 else mid_close
    tr_vwap_bp_dev = (tr_vwap - mid_close) / mid_close * 10000

    # OFI proxy: sum of top-of-book imbalance changes during bar. WS doesn't directly
    # give us bid_0_size deltas at 1Hz here; using a coarse proxy = signed trade flow / total depth.
    # Caller can override if it tracks bid_0_size in the intra cache.
    ofi_proxy = (buy_size - sell_size) / (tot_d5 + 1e-9) if tot_d5 > 0 else 0.0

    return {
        "bar_idx": (bar_start.hour * 3600 + bar_start.minute * 60) // BAR_SECONDS,
        "bar_open_ts": bar_start.isoformat(),
        "bar_close_ts": bar_end.isoformat(),
        "date": bar_start.strftime("%Y-%m-%d"),
        "symbol": SYMBOL,
        "mid_open": mid_open, "mid_close": mid_close, "mid_high": mid_high, "mid_low": mid_low,
        "rv_bar_bp": rv_bar_bp,
        "vel_mean_bp": vel_mean_bp, "vel_abs_mean_bp": vel_abs_mean_bp,
        "spread_bp_mean": spread_bp, "spread_bp_max": spread_bp, "spread_bp_last": spread_bp,
        "micro_dev_bp_mean": micro_dev_bp_last, "micro_dev_bp_last": micro_dev_bp_last,
        "obi1_last": obi1, "obi5_mean": obi5_mean, "obi5_last": obi5, "obi5_std": obi5_std,
        "obi10_last": obi10, "obi20_last": obi20, "obi50_last": obi50, "obi5_step_std": obi5_step_std,
        "tot_d5_mean": tot_d5, "tot_d10_mean": tot_d10, "tot_d20_mean": tot_d20, "tot_d50_mean": tot_d50,
        "bid_d5_last": depth_at(bids, 5), "ask_d5_last": depth_at(asks, 5),
        "bid_d50_last": depth_at(bids, 50), "ask_d50_last": depth_at(asks, 50),
        "concentration_5_50_mean": concentration_5_50, "depth_asym_50": depth_asym_50,
        "bid_sl5_mean": bid_sl5, "ask_sl5_mean": ask_sl5,
        "bid_sl10_mean": bid_sl10, "ask_sl10_mean": ask_sl10,
        "bid_sl20_mean": bid_sl20, "ask_sl20_mean": ask_sl20,
        "bid_wall_mean": bid_wall, "ask_wall_mean": ask_wall,
        "ofi_proxy": ofi_proxy,
        "n_updates": intra_used,
        "return_5m_bar_bp": (np.log(mid_close) - np.log(mid_open)) * 10000 if mid_open > 0 else 0,
        # Trade aggregates
        "tr_count": n_trades, "tr_buy_count": len(buys), "tr_sell_count": len(sells),
        "tr_buy_size": buy_size, "tr_sell_size": sell_size,
        "tr_total_notional": total_notional, "tr_buy_notional": buy_notional, "tr_sell_notional": sell_notional,
        "tr_vwap": tr_vwap, "tr_large_count": tr_large_count,
        "tr_plus_count": 0, "tr_minus_count": 0,   # WS publicTrade doesn't expose tickDirection
        "tr_max_size": tr_max_size, "tr_size_p95_in_bar": p95,
        "tr_net_size": buy_size - sell_size, "tr_net_notional": buy_notional - sell_notional,
        "tr_tick_imb": 0.0,
        "tr_intensity": n_trades / BAR_SECONDS,
        "tr_buy_ratio": buy_size / (buy_size + sell_size) if (buy_size + sell_size) > 0 else 0.5,
        "tr_vwap_bp_dev": tr_vwap_bp_dev,
    }


# ============== Connection manager ==============
class LiveFeed:
    """Manages WS connection + bar emission. Thread-safe."""
    def __init__(self, log):
        self.log = log
        self.ob = OrderBookState()
        self.trades = TradeBuffer()
        self.intra = IntraBarCache()
        self.ws: websocket.WebSocketApp = None
        self.connected = False
        self.last_msg_ts = time.time()
        self.reconnect_count_today = 0
        self.last_reconnect_day = None
        self.snapshot_received = False
        self.snapshot_received_ts = 0.0   # for reconnect-grace window
        self.bar_thread = None
        self.shutdown_evt = threading.Event()

    def _push_intra_sample(self, ts_ms: int):
        """Push the current OB state into intra cache, keyed by wall-clock second.
        Called from on_message after each orderbook update — matches backtest's
        per-sec-last compression (quote-saver keeps the last event of each second).
        """
        if not self.snapshot_received:
            return
        snap = self.ob.snapshot_top(5)
        if not snap["bids"] or not snap["asks"]:
            return
        mid = (snap["bids"][0][0] + snap["asks"][0][0]) / 2
        b5 = sum(s for _, s in snap["bids"][:5])
        a5 = sum(s for _, s in snap["asks"][:5])
        obi5 = (b5 - a5) / (b5 + a5) if (b5 + a5) > 0 else 0.0
        ts_sec = ts_ms // 1000   # event's own ms timestamp → sec
        # IntraBarCache.sample() already overwrites same-sec entries with the latest
        # (matching backtest's "last event per second" rule). De-dup is built-in.
        self.intra.sample(ts_sec, mid, obi5)

    def on_open(self, ws):
        self.connected = True
        # ⚠️ subscribe orderbook.200 (not .50) — backtest source (quote-saver) uses
        # orderbook.200 stream; .50 and .200 are SEPARATE channels with DIFFERENT
        # publish events. Mismatch was source of ~$1 systematic mid bias.
        sub_msg = json.dumps({"op": "subscribe", "args": [
            f"orderbook.200.{SYMBOL}",
            f"publicTrade.{SYMBOL}",
        ]})
        ws.send(sub_msg)
        self.log.info(f"  [WS] connected → subscribed orderbook.200 + publicTrade.{SYMBOL}")

    def on_message(self, ws, msg):
        self.last_msg_ts = time.time()
        try:
            d = json.loads(msg)
        except Exception:
            return
        if "topic" not in d:
            # Subscribe ack or pong
            if d.get("op") == "subscribe":
                self.log.info(f"  [WS] subscribe ack: success={d.get('success')}")
            return
        topic = d["topic"]
        data = d.get("data", {})
        ts = d.get("ts", int(time.time()*1000))
        if topic.startswith("orderbook"):
            update_type = d.get("type", "delta")
            bids = data.get("b", []); asks = data.get("a", [])
            u = data.get("u", 0)
            if update_type == "snapshot":
                self.ob.apply_snapshot(bids, asks, u, ts)
                self.snapshot_received = True
                # Mark reconnect grace start for n-sec drop window
                self.snapshot_received_ts = time.time()
                self.log.info(f"  [WS] OB snapshot: u={u}, bids={len(bids)}, asks={len(asks)}")
            elif self.snapshot_received:
                self.ob.apply_delta(bids, asks, u, ts)
            # On EVERY orderbook event, push to intra cache (per-sec last via overwrite).
            # This replaces the old 1-Hz sampler_loop thread.
            self._push_intra_sample(ts)
        elif topic.startswith("publicTrade"):
            for t in (data if isinstance(data, list) else [data]):
                side = t.get("S", "Buy")  # Buy|Sell
                price = float(t.get("p", 0))
                size = float(t.get("v", 0))
                t_ts = int(t.get("T", ts))
                self.trades.add(t_ts, side, price, size)

    def on_error(self, ws, err):
        self.log.error(f"  [WS] error: {err}")

    def on_close(self, ws, code, msg):
        self.connected = False
        self.snapshot_received = False
        self.log.warning(f"  [WS] closed: code={code} msg={msg}")
        # Discord warning for excess reconnects today
        if dn and self.reconnect_count_today >= 5:
            dn.warning(f"WS reconnect #{self.reconnect_count_today} today",
                        f"Closed: code={code} msg={msg}. Threshold exceeded; investigate stability.")

    def connect_loop(self):
        backoff = 1.0
        while not self.shutdown_evt.is_set():
            try:
                today = datetime.now(timezone.utc).date()
                if today != self.last_reconnect_day:
                    self.reconnect_count_today = 0
                    self.last_reconnect_day = today
                self.reconnect_count_today += 1
                self.log.info(f"  [WS] connecting (reconnect#{self.reconnect_count_today} today)")
                self.ws = websocket.WebSocketApp(WS_URL,
                    on_open=self.on_open, on_message=self.on_message,
                    on_error=self.on_error, on_close=self.on_close)
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                self.log.error(f"  [WS] connect_loop exception: {e}")
            if self.shutdown_evt.is_set():
                break
            self.log.info(f"  [WS] reconnect in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_SEC)

    def is_stale(self) -> bool:
        return (time.time() - self.last_msg_ts) > STALE_THRESHOLD_SEC

    def shutdown(self):
        self.shutdown_evt.set()
        if self.ws:
            self.ws.close()


# ============== Bar emitter loop ==============
def bar_loop(feed: LiveFeed, log):
    """Every BAR_SECONDS at the boundary, emit a bar to disk."""
    last_bar_start = None
    while not feed.shutdown_evt.is_set():
        now = datetime.now(timezone.utc)
        sec_of_day = now.hour * 3600 + now.minute * 60 + now.second
        bar_idx = sec_of_day // BAR_SECONDS
        bar_start = now.replace(microsecond=0) - timedelta(seconds=sec_of_day % BAR_SECONDS)
        if last_bar_start is None:
            last_bar_start = bar_start
        # When a new bar starts, emit the previous bar
        if bar_start > last_bar_start and feed.snapshot_received and not feed.is_stale():
            prev_end = bar_start
            prev_start = last_bar_start
            try:
                ob_snap = feed.ob.snapshot_top(50)
                trades = feed.trades.consume_window(prev_start.timestamp(), prev_end.timestamp())
                intra = feed.intra.drain_window(int(prev_start.timestamp()), int(prev_end.timestamp()))
                # Reconnect grace: if WS reconnected during this bar, drop it
                grace_active = (time.time() - feed.snapshot_received_ts) < 60.0 and \
                                feed.snapshot_received_ts > prev_start.timestamp()
                bar = build_bar(ob_snap, trades, intra, prev_start, prev_end)
                if bar is None:
                    log.warning(f"  [BAR] {prev_start.isoformat()} SKIPPED (empty OB)")
                elif grace_active:
                    log.warning(f"  [BAR] {prev_start.isoformat()} SKIPPED (reconnect grace, snapshot ts={feed.snapshot_received_ts})")
                else:
                    # n_updates gate: backtest has ~300; below 250 = unreliable
                    n_upd = int(bar.get("n_updates", 0))
                    bar["partial"] = (n_upd < 250)
                    date_str = prev_start.strftime("%Y-%m-%d")
                    out_path = PERSIST_DIR / f"{date_str}.parquet"
                    if out_path.exists():
                        prev = pd.read_parquet(out_path)
                        new = pd.concat([prev, pd.DataFrame([bar])], ignore_index=True)
                    else:
                        new = pd.DataFrame([bar])
                    new.to_parquet(out_path, compression="zstd", index=False)
                    label = "BAR" if not bar["partial"] else "BAR-PARTIAL"
                    log.info(f"  [{label}] {prev_start.isoformat()} mid={bar['mid_close']:.2f} n_upd={n_upd} trades={bar['tr_count']} → {out_path.name}")
            except Exception as e:
                log.error(f"  [BAR] failed: {type(e).__name__}: {e}")
            last_bar_start = bar_start
        time.sleep(1.0)


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_file = LOG_DIR / f"ws_{stamp}.log"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger()
    log.info(f"=== Bybit V5 WS live feed (ETHUSDT linear perp) ===")
    log.info(f"  URL: {WS_URL}")
    log.info(f"  Bar persist dir: {PERSIST_DIR}")
    log.info(f"  Log: {log_file}")

    feed = LiveFeed(log)

    def sig_handler(signum, frame):
        log.info("SIGINT/SIGTERM — shutting down...")
        feed.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Connection thread
    conn_thread = threading.Thread(target=feed.connect_loop, daemon=True)
    conn_thread.start()

    # NOTE: previous design used a separate 1Hz sampler thread. Removed —
    # cache is now populated directly inside on_message (per-update push,
    # de-duplicated by second via IntraBarCache.sample overwrite).
    # See Update 2026-05-25 memory note on Gate 1B root cause.

    # Bar emitter thread
    bar_thread = threading.Thread(target=bar_loop, args=(feed, log), daemon=True)
    bar_thread.start()

    # Main loop: just monitor and log periodic status
    while True:
        time.sleep(30)
        status = "CONNECTED" if feed.connected else "DISCONNECTED"
        stale = "STALE" if feed.is_stale() else "fresh"
        log.info(f"  [STATUS] {status}/{stale}  reconnects_today={feed.reconnect_count_today}  "
                 f"OB u={feed.ob.last_update_id}  last_trade_ts={feed.trades.last_trade_ts:.0f}")


if __name__ == "__main__":
    main()
