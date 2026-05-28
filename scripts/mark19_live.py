"""
mark19 live trading bot — 4h Direction strategy.

ONE codebase, two modes:
  MODE = "shadow"  → log intended orders, NO real execution (default)
  MODE = "live"    → real orders via Bybit (requires API key + explicit confirmation)

Strategy (validated 2026-05-25):
  - ETHUSDT perpetual, 4h cycle, hold 48 5-min bars
  - Long features (mom_1d, rv_1d, cumflow_1d, dist_ma_1d, ...)
  - Model: ~/mark19_data/models_prod/4h_direction_v2.joblib (XGB)
  - Entry: thr |p-0.5| > 0.05 → maker limit at top-of-book
  - Fallback: queue sim showed pure-taker still positive (Sh +2.48)
  - Expected: ~0.9-2.2 trades/day, +46 bp/day with 38% maker fill

Risk rails (NON-NEGOTIABLE):
  - MAX_POSITION_SIZE_ETH = 0.01     (start with smallest viable)
  - MAX_CONCURRENT_POSITIONS = 1
  - DAILY_LOSS_LIMIT_PCT = -3.0       (stop trading for the day)
  - MAX_DRAWDOWN_KILL_PCT = -10.0     (full halt, manual restart)
  - MIN_BALANCE_USDT = 50             (refuse if wallet too small)
  - PRICE_SANITY_PCT = 0.5            (reject limit > mark ±0.5%)
  - LIVE_FIRST_TRADE_FLAG = required  (must be set explicitly)

Kill criteria (auto-halt):
  - Feature distribution drift > 2σ vs training (regime shift)
  - Estimated fill rate < 20% over 5 days
  - Daily loss limit hit
  - Trade frequency outside [0.3, 5] / day
  - WebSocket disconnected > 60s during decision window
  - Reconciliation mismatch with exchange (position/balance/orders)
  - 3 consecutive order rejections / exchange errors

Reconciliation (testnet 대체 — primary defense in live):
  - Every cycle (pre-order, post-order, periodic): fetch exchange truth
    * position list (size, side, avgPrice, unrealisedPnl)
    * wallet balance (available, usedMargin, totalEquity)
    * open orders (orderId, side, qty, price, status)
  - Compare against internal book; ANY mismatch → halt
  - Exchange values are AUTHORITATIVE; internal book is reconstructed/corrected
  - Tolerances: size ±0.0001 ETH, price ±$0.05, balance ±$0.10
  - First N (=5) live fills require manual confirmation (env: MARK19_MANUAL_CONFIRM=1)
  - Position size violates rail
  - 3 consecutive order rejections

Configuration via env vars (override defaults):
  MARK19_MODE = shadow | live           (default: shadow)
  MARK19_MODEL_PATH = <path>            (default: ~/mark19_data/models_prod/4h_direction_v2.joblib)
  MARK19_BYBIT_KEY = <api key>          (live only)
  MARK19_BYBIT_SECRET = <secret>        (live only)
  MARK19_LIVE_CONFIRM = "I_HAVE_READ_THE_RAILS_AND_ACCEPT_LOSS_RISK"
"""
from __future__ import annotations
import argparse, json, logging, os, signal, sys, time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============== Configuration ==============
SYMBOL = "ETHUSDT"
BAR_SECONDS = 300                          # 5-min bar
HOLD_BARS = 48                             # 4h hold
SIGNAL_THRESHOLD = 0.05                    # |p-0.5| > 0.05
DEFAULT_MODE = os.environ.get("MARK19_MODE", "shadow")
MODEL_PATH = Path(os.environ.get("MARK19_MODEL_PATH",
                                  "/Users/mark/mark19_data/models_prod/4h_direction_v2.joblib"))
LOG_DIR = Path("/Users/mark/mark19_data/live_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
BUFFER_BARS = 2100                         # ~7.3 days (need ≥7d for long features)

# Risk rails (hard limits)
MAX_POSITION_SIZE_ETH = 0.01
MAX_CONCURRENT_POSITIONS = 1
DAILY_LOSS_LIMIT_PCT = -3.0
MAX_DRAWDOWN_KILL_PCT = -10.0
MIN_BALANCE_USDT = 50.0
PRICE_SANITY_PCT = 0.5

# Leverage controls (validated 2026-05-25 — backtest assumed 1x, no liquidation model)
# Historical analysis on 1198 ETH days: at 3x leverage, ZERO 4h liquidation events occurred.
# But backtest does NOT model intra-hold liquidation, so verification phase = 1x only.
LEVERAGE_DEFAULT = 1                              # validation phase enforced 1x
LEVERAGE_MAX_ALLOWED = 1                          # raise only after live validation
LIQUIDATION_BUFFER_BP = {1: 5000, 3: 3300, 5: 2000, 10: 1000, 20: 500}  # rough liq distance per leverage

# Bybit fees (standard VIP0)
FEE_MAKER_BP = 2.0
FEE_TAKER_BP = 5.5


# ============== Risk Rail ==============
class RiskRail:
    """Hard safety limits. Any violation → block order or halt."""
    def __init__(self, mode: str, log):
        self.mode = mode
        self.log = log
        self.daily_pnl_usdt = 0.0
        self.peak_balance = 0.0
        self.current_balance = 0.0
        self.day_start = None
        self.halted = False
        self.halt_reason = None
        self.consec_rejections = 0
        self.fills_24h: deque = deque(maxlen=200)

    def reset_day(self, ts: datetime):
        self.daily_pnl_usdt = 0.0
        self.day_start = ts
        self.log.info(f"  [RISK] day reset at {ts.isoformat()}")

    def update_balance(self, balance_usdt: float):
        self.current_balance = balance_usdt
        if balance_usdt > self.peak_balance:
            self.peak_balance = balance_usdt
        # Max drawdown
        if self.peak_balance > 0:
            dd_pct = (balance_usdt - self.peak_balance) / self.peak_balance * 100
            if dd_pct <= MAX_DRAWDOWN_KILL_PCT:
                self.halt(f"max drawdown {dd_pct:.1f}% ≤ {MAX_DRAWDOWN_KILL_PCT}%")

    def record_fill(self, pnl_usdt: float, ts: datetime):
        self.daily_pnl_usdt += pnl_usdt
        self.fills_24h.append((ts, pnl_usdt))
        if self.current_balance > 0:
            daily_pct = self.daily_pnl_usdt / self.current_balance * 100
            if daily_pct <= DAILY_LOSS_LIMIT_PCT:
                self.halt(f"daily loss {daily_pct:.2f}% ≤ {DAILY_LOSS_LIMIT_PCT}%")

    def halt(self, reason: str):
        if not self.halted:
            self.halted = True
            self.halt_reason = reason
            self.log.error(f"  [RISK] 🛑 HALT: {reason}")

    def check_order(self, side: str, size: float, limit_price: float,
                    mark_price: float, current_positions: int,
                    leverage: int = LEVERAGE_DEFAULT) -> tuple[bool, str]:
        """Return (allowed, reason). False blocks order."""
        if self.halted:
            return False, f"already halted: {self.halt_reason}"
        if size > MAX_POSITION_SIZE_ETH:
            return False, f"size {size} > MAX {MAX_POSITION_SIZE_ETH}"
        if current_positions >= MAX_CONCURRENT_POSITIONS:
            return False, f"positions {current_positions} >= MAX {MAX_CONCURRENT_POSITIONS}"
        if self.current_balance < MIN_BALANCE_USDT:
            return False, f"balance {self.current_balance:.2f} < MIN {MIN_BALANCE_USDT}"
        # Price sanity
        deviation_pct = abs(limit_price - mark_price) / mark_price * 100
        if deviation_pct > PRICE_SANITY_PCT:
            return False, f"limit {limit_price} deviates {deviation_pct:.2f}% from mark {mark_price}"
        # Leverage gate: refuse anything > LEVERAGE_MAX_ALLOWED unless explicitly upgraded
        if leverage > LEVERAGE_MAX_ALLOWED:
            return False, f"leverage {leverage}x > MAX_ALLOWED {LEVERAGE_MAX_ALLOWED}x (validation phase)"
        return True, "ok"

    def estimate_liquidation_price(self, side: str, entry_price: float, leverage: int) -> float:
        """Rough liquidation price for a single isolated-margin position.
        Bybit uses ~80% of maintenance margin as buffer; approx liquidation at
        ~1/leverage adverse move minus maintenance margin (~0.5-1% for ETH).
        Conservative estimate: liquidation at (1/leverage - 0.01) adverse move.
        """
        adverse_pct = (1.0 / leverage) - 0.01     # subtract maintenance margin buffer
        if side.upper() in ("LONG", "BUY"):
            return entry_price * (1.0 - adverse_pct)
        else:
            return entry_price * (1.0 + adverse_pct)


# ============== Exchange Adapter (shadow/live unified interface) ==============
@dataclass
class ExchangeState:
    """Snapshot of authoritative exchange state."""
    ts: str
    balance_usdt: float            # available
    total_equity: float
    used_margin: float
    positions: list                # [{'symbol', 'side', 'size', 'avgPrice', 'unrealisedPnl'}]
    open_orders: list              # [{'orderId', 'symbol', 'side', 'qty', 'price', 'status'}]


class ExchangeAdapter:
    """Unified interface. Shadow returns synthetic state; Live calls Bybit REST."""
    def __init__(self, mode: str, log):
        self.mode = mode
        self.log = log
        # Shadow ledger (simulated state)
        self._shadow_balance = 1000.0  # mock starting balance
        self._shadow_positions = []
        self._shadow_orders = []
        self._shadow_next_id = 1
        # Live API (lazy init)
        self._live_session = None

    def fetch_state(self) -> ExchangeState:
        """Return authoritative state. Source of truth."""
        if self.mode == "shadow":
            return ExchangeState(
                ts=datetime.now(timezone.utc).isoformat(),
                balance_usdt=self._shadow_balance,
                total_equity=self._shadow_balance + sum(p.get('unrealisedPnl', 0) for p in self._shadow_positions),
                used_margin=sum(p['size'] * p['avgPrice'] for p in self._shadow_positions),
                positions=list(self._shadow_positions),
                open_orders=list(self._shadow_orders),
            )
        else:
            return self._fetch_live_state()

    def _fetch_live_state(self) -> ExchangeState:
        """Bybit V5 REST via verified bybit_v5 wrapper. Returns ExchangeState in
        Reconciler-compatible format. Raises if API key missing or HTTP fails."""
        # Lazy import to avoid pulling bybit_v5 in shadow-only runs
        from pathlib import Path
        import bybit_v5

        # Load .env if env vars not already set
        key = os.environ.get("MARK19_BYBIT_KEY")
        secret = os.environ.get("MARK19_BYBIT_SECRET")
        if not key or not secret:
            env_path = Path("/Users/mark/Desktop/Mark/mark19/live_bot/.env")
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        k = k.strip(); v = v.strip().strip("'\"")
                        if k == "BYBIT_API_KEY" and not key: key = v
                        elif k == "BYBIT_API_SECRET" and not secret: secret = v
        if not key or not secret:
            raise RuntimeError("Bybit API key/secret unavailable (env or live_bot/.env)")

        out = bybit_v5.fetch_state_and_map(key, secret, SYMBOL)
        m = out["mapped"]
        if m.get("_mapping_warnings"):
            self.log.warning(f"  [V5] mapping warnings: {m['_mapping_warnings']}")
        return ExchangeState(
            ts=m["ts"],
            balance_usdt=float(m["balance_usdt"] or 0),
            total_equity=float(m["total_equity"] or 0),
            used_margin=float(m["used_margin"] or 0),
            positions=m["positions"],
            open_orders=m["open_orders"],
        )

    def place_limit(self, side: str, qty: float, price: float) -> dict:
        """Submit limit order. Returns {orderId, status, ...} or raises."""
        if self.mode == "shadow":
            oid = f"shadow-{self._shadow_next_id}"
            self._shadow_next_id += 1
            order = {'orderId': oid, 'symbol': SYMBOL, 'side': side,
                     'qty': qty, 'price': price, 'status': 'New'}
            self._shadow_orders.append(order)
            return order
        raise NotImplementedError("Live place_limit — implement Bybit V5 /v5/order/create")

    def cancel_order(self, order_id: str) -> dict:
        if self.mode == "shadow":
            self._shadow_orders = [o for o in self._shadow_orders if o['orderId'] != order_id]
            return {'orderId': order_id, 'status': 'Cancelled'}
        raise NotImplementedError("Live cancel — Bybit V5 /v5/order/cancel")

    # Shadow-only helpers (for testing/simulation)
    def _shadow_simulate_fill(self, order_id: str, fill_price: float, fee_bp: float = 2.0):
        """Mark a shadow order as filled and update position."""
        order = next((o for o in self._shadow_orders if o['orderId'] == order_id), None)
        if not order: return
        self._shadow_orders.remove(order)
        side = order['side']; qty = order['qty']
        sign = 1 if side == 'Buy' else -1
        fee_usdt = qty * fill_price * fee_bp / 10000
        self._shadow_balance -= fee_usdt
        # Net or open position
        existing = next((p for p in self._shadow_positions if p['symbol'] == SYMBOL), None)
        if existing is None:
            self._shadow_positions.append({
                'symbol': SYMBOL, 'side': side, 'size': qty,
                'avgPrice': fill_price, 'unrealisedPnl': 0.0
            })
        else:
            # Same side → average; opposite side → reduce
            if existing['side'] == side:
                new_size = existing['size'] + qty
                existing['avgPrice'] = (existing['avgPrice'] * existing['size'] + fill_price * qty) / new_size
                existing['size'] = new_size
            else:
                if qty >= existing['size']:
                    remaining = qty - existing['size']
                    self._shadow_positions.remove(existing)
                    if remaining > 1e-9:
                        self._shadow_positions.append({
                            'symbol': SYMBOL, 'side': side, 'size': remaining,
                            'avgPrice': fill_price, 'unrealisedPnl': 0.0
                        })
                else:
                    existing['size'] -= qty


# ============== Reconciler (testnet 대체 — primary live defense) ==============
@dataclass
class InternalBook:
    """Bot's internal record. Compared against exchange truth every cycle."""
    balance_usdt: float = 0.0
    positions: list = field(default_factory=list)      # [{'symbol','side','size','avgPrice'}]
    open_order_ids: set = field(default_factory=set)


class Reconciler:
    """Compare internal book to exchange state. Mismatch → halt."""
    SIZE_TOL = 0.0001       # ETH
    PRICE_TOL = 0.05        # USD
    BALANCE_TOL = 0.10      # USDT

    def __init__(self, log, rail):
        self.log = log
        self.rail = rail
        self.last_check_ts = None
        self.mismatch_count = 0

    def check(self, internal: InternalBook, exchange: ExchangeState) -> tuple[bool, list[str]]:
        """Return (ok, mismatches). If mismatches present, halt rail."""
        diffs = []
        # 1. Balance
        if abs(internal.balance_usdt - exchange.balance_usdt) > self.BALANCE_TOL:
            diffs.append(f"BALANCE: internal={internal.balance_usdt:.4f}, exchange={exchange.balance_usdt:.4f}")

        # 2. Position count
        if len(internal.positions) != len(exchange.positions):
            diffs.append(f"POSITION_COUNT: internal={len(internal.positions)}, exchange={len(exchange.positions)}")

        # 3. Position details (size, side, avgPrice)
        ex_by_sym = {p['symbol']: p for p in exchange.positions}
        int_by_sym = {p['symbol']: p for p in internal.positions}
        all_syms = set(ex_by_sym) | set(int_by_sym)
        for sym in all_syms:
            ep = ex_by_sym.get(sym); ip = int_by_sym.get(sym)
            if ep and not ip:
                diffs.append(f"POSITION {sym}: exchange has {ep['side']} {ep['size']} @ {ep['avgPrice']}; internal NONE")
            elif ip and not ep:
                diffs.append(f"POSITION {sym}: internal has {ip['side']} {ip['size']} @ {ip['avgPrice']}; exchange NONE")
            else:
                if ep['side'] != ip['side']:
                    diffs.append(f"POSITION {sym} SIDE: internal={ip['side']}, exchange={ep['side']}")
                if abs(ep['size'] - ip['size']) > self.SIZE_TOL:
                    diffs.append(f"POSITION {sym} SIZE: internal={ip['size']}, exchange={ep['size']}")
                if abs(ep['avgPrice'] - ip['avgPrice']) > self.PRICE_TOL:
                    diffs.append(f"POSITION {sym} avgPrice: internal={ip['avgPrice']:.2f}, exchange={ep['avgPrice']:.2f}")

        # 4. Open orders: any tracked ID that disappeared OR untracked exchange-side order
        ex_ids = {o['orderId'] for o in exchange.open_orders}
        disappeared = internal.open_order_ids - ex_ids
        untracked = ex_ids - internal.open_order_ids
        if disappeared:
            diffs.append(f"ORDERS disappeared (filled/cancelled silently?): {sorted(disappeared)}")
        if untracked:
            diffs.append(f"ORDERS untracked (external/duplicate?): {sorted(untracked)}")

        self.last_check_ts = exchange.ts
        if diffs:
            self.mismatch_count += 1
            self.log.error(f"  [RECONCILE] 🚨 {len(diffs)} mismatch(es):")
            for d in diffs:
                self.log.error(f"    - {d}")
            self.rail.halt(f"reconciliation mismatch ({len(diffs)} diffs at {exchange.ts})")
            return False, diffs
        self.log.info(f"  [RECONCILE] ✓ aligned (balance ${exchange.balance_usdt:.2f}, pos={len(exchange.positions)}, orders={len(exchange.open_orders)})")
        return True, []


# ============== Order Manager (decision → exchange, with safety) ==============
class OrderManager:
    def __init__(self, exchange: ExchangeAdapter, rail: RiskRail, reconciler: Reconciler,
                 internal: InternalBook, log, manual_confirm_first: int = 5):
        self.exchange = exchange
        self.rail = rail
        self.reconciler = reconciler
        self.internal = internal
        self.log = log
        self.manual_confirm_remaining = manual_confirm_first
        self.consecutive_rejections = 0

    def place(self, decision: 'Decision', mark_price: float):
        """Execute decision: reconcile → rail → confirm → place → log."""
        if decision.action == "SKIP":
            self.log.info(f"  [ORDER] SKIP (conf={decision.confidence:.4f})")
            return None
        # 1. Pre-order reconciliation
        ex_state = self.exchange.fetch_state()
        self.rail.update_balance(ex_state.balance_usdt)
        ok, diffs = self.reconciler.check(self.internal, ex_state)
        if not ok:
            self.log.error(f"  [ORDER] BLOCKED by reconcile pre-check")
            return None
        # 2. Rail check (includes leverage validation)
        ok, reason = self.rail.check_order(decision.action, decision.size,
                                             decision.limit_price, mark_price,
                                             len(ex_state.positions),
                                             leverage=LEVERAGE_DEFAULT)
        if not ok:
            self.log.error(f"  [ORDER] BLOCKED by rail: {reason}")
            self.consecutive_rejections += 1
            if self.consecutive_rejections >= 3:
                self.rail.halt(f"{self.consecutive_rejections} consecutive rejections")
            return None
        # 2b. Log estimated liquidation price (defense-in-depth visibility)
        liq_price = self.rail.estimate_liquidation_price(decision.action, decision.limit_price, LEVERAGE_DEFAULT)
        liq_distance_pct = abs(liq_price - decision.limit_price) / decision.limit_price * 100
        self.log.info(f"  [LIQ] est liquidation @ {liq_price:.2f} ({liq_distance_pct:.1f}% away)  leverage={LEVERAGE_DEFAULT}x")
        # 3. Manual confirmation (first N live fills)
        if self.exchange.mode == "live" and self.manual_confirm_remaining > 0:
            confirm = os.environ.get("MARK19_MANUAL_CONFIRM", "0")
            if confirm != "1":
                self.log.warning(f"  [ORDER] LIVE first-{self.manual_confirm_remaining}-fills require MARK19_MANUAL_CONFIRM=1")
                return None
            self.log.warning(f"  [ORDER] LIVE manual-confirmed ({self.manual_confirm_remaining} remaining)")
            self.manual_confirm_remaining -= 1
        # 4. Place
        try:
            side = "Buy" if decision.action == "LONG" else "Sell"
            order = self.exchange.place_limit(side, decision.size, decision.limit_price)
            self.internal.open_order_ids.add(order['orderId'])
            self.consecutive_rejections = 0
            self.log.info(f"  [ORDER] {side} {decision.size} @ {decision.limit_price} → orderId={order['orderId']}")
            return order
        except Exception as e:
            self.log.error(f"  [ORDER] FAILED: {type(e).__name__}: {e}")
            self.consecutive_rejections += 1
            if self.consecutive_rejections >= 3:
                self.rail.halt(f"{self.consecutive_rejections} consecutive errors: {e}")
            return None

    def update_after_fill(self, order_id: str, fill_price: float, fill_qty: float, side: str):
        """Post-fill: update internal book + record on rail + reconcile."""
        self.internal.open_order_ids.discard(order_id)
        # Update internal position (mirror what we expect exchange to record)
        existing = next((p for p in self.internal.positions if p['symbol'] == SYMBOL), None)
        if existing is None:
            self.internal.positions.append({'symbol': SYMBOL, 'side': side,
                                              'size': fill_qty, 'avgPrice': fill_price})
        else:
            if existing['side'] == side:
                new_size = existing['size'] + fill_qty
                existing['avgPrice'] = (existing['avgPrice']*existing['size'] + fill_price*fill_qty) / new_size
                existing['size'] = new_size
            else:
                if fill_qty >= existing['size']:
                    self.internal.positions.remove(existing)
                else:
                    existing['size'] -= fill_qty
        # Post-fill reconcile
        ex_state = self.exchange.fetch_state()
        self.reconciler.check(self.internal, ex_state)


# ============== Bar Buffer ==============
class BarBuffer:
    """Ring buffer of recent 5-min bars; computes long features causally on demand."""
    def __init__(self, max_bars: int = BUFFER_BARS):
        self.bars: list[dict] = []
        self.max_bars = max_bars

    def add_bar(self, bar: dict):
        self.bars.append(bar)
        if len(self.bars) > self.max_bars:
            self.bars = self.bars[-self.max_bars:]

    def to_df(self) -> pd.DataFrame:
        if not self.bars:
            return pd.DataFrame()
        df = pd.DataFrame(self.bars)
        df['bar_open_ts'] = pd.to_datetime(df['bar_open_ts'])
        df['date'] = df['bar_open_ts'].dt.floor('D')
        df = df.sort_values('bar_open_ts').reset_index(drop=True)
        return df

    def warm_up_from_disk(self, path: Path):
        """Seed buffer with historical bars before live start."""
        if not path.exists():
            return 0
        files = sorted(path.glob("*.parquet"))[-15:]  # last ~15 days, enough for 7d rolling
        loaded = 0
        for f in files:
            try:
                d = pd.read_parquet(f)
                for _, row in d.iterrows():
                    self.bars.append(row.to_dict())
                loaded += len(d)
            except Exception as e:
                pass
        self.bars = self.bars[-self.max_bars:]
        return loaded


def compute_long_features(bars_df: pd.DataFrame) -> pd.DataFrame:
    """EXACT replica of backtest pipeline. DO NOT diverge — both must update together."""
    df = bars_df.copy().sort_values('bar_open_ts').reset_index(drop=True)
    SHORT_LAG = ['return_5m_bar_bp','obi5_last','micro_dev_bp_last','rv_bar_bp','ofi_proxy','tr_net_size','tr_tick_imb']
    for c in SHORT_LAG:
        if c not in df.columns: continue
        for k in [1,3,6,12]:
            df[f'{c}_lag{k}'] = df[c].shift(k)
    for N,lbl in [(12,'1h'),(48,'4h'),(288,'1d'),(2016,'7d')]:
        df[f'mom_{lbl}_bp'] = (np.log(df.mid_close) - np.log(df.mid_close.shift(N)))*10000
        mma = df.mid_close.rolling(N, min_periods=N//2).mean()
        df[f'dist_ma_{lbl}_bp'] = (df.mid_close - mma) / mma * 10000
        df[f'rv_{lbl}_bp'] = df.return_5m_bar_bp.rolling(N, min_periods=N//2).std()
    for N,lbl in [(48,'4h'),(288,'1d'),(2016,'7d')]:
        df[f'obi5_ma_{lbl}'] = df.obi5_mean.rolling(N, min_periods=N//2).mean()
        df[f'cumflow_{lbl}'] = df.tr_net_size.rolling(N, min_periods=N//2).sum()
        df[f'buyratio_{lbl}'] = df.tr_buy_ratio.rolling(N, min_periods=N//2).mean()
        df[f'spread_ma_{lbl}'] = df.spread_bp_mean.rolling(N, min_periods=N//2).mean()
    return df


# ============== Strategy Decision ==============
@dataclass
class Decision:
    ts: str
    p_up: float
    confidence: float
    action: str           # "LONG" | "SHORT" | "SKIP"
    limit_price: Optional[float]
    size: float
    mode: str
    mid: float
    best_bid: float
    best_ask: float
    reason: str = ""

def make_decision(model_artifact, feature_row: dict, mid: float, best_bid: float,
                   best_ask: float, ts: datetime, mode: str) -> Decision:
    feature_cols = model_artifact['feature_columns']
    medians = model_artifact['train_medians']
    # Build input array in EXACT order
    x = []
    for c in feature_cols:
        v = feature_row.get(c, np.nan)
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            v = medians.get(c, 0.0)
        x.append(float(v))
    X = np.array([x], dtype=np.float32)
    # Force ALL trained trees (predict default uses best_iteration → underfitted on v1).
    _n_trees = model_artifact['model'].get_booster().num_boosted_rounds()
    p_up = float(model_artifact['model'].predict_proba(X, iteration_range=(0, _n_trees))[0, 1])
    conf = abs(p_up - 0.5)
    if conf <= SIGNAL_THRESHOLD:
        return Decision(ts.isoformat(), p_up, conf, "SKIP", None, 0.0, mode, mid, best_bid, best_ask,
                        f"|p-0.5|={conf:.4f} ≤ {SIGNAL_THRESHOLD}")
    direction = "LONG" if p_up > 0.5 else "SHORT"
    limit = best_bid if direction == "LONG" else best_ask
    return Decision(ts.isoformat(), p_up, conf, direction, float(limit),
                    MAX_POSITION_SIZE_ETH, mode, mid, best_bid, best_ask,
                    f"confidence={conf:.4f}, maker @ top-of-book")


# ============== Logger ==============
def get_logger(log_file: Path):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    return logging.getLogger()


# ============== Main loop (skeleton) ==============
def main():
    import joblib
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=["shadow", "live"])
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--smoke-test", action="store_true",
                        help="One-shot decision on the last cached bars (no live data needed)")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_file = LOG_DIR / f"mark19_{args.mode}_{stamp}.log"
    log = get_logger(log_file)
    log.info(f"=== mark19 live ({args.mode} mode) ===")
    log.info(f"  Log: {log_file}")
    log.info(f"  Model: {args.model_path}")

    # === LIVE MODE GUARDS ===
    if args.mode == "live":
        confirm = os.environ.get("MARK19_LIVE_CONFIRM", "")
        if confirm != "I_HAVE_READ_THE_RAILS_AND_ACCEPT_LOSS_RISK":
            log.error("LIVE mode requires: export MARK19_LIVE_CONFIRM='I_HAVE_READ_THE_RAILS_AND_ACCEPT_LOSS_RISK'")
            sys.exit(2)
        if not os.environ.get("MARK19_BYBIT_KEY") or not os.environ.get("MARK19_BYBIT_SECRET"):
            log.error("LIVE mode requires MARK19_BYBIT_KEY and MARK19_BYBIT_SECRET env vars")
            sys.exit(2)
        log.warning("⚠️  LIVE MODE — real orders will be placed")
    else:
        log.info("SHADOW MODE — orders will be LOGGED ONLY, never sent")

    # Load model
    if not args.model_path.exists():
        log.error(f"Model not found: {args.model_path}"); sys.exit(2)
    artifact = joblib.load(args.model_path)
    log.info(f"  Model: {len(artifact['feature_columns'])} features, train AUC {artifact['config']['auc_train']:.4f}, val AUC {artifact['config']['auc_val']:.4f}")

    rail = RiskRail(args.mode, log)
    buf = BarBuffer()

    # === Smoke test: one-shot decision on last known v3 bar ===
    if args.smoke_test:
        log.info("\n=== SMOKE TEST: one-shot decision on last v3 bar ===")
        n_loaded = buf.warm_up_from_disk(Path("/Users/mark/mark19_data/bars_5min_v3/ETHUSDT"))
        log.info(f"  Warmed buffer with {n_loaded} bars")
        if n_loaded < 2016:
            log.warning(f"  Warm-up < 7 days ({n_loaded} bars); long features will have NaN. Need real backfill.")
            # Try larger backfill: load last 30 day-files for 7d+ coverage
            buf.bars = []  # clear
            files = sorted(Path("/Users/mark/mark19_data/bars_5min_v3/ETHUSDT").glob("*.parquet"))[-30:]
            for f in files:
                d = pd.read_parquet(f)
                for _, row in d.iterrows():
                    buf.bars.append(row.to_dict())
            buf.bars = buf.bars[-BUFFER_BARS:]
            log.info(f"  Re-warmed with {len(buf.bars)} bars (last {len(files)} files)")

        bars_df = buf.to_df()
        if len(bars_df) < 100:
            log.error("No bars to test"); return
        feats_df = compute_long_features(bars_df)
        last = feats_df.iloc[-1]
        log.info(f"  Last bar: {last['bar_open_ts']}, mid={last['mid_close']:.2f}")
        # Mock best_bid/ask = mid ± half-spread
        best_bid = float(last['mid_close']) - float(last['spread_bp_mean']) / 20000 * float(last['mid_close'])
        best_ask = float(last['mid_close']) + float(last['spread_bp_mean']) / 20000 * float(last['mid_close'])
        decision = make_decision(artifact, last.to_dict(), float(last['mid_close']),
                                  best_bid, best_ask, datetime.now(timezone.utc), args.mode)
        log.info(f"  Decision: {decision.action}  p_up={decision.p_up:.4f}  conf={decision.confidence:.4f}")
        if decision.action != "SKIP":
            log.info(f"  Limit price: ${decision.limit_price:.2f}  size: {decision.size} ETH")
            # Risk rail check (simulate balance)
            rail.update_balance(100.0)
            ok, reason = rail.check_order(decision.action, decision.size, decision.limit_price,
                                           float(last['mid_close']), 0)
            log.info(f"  Risk rail: {'PASS' if ok else 'BLOCK'} ({reason})")
        # Log decision as JSON
        decision_log = LOG_DIR / f"decisions_{stamp}.jsonl"
        with open(decision_log, 'a') as f:
            f.write(json.dumps(asdict(decision)) + "\n")
        log.info(f"  Decision logged: {decision_log}")
        return

    # === Live loop (to be wired to WebSocket — skeleton) ===
    log.info("""
=== LIVE LOOP — TO IMPLEMENT ===
  1. Subscribe Bybit WS: orderbook.50.ETHUSDT + publicTrade.ETHUSDT
  2. Maintain 1Hz orderbook state via deltas
  3. Build 5-min bars (mid OHLC, OBI, depth, slope, OFI, trades aggregates)
  4. Append to BarBuffer; persist to disk
  5. Every BAR_SECONDS, check if it's a HOLD_BARS boundary (decision time)
  6. If decision time:
     - Compute long features (compute_long_features)
     - make_decision(model, ...)
     - Log JSON
     - mode='shadow': stop here
     - mode='live': rail.check_order → if ok, send to Bybit REST (V5 /v5/order/create)
  7. Track open positions; exit after HOLD_BARS bars (or end-of-day flatten)
  8. Risk rail update on every fill

  See test/mark19_live_test.py for unit tests (to be written).
""")


if __name__ == "__main__":
    main()
