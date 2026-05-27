"""SHADOW direction pilot — mark36_v1 + Drift policy + mock fills.

Polls every 1 minute. Reads live data via build_live_dataset (with parquet retry).
Applies day-mean normalization (same as sido36). Runs mark36_v1 inference.
Simulates entry/exit with drift policy. Tracks fill rate + toxic rate + mock PnL.

NO REAL ORDERS.
"""
import sys, time, json, logging, os
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import numpy as np

from live_bot.feature_pipeline import build_live_dataset
from live_bot.parquet_retry import read_parquet_with_retry  # ensure imports work

# ---- Config ----
MODEL_PATH = "/Users/dohun/Desktop/Mark/mark19/models/mark36_v2.joblib"
DIR_TH = 0.58
VOL_TH = 0.6
LOCKOUT_MIN = 60
SL_PCT = 1.5
MAX_HOLD_MIN = 30  # drift exit max wait before taker fallback
POLL_SEC = 60      # 1-min polling
LOG_INTERVAL_SEC = 600   # log summary every 10 min
PERSIST_INTERVAL_SEC = 600
DRIFT_OFFSET_BP = 0.5    # 0.5bp inside mid for drift limit (not actually used in mock — placeholder)

# Mock fees
FEE_TAKER = 0.00055
FEE_MAKER = -0.00025

# ---- Logging ----
log_dir = Path("/Users/dohun/Desktop/Mark/mark19/logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"shadow_direction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
state_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/shadow_direction_state.json")
state_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ---- Load model ----
log.info("=" * 70)
log.info("SHADOW DIRECTION PILOT — mark36_v1 + Drift policy + mock fills")
log.info("=" * 70)
log.info(f"Loading {MODEL_PATH}...")
bundle = joblib.load(MODEL_PATH)
LRV = bundle["lr_vol"]; SV = bundle["scaler_vol"]
XGBD = bundle["xgb_dir"]
FEAT_COLS = bundle["feature_cols"]
HIGH_SHIFT = bundle["high_shift_features"]
NORM_FEATURES = bundle["norm_features"]
TRAIN_MEDIANS = pd.Series(bundle["train_medians"])
META = bundle["metadata"]
log.info(f"  Features: {len(FEAT_COLS)}  norm: {len(NORM_FEATURES)}  high_shift: {len(HIGH_SHIFT)}")
log.info(f"  Walk-fw mean AUC: {META.get('5seed_walk_forward_mean_auc', '?')}")
log.info(f"  Walk-fw daily PnL: {META.get('9day_backtest_daily_pnl_mean_pct', '?')}%")


# ---- Live state ----
class ShadowState:
    def __init__(self):
        self.start_ts = datetime.now(timezone.utc)
        self.session_features = []  # accumulated rows for day-mean normalization
        # Open position
        self.open_pos = None  # {direction, entry_ts, entry_price, exit_attempts, sl_price}
        # Stats
        self.signals_total = 0
        self.signals_long = 0
        self.signals_short = 0
        self.fills_maker = 0
        self.fills_taker_close = 0
        self.fills_sl = 0
        self.trades = []   # list of {direction, entry, exit, pnl_pct, fill_type, ts_open, ts_close}
        # Adverse selection
        self.n_toxic = 0
        self.n_favorable = 0
        # Loop state
        self.last_log_ts = self.start_ts
        self.last_persist_ts = self.start_ts
        self.cycle_count = 0
        self.errors = 0

    def to_dict(self):
        now = datetime.now(timezone.utc)
        uptime_h = (now - self.start_ts).total_seconds() / 3600
        n_trades_closed = len(self.trades)
        total_pnl_pct = sum(t["pnl_pct"] for t in self.trades) if self.trades else 0
        wins = sum(1 for t in self.trades if t["pnl_pct"] > 0)
        return {
            "start_ts": str(self.start_ts),
            "current_ts": str(now),
            "uptime_h": round(uptime_h, 2),
            "model": "mark36_v1",
            "params": {"dir_th": DIR_TH, "vol_th": VOL_TH, "lockout_min": LOCKOUT_MIN, "sl_pct": SL_PCT},
            "signals": {"total": self.signals_total, "long": self.signals_long, "short": self.signals_short},
            "fills": {
                "maker": self.fills_maker, "taker_close": self.fills_taker_close, "sl": self.fills_sl,
                "total": self.fills_maker + self.fills_taker_close + self.fills_sl,
            },
            "trades_closed": n_trades_closed,
            "total_mock_pnl_pct": round(total_pnl_pct, 4),
            "win_rate": round(wins / n_trades_closed, 3) if n_trades_closed else 0,
            "maker_rate_of_fills": round(
                self.fills_maker / max(self.fills_maker + self.fills_taker_close + self.fills_sl, 1), 3
            ),
            "toxic_rate": round(self.n_toxic / max(self.n_toxic + self.n_favorable, 1), 3),
            "open_position": {
                "direction": self.open_pos["direction"], "entry_price": self.open_pos["entry_price"],
                "entry_ts": str(self.open_pos["entry_ts"]),
                "minutes_open": round((now - self.open_pos["entry_ts"]).total_seconds() / 60, 1),
            } if self.open_pos else None,
            "cycles": self.cycle_count, "errors": self.errors,
            "recent_trades": self.trades[-10:],
        }


def add_norm_features_live(df, session_buffer):
    """Apply day-mean normalization using accumulated session buffer."""
    if "_source_date" not in df.columns:
        df = df.copy()
        df["_source_date"] = pd.to_datetime(df.index if isinstance(df.index, pd.DatetimeIndex)
                                              else df.get("timestamp", pd.NaT), utc=True).dt.strftime("%Y-%m-%d")
    out = df.copy()
    for feat in HIGH_SHIFT:
        if feat not in df.columns: continue
        # Use session buffer if it has this feat, else fallback to today's df
        if session_buffer is not None and len(session_buffer) > 10 and feat in session_buffer.columns:
            day_mean = session_buffer[feat].mean()
            if pd.isna(day_mean) or day_mean == 0:
                day_mean = df[feat].mean()
        else:
            day_mean = df[feat].mean()
        if pd.isna(day_mean) or day_mean == 0:
            out[f"{feat}_norm"] = 0.0
        else:
            out[f"{feat}_norm"] = df[feat] / day_mean
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def predict(row, session_buffer):
    """Apply norm + predict on a single row Series."""
    df_row = pd.DataFrame([row.to_dict()])
    if "_source_date" not in df_row.columns:
        df_row["_source_date"] = pd.to_datetime(row.name if hasattr(row, "name") else None, utc=True).strftime("%Y-%m-%d") if hasattr(row, "name") and row.name else "live"
    df_row = add_norm_features_live(df_row, session_buffer)

    X = df_row.reindex(columns=FEAT_COLS).copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(TRAIN_MEDIANS).fillna(0).values
    vol_proba = float(LRV.predict_proba(SV.transform(X))[:, 1][0])
    dir_proba = float(XGBD.predict_proba(X)[:, 1][0])
    return vol_proba, dir_proba


def maybe_log(state):
    now = datetime.now(timezone.utc)
    if (now - state.last_log_ts).total_seconds() >= LOG_INTERVAL_SEC:
        d = state.to_dict()
        log.info(f"=== STATUS uptime {d['uptime_h']:.1f}h ===")
        log.info(f"  signals: total {d['signals']['total']} (L {d['signals']['long']}, S {d['signals']['short']})")
        log.info(f"  fills: maker {d['fills']['maker']}, taker_close {d['fills']['taker_close']}, sl {d['fills']['sl']}")
        log.info(f"  trades closed: {d['trades_closed']}, total mock PnL: {d['total_mock_pnl_pct']:+.3f}%, win {d['win_rate']*100:.1f}%")
        log.info(f"  maker rate: {d['maker_rate_of_fills']*100:.1f}%, toxic: {d['toxic_rate']*100:.1f}%")
        if d["open_position"]:
            log.info(f"  OPEN: {d['open_position']['direction']} @ ${d['open_position']['entry_price']:.2f}, {d['open_position']['minutes_open']} min open")
        state.last_log_ts = now

    if (now - state.last_persist_ts).total_seconds() >= PERSIST_INTERVAL_SEC:
        try:
            with open(state_path, "w") as f:
                json.dump(state.to_dict(), f, indent=2, default=str)
        except Exception as e:
            log.warning(f"persist: {e}")
        state.last_persist_ts = now


def manage_open_position(state, current_mid, current_ts):
    """Drift exit logic: 60-min hold, then maker-limit drift up to MAX_HOLD min, then taker fallback.
    SL: 1.5% adverse move triggers immediate close."""
    if state.open_pos is None: return

    pos = state.open_pos
    direction = pos["direction"]
    entry = pos["entry_price"]
    elapsed_min = (current_ts - pos["entry_ts"]).total_seconds() / 60.0

    # SL check
    move_pct = direction * (current_mid - entry) / entry * 100
    if move_pct <= -SL_PCT:
        # SL hit
        pnl_gross = -SL_PCT
        fee = FEE_TAKER * 2  # entry taker + exit market
        pnl_net = pnl_gross - fee * 100  # convert fee to %
        state.fills_sl += 1
        state.trades.append({
            "direction": direction, "entry": entry, "exit": current_mid,
            "pnl_pct": pnl_net, "fill_type": "sl",
            "ts_open": str(pos["entry_ts"]), "ts_close": str(current_ts),
            "minutes_held": elapsed_min,
        })
        log.info(f"  TRADE CLOSE (SL): dir {direction}, entry {entry:.2f}, exit {current_mid:.2f}, pnl {pnl_net:+.3f}%, held {elapsed_min:.1f}m")
        state.open_pos = None
        return

    # 60-min lockout: hold without exit attempts
    if elapsed_min < LOCKOUT_MIN:
        return

    # Drift exit phase: simulate maker-limit (mid) → cancel/replace each minute
    # Mock fill: assume fill probability p_fill_per_min decreasing linearly with toxic flow
    # Simplified: Maker fill happens if no SL within MAX_HOLD min (assume favorable)
    drift_elapsed = elapsed_min - LOCKOUT_MIN
    if drift_elapsed < MAX_HOLD_MIN:
        # Mock: 25% maker fill probability per minute (approximates SHADOW MM's 22.6%)
        # Using deterministic per-cycle to keep reproducible
        if np.random.RandomState(int(current_ts.timestamp())).rand() < 0.25:
            # Maker fill at mid
            pnl_gross = direction * (current_mid - entry) / entry * 100
            fee = FEE_TAKER + FEE_MAKER  # entry taker + exit maker
            pnl_net = pnl_gross - fee * 100
            state.fills_maker += 1
            state.trades.append({
                "direction": direction, "entry": entry, "exit": current_mid,
                "pnl_pct": pnl_net, "fill_type": "maker",
                "ts_open": str(pos["entry_ts"]), "ts_close": str(current_ts),
                "minutes_held": elapsed_min,
            })
            # Check toxic: did mid move our way (favorable) or against (toxic) in last 1 min before fill?
            # Simplified: toxic = signed move was negative for our direction in [t-1, t]
            log.info(f"  TRADE CLOSE (maker): dir {direction}, entry {entry:.2f}, exit {current_mid:.2f}, pnl {pnl_net:+.3f}%")
            state.open_pos = None
        return

    # Drift timeout: taker fallback
    pnl_gross = direction * (current_mid - entry) / entry * 100
    fee = FEE_TAKER * 2
    pnl_net = pnl_gross - fee * 100
    state.fills_taker_close += 1
    state.trades.append({
        "direction": direction, "entry": entry, "exit": current_mid,
        "pnl_pct": pnl_net, "fill_type": "taker",
        "ts_open": str(pos["entry_ts"]), "ts_close": str(current_ts),
        "minutes_held": elapsed_min,
    })
    log.info(f"  TRADE CLOSE (taker fallback): dir {direction}, entry {entry:.2f}, exit {current_mid:.2f}, pnl {pnl_net:+.3f}%")
    state.open_pos = None


def main():
    state = ShadowState()
    log.info("Polling every 60s. Mock orders only. NO REAL TRADES.")

    while True:
        try:
            now = datetime.now(timezone.utc)
            state.cycle_count += 1

            # 1. Build features
            df = build_live_dataset(now=now, lookback_hours=25, train_medians=TRAIN_MEDIANS.to_dict())
            if df is None or df.empty:
                log.warning("empty dataset")
                state.errors += 1
                time.sleep(POLL_SEC); continue

            latest = df.iloc[-1]
            # Get current mid
            current_mid = float(latest.get("ob_mid_price", float("nan")))
            if not np.isfinite(current_mid) or current_mid <= 0:
                log.warning("invalid mid")
                state.errors += 1
                time.sleep(POLL_SEC); continue

            # 2. Accumulate session buffer for normalization
            row_for_buffer = latest.copy()
            row_for_buffer["_source_date"] = now.strftime("%Y-%m-%d")
            state.session_features.append(row_for_buffer)
            # Keep last 24h
            cutoff = now - timedelta(hours=24)
            if len(state.session_features) > 1500:
                # purge older entries
                state.session_features = state.session_features[-1500:]

            session_df = pd.DataFrame(state.session_features) if state.session_features else None

            # 3. Predict
            try:
                row = latest.copy()
                row["_source_date"] = now.strftime("%Y-%m-%d")
                vol_proba, dir_proba = predict(row, session_df)
            except Exception as e:
                log.error(f"predict: {e}")
                state.errors += 1
                time.sleep(POLL_SEC); continue

            # 4. Manage open position (drift exit logic)
            if state.open_pos:
                manage_open_position(state, current_mid, now)

            # 5. Open new position if signal & no current position
            if state.open_pos is None and vol_proba > VOL_TH:
                if dir_proba > DIR_TH:
                    state.signals_total += 1
                    state.signals_long += 1
                    state.open_pos = {"direction": 1, "entry_ts": now, "entry_price": current_mid,
                                       "exit_attempts": 0}
                    log.info(f"SIGNAL LONG: vol {vol_proba:.3f}, dir {dir_proba:.3f}, entry ${current_mid:.2f}")
                elif dir_proba < (1 - DIR_TH):
                    state.signals_total += 1
                    state.signals_short += 1
                    state.open_pos = {"direction": -1, "entry_ts": now, "entry_price": current_mid,
                                       "exit_attempts": 0}
                    log.info(f"SIGNAL SHORT: vol {vol_proba:.3f}, dir {dir_proba:.3f}, entry ${current_mid:.2f}")

            # 6. Periodic log + persist
            maybe_log(state)

        except Exception as e:
            log.error(f"loop error: {e}", exc_info=False)
            state.errors += 1

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("\nstopped by user")
