"""
Shadow forward test for 4h Direction strategy.

NO REAL TRADES. Logs:
  - Live 1Hz orderbook snapshot (Bybit WebSocket or REST)
  - Live trades stream
  - Every 4 hours: build long features (same code as backtest), run model, log:
      * predicted P(up)
      * intended action (long / short / skip)
      * "limit price" we would post (best ± offset)
      * timestamp
  - After 4h: check if our limit price was crossed during the hold (estimated fill)
  - Compare to mid_close[t+48] for hypothetical PnL
  - Aggregate over days

Goals:
  1. Verify live feature distribution matches backtest training distribution
  2. Estimate realistic maker fill rate (vs 38% assumption)
  3. Sanity-check P_entry distribution and trade frequency
  4. Detect model drift before risking capital

Requires:
  - Bybit API key (read-only, no withdraw permissions)
  - or use public WebSocket (no auth needed for market data)
  - Trained model checkpoint (~/mark19_data/4h_direction_model.joblib)

This is a SKELETON — adjust to live data source you choose.
"""
from __future__ import annotations
import argparse, json, logging, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ====== Config ======
SYMBOL = "ETHUSDT"
BAR_SECONDS = 300            # 5-min bar
HOLD_BARS = 48               # 4h hold
THRESHOLD = 0.05             # |p-0.5| > 0.05
LOG_DIR = Path("/Users/mark/mark19_data/shadow_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path("/Users/mark/mark19_data/4h_direction_model.joblib")
# Buffer for live data
BUFFER_HOURS = 24 * 8        # Need at least 7d of history for long features


def long_features_from_bars(bars_df: pd.DataFrame) -> pd.DataFrame:
    """Exact replica of backtest feature engineering — must stay in sync."""
    df = bars_df.copy()
    SHORT_LAG = ['return_5m_bar_bp','obi5_last','micro_dev_bp_last','rv_bar_bp','ofi_proxy','tr_net_size','tr_tick_imb']
    for c in SHORT_LAG:
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


def emit_intent(p_up: float, mid: float, best_bid: float, best_ask: float, ts: datetime) -> dict:
    """Decide what we would do at this 4h boundary. Log only — no order placed."""
    confidence = abs(p_up - 0.5)
    if confidence <= THRESHOLD:
        action = "SKIP"
        limit_price = None
    else:
        direction = "LONG" if p_up > 0.5 else "SHORT"
        # Maker: post inside the spread at best ± 1 tick? Or just at best?
        # Simplest: post at best (top-of-book) on our side.
        limit_price = best_bid if direction == "LONG" else best_ask
        action = direction
    return {
        "decision_ts": ts.isoformat(),
        "p_up": float(p_up),
        "confidence": float(confidence),
        "mid": float(mid),
        "best_bid": float(best_bid),
        "best_ask": float(best_ask),
        "action": action,
        "limit_price": float(limit_price) if limit_price else None,
        "hold_until_ts": (ts + timedelta(hours=4)).isoformat(),
        "result": None,  # filled in 4h later
    }


def check_fill_and_pnl(intent: dict, bars_during_hold: pd.DataFrame) -> dict:
    """Given the bar series during the 4h hold, estimate if limit got filled and P&L."""
    if intent["action"] == "SKIP" or len(bars_during_hold) == 0:
        return {**intent, "result": "no_trade"}

    limit = intent["limit_price"]
    direction = intent["action"]
    # Maker fill estimate: limit got hit if market crossed our level during the hold
    if direction == "LONG":  # we bid at best_bid; filled when ask drops to ≤ limit
        filled = (bars_during_hold["mid_low"] <= limit).any()
    else:                    # we ask at best_ask; filled when bid rises to ≥ limit
        filled = (bars_during_hold["mid_high"] >= limit).any()

    exit_mid = bars_during_hold["mid_close"].iloc[-1]
    if filled:
        if direction == "LONG":
            ret_bp = (np.log(exit_mid) - np.log(limit)) * 10000
        else:
            ret_bp = (np.log(limit) - np.log(exit_mid)) * 10000
        # Apply fees: 2bp maker entry + ~5.5bp taker exit (conservative)
        net_bp = ret_bp - 2.0 - 5.5
        return {**intent, "result": "filled_maker", "fill_price": limit,
                "exit_mid": float(exit_mid), "pre_fee_bp": float(ret_bp),
                "net_bp_conservative": float(net_bp)}
    else:
        # Strategy choice: would we taker-chase or skip?
        # Log both for later analysis.
        if direction == "LONG":
            taker_entry = bars_during_hold["mid_close"].iloc[0]  # approx best_ask at entry
            ret_taker_bp = (np.log(exit_mid) - np.log(taker_entry)) * 10000
        else:
            taker_entry = bars_during_hold["mid_close"].iloc[0]
            ret_taker_bp = (np.log(taker_entry) - np.log(exit_mid)) * 10000
        net_taker = ret_taker_bp - 5.5 - 5.5  # both legs taker
        return {**intent, "result": "missed_maker",
                "would_taker_fill": float(taker_entry),
                "if_taker_pre_fee_bp": float(ret_taker_bp),
                "if_taker_net_bp": float(net_taker)}


def run(args):
    """Main loop — to be wired to live data feed."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger()
    log.info("Shadow forward test started (NO REAL TRADES)")
    log.info(f"  Symbol={SYMBOL}  bar={BAR_SECONDS}s  hold={HOLD_BARS}*5m={HOLD_BARS*5}min  thr={THRESHOLD}")
    log.info(f"  Log dir: {LOG_DIR}")
    log.info(f"  Model: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        log.error(f"  Model not found. Train + save first (see save_model.py).")
        return

    log.info("""
=== TO IMPLEMENT (your choice of data source) ===
  Option A: WebSocket subscribe to Bybit public stream
    - orderbook.50.ETHUSDT (Bybit's 50-level perpetual orderbook delta)
    - publicTrade.ETHUSDT
    - On each event, update local OB state + append trades
    - Every BAR_SECONDS, snapshot to bar; append to ring buffer
    - Every HOLD_BARS bars (4h), run feature pipeline → model → emit_intent → log JSON

  Option B: REST polling
    - GET /v5/market/orderbook?symbol=ETHUSDT&limit=50 every second
    - GET /v5/market/recent-trade?symbol=ETHUSDT every 5s
    - Build bars, same flow

  Implementation note:
    - Start with 7+ days of historical data (the 7d rolling features need warm-up).
      Easy way: backfill from existing ~/mark19_data/bars_5min_v3/ETHUSDT/ as initial buffer.
    - Persist buffer to disk each bar so restarts resume cleanly.

=== TO MEASURE OVER 1-2 WEEKS ===
  1. Live feature distribution vs backtest (per-feature mean/std drift)
  2. Estimated maker fill rate (% of intents marked 'filled_maker')
  3. Trade frequency (target: ~0.9/day with thr=0.05)
  4. Shadow PnL vs backtest expectation (+46 bp/day with 38% fill)

=== KILL CRITERIA (DO NOT go live if) ===
  - Live feature distribution drifts > 2σ from training (regime shift)
  - Estimated fill rate < 20% (model places too aggressive limits)
  - Shadow daily PnL < 0 over 14 days
  - Trade frequency >> 0.9/day or << 0.5/day (model behaving abnormally)
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=SYMBOL)
    args = parser.parse_args()
    run(args)
