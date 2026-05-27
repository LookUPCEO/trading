"""
Daily feature-mismatch monitor — live bars vs backtest re-computation.

Runs once per day (cron or daemon). Compares yesterday's live bars
(from bybit_ws.py) against backtest reproduction from the same period's 1Hz raw.

Workflow:
  1. Download yesterday's 1Hz orderbook (via quote-saver.bycsi.com if not cached)
  2. Convert to mark19 schema (via convert_bybit_raw_to_mark19.py)
  3. Build v3 bars (via build_intraday_bars_v3.py)
  4. Load same date's live bars from ~/mark19_data/bars_5min_v3_live/ETHUSDT/
  5. Per-bar diff on 12 core fields (OHLC + rv/vel/obi)
  6. Write JSON to ~/mark19_data/shadow_feature_mismatch/{date}.json

For initial deploy, before nssanta-style downloader is wired here, this script
can be invoked manually with --date YYYY-MM-DD on a date that already exists in
both ~/mark19_data/ETHUSDT/ (backtest source) and bars_5min_v3_live/.
"""
from __future__ import annotations
import argparse, json, logging, subprocess, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


SYMBOL = "ETHUSDT"
RAW_1HZ_DIR = Path("/Users/mark/mark19_data") / SYMBOL  # mark19-schema 1Hz parquet
V3_BARS_DIR = Path("/Users/mark/mark19_data/bars_5min_v3") / SYMBOL
LIVE_BARS_DIR = Path("/Users/mark/mark19_data/bars_5min_v3_live") / SYMBOL
OUT_DIR = Path("/Users/mark/mark19_data/shadow_feature_mismatch")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKS = [
    # (field, abs_tolerance) — based on 33/33 equivalence test thresholds
    ('mid_open', 0.05),       # USD
    ('mid_close', 0.05),
    ('mid_high', 0.05),
    ('mid_low', 0.05),
    ('rv_bar_bp', 0.5),
    ('vel_mean_bp', 0.5),
    ('vel_abs_mean_bp', 0.5),
    ('obi5_std', 0.05),
    ('obi5_step_std', 0.05),
    ('obi5_mean', 0.05),
    ('spread_bp_mean', 0.01),
    ('tr_count', 5),          # trade count can differ by a few due to WS msg timing
]


def compare(live_path: Path, bt_path: Path, log) -> dict:
    """Per-bar diff between live and backtest. Returns a JSON-serializable report."""
    live = pd.read_parquet(live_path)
    bt = pd.read_parquet(bt_path)
    live['bar_open_ts'] = pd.to_datetime(live['bar_open_ts'], utc=True)
    bt['bar_open_ts'] = pd.to_datetime(bt['bar_open_ts'], utc=True)
    # Round to nearest 5min boundary for matching
    live['bar_key'] = live['bar_open_ts'].dt.floor('5min')
    bt['bar_key'] = bt['bar_open_ts'].dt.floor('5min')
    merged = live.merge(bt, on='bar_key', suffixes=('_live', '_bt'))
    log.info(f"  bars merged: live={len(live)} bt={len(bt)} matched={len(merged)}")

    if len(merged) == 0:
        return {'error': 'no bars matched', 'live_count': len(live), 'bt_count': len(bt)}

    field_results = {}
    overall_max_diff_within_tol = True
    for field, tol in CHECKS:
        lc = f"{field}_live"; bc = f"{field}_bt"
        if lc not in merged.columns or bc not in merged.columns:
            field_results[field] = {'status': 'missing_column'}
            overall_max_diff_within_tol = False
            continue
        diff = (merged[lc] - merged[bc]).abs()
        n_within = int((diff <= tol).sum())
        n_total = len(merged)
        worst = float(diff.max())
        worst_idx = int(diff.idxmax())
        worst_bar = merged.iloc[worst_idx]
        field_results[field] = {
            'tolerance': tol,
            'matched_bars': n_total,
            'within_tolerance_count': n_within,
            'pass_rate': n_within / n_total,
            'worst_abs_diff': worst,
            'worst_bar_ts': str(worst_bar['bar_key']),
            'worst_live_value': float(worst_bar[lc]),
            'worst_bt_value': float(worst_bar[bc]),
            'mean_abs_diff': float(diff.mean()),
        }
        if n_within < n_total:
            overall_max_diff_within_tol = False

    report = {
        'date_compared': str(live['bar_key'].iloc[0].date()) if len(live) else None,
        'bars_live': len(live),
        'bars_bt': len(bt),
        'bars_matched': len(merged),
        'fields_checked': len(CHECKS),
        'all_within_tolerance': overall_max_diff_within_tol,
        'max_diff_within_tolerance': overall_max_diff_within_tol,  # alias for runner
        'fields': field_results,
    }
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYY-MM-DD; default = yesterday UTC")
    p.add_argument("--download", action="store_true",
                   help="Download + convert + build if backtest bar missing (TODO)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger()

    if args.date:
        date_str = args.date
    else:
        date_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    log.info(f"=== Feature mismatch monitor for {date_str} ===")

    live_path = LIVE_BARS_DIR / f"{date_str}.parquet"
    bt_path = V3_BARS_DIR / f"{date_str}.parquet"

    if not live_path.exists():
        log.error(f"  Live bars missing: {live_path}"); return
    if not bt_path.exists():
        log.warning(f"  Backtest bars missing: {bt_path}")
        if args.download:
            log.info(f"  --download set, but pipeline TBD. Skipping (manual: run nssanta download + convert_bybit_raw_to_mark19 + build_intraday_bars_v3 for {date_str}).")
        else:
            log.warning(f"  Re-run with --download or manually build {bt_path}.")
        return

    report = compare(live_path, bt_path, log)
    out_path = OUT_DIR / f"{date_str}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log.info(f"  Written: {out_path}")
    log.info(f"  ALL within tolerance: {report.get('all_within_tolerance')}")
    if not report.get('all_within_tolerance', False):
        log.warning(f"  FAILED fields:")
        for f, r in report.get('fields', {}).items():
            if r.get('pass_rate', 1.0) < 1.0:
                log.warning(f"    {f}: pass {r['pass_rate']*100:.1f}%, worst |diff|={r['worst_abs_diff']:.4f} at {r['worst_bar_ts']}")


if __name__ == "__main__":
    main()
