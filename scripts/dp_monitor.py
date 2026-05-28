"""Retrospective ΔP monitor — stale-OB decision-impact safety net.

For each 4h boundary on a given day:
  1. Reads live bars (what the bot saw, possibly stale).
  2. Reads backtest bars (ground-truth via quote-saver archive).
  3. Computes P_entry under both, with the same feature engineering + same model.
  4. Records ΔP and decision-change flag. Discord alerts on |ΔP|>0.01 or decision flip.

Usage:
  python scripts/dp_monitor.py --date 2026-05-26
  python scripts/dp_monitor.py --date 2026-05-26 --send-discord
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import discord_notify as dn
except Exception:
    dn = None

ROOT = Path("/Users/mark/mark19_data")
MODEL_PATH = ROOT / "models_prod" / "4h_direction_v2.joblib"
HIST_DIR = ROOT / "bars_5min_v3" / "ETHUSDT"
LIVE_DIR = ROOT / "bars_5min_v3_live" / "ETHUSDT"
DP_LOG = ROOT / "dp_monitor.jsonl"
DP_THRESHOLD = 0.01    # |ΔP| above this triggers alert

ART = joblib.load(MODEL_PATH)
MODEL = ART['model']
FCOLS = ART['feature_columns']
MEDIANS = ART['train_medians']
CONFIG = ART['config']
DECISION_THR = CONFIG['threshold']   # 0.05


def compute_long_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror of shadow_runner.compute_long_features."""
    df = df.copy().sort_values("bar_open_ts").reset_index(drop=True)
    SHORT_LAG = ['return_5m_bar_bp','obi5_last','micro_dev_bp_last','rv_bar_bp',
                 'ofi_proxy','tr_net_size','tr_tick_imb']
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


def predict_at(df_features: pd.DataFrame, ts: pd.Timestamp) -> tuple[float, str] | None:
    row = df_features[df_features.bar_open_ts == ts]
    if row.empty:
        return None
    X = row[FCOLS].copy()
    for c in FCOLS:
        X[c] = X[c].fillna(MEDIANS.get(c, 0))
    # Force ALL trained trees (consistent with shadow_runner + mark19_live).
    _n_trees = MODEL.get_booster().num_boosted_rounds()
    p_up = float(MODEL.predict_proba(X, iteration_range=(0, _n_trees))[0, 1])
    if p_up > 0.5 + DECISION_THR: dec = "LONG"
    elif p_up < 0.5 - DECISION_THR: dec = "SHORT"
    else: dec = "SKIP"
    return p_up, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD to verify (must have both live + backtest bars)")
    ap.add_argument("--bt-bars-dir", type=Path, default=Path("/tmp/raw_compare/bars_v3/ETHUSDT"),
                    help="Backtest bars (ground truth) directory")
    ap.add_argument("--send-discord", action="store_true")
    args = ap.parse_args()

    date_str = args.date
    bt_path = args.bt_bars_dir / f"{date_str}.parquet"
    live_path = LIVE_DIR / f"{date_str}.parquet"
    if not bt_path.exists():
        print(f"❌ Backtest bars missing: {bt_path}"); sys.exit(1)
    if not live_path.exists():
        print(f"❌ Live bars missing: {live_path}"); sys.exit(1)

    # Build feature frames (one with bt, one with live, shared warm-up history)
    bt = pd.read_parquet(bt_path)
    live = pd.read_parquet(live_path)
    bt['bar_open_ts'] = pd.to_datetime(bt['bar_open_ts'], utc=True).dt.floor('5min')
    live['bar_open_ts'] = pd.to_datetime(live['bar_open_ts'], utc=True)

    # Warm-up: last 15 days of historical bars
    hist_files = sorted(HIST_DIR.glob("*.parquet"))[-15:]
    hist = pd.concat([pd.read_parquet(f) for f in hist_files], ignore_index=True)
    hist['bar_open_ts'] = pd.to_datetime(hist['bar_open_ts'], utc=True)
    # Drop the target date from history to avoid leakage
    hist = hist[hist['bar_open_ts'].dt.date != pd.Timestamp(date_str).date()]

    df_gt = pd.concat([hist, bt], ignore_index=True).sort_values('bar_open_ts').drop_duplicates(
        'bar_open_ts', keep='last').reset_index(drop=True)
    df_lv = pd.concat([hist, live], ignore_index=True).sort_values('bar_open_ts').drop_duplicates(
        'bar_open_ts', keep='last').reset_index(drop=True)

    df_gt = compute_long_features(df_gt)
    df_lv = compute_long_features(df_lv)

    # 4h boundaries on this date — 00, 04, 08, 12, 16, 20 UTC
    boundaries = [pd.Timestamp(f"{date_str} {h:02d}:00:00", tz='UTC') for h in [0, 4, 8, 12, 16, 20]]
    rows = []
    alerts = []
    for ts in boundaries:
        g = predict_at(df_gt, ts)
        l = predict_at(df_lv, ts)
        if g is None or l is None:
            continue
        p_gt, d_gt = g
        p_lv, d_lv = l
        dp = p_lv - p_gt
        decision_flip = (d_gt != d_lv)
        row = {"ts": ts.isoformat(), "p_gt": p_gt, "p_live": p_lv,
               "dP": dp, "d_gt": d_gt, "d_live": d_lv, "flip": decision_flip}
        rows.append(row)
        msg = f"  {ts}: gt={d_gt}({p_gt:.4f}) live={d_lv}({p_lv:.4f}) ΔP={dp:+.4f}{' ⚠️FLIP' if decision_flip else ''}"
        print(msg)
        if abs(dp) > DP_THRESHOLD or decision_flip:
            alerts.append((ts, dp, decision_flip, d_gt, d_lv))

    if not rows:
        print("No comparable boundaries (missing bars).")
        return

    # Persist to log
    DP_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DP_LOG, "a") as f:
        for r in rows:
            f.write(json.dumps({**r, "date_verified": date_str,
                                "verified_at": datetime.now(timezone.utc).isoformat()}) + "\n")

    # Summary
    abs_dp = [abs(r["dP"]) for r in rows]
    n_flip = sum(r["flip"] for r in rows)
    print(f"\n  Boundaries: {len(rows)} | mean |ΔP|={np.mean(abs_dp):.4f} | max |ΔP|={max(abs_dp):.4f} | flips={n_flip}")
    print(f"  Alerts (|ΔP|>{DP_THRESHOLD} or flip): {len(alerts)}")
    print(f"  Persisted to {DP_LOG}")

    if args.send_discord and dn and alerts:
        lines = [f"⚠️ Stale-OB ΔP alert for {date_str}"]
        for ts, dp, flip, dg, dl in alerts:
            lines.append(f"  {ts.strftime('%H:%M')}: gt={dg} live={dl} ΔP={dp:+.4f}{' FLIP' if flip else ''}")
        dn.warning(f"ΔP monitor {date_str}", "\n".join(lines))


if __name__ == "__main__":
    main()
