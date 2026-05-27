"""
Extract daily microstructure features from converted Mark19-schema parquet files.

Output: one row per (symbol, date) with regime features for the day.
Computed from 1Hz top-50-level snapshots.

Features (computable from 1Hz snapshots, no trades/raw-delta needed):
  - mean_spread_bp
  - mean_depth_top5         (sum sizes at bid_0..4 + ask_0..4)
  - mid_realized_vol_1m_pct (1-min mid log-return std, in %)
  - update_rate_per_sec     (update_id deltas / seconds)
  - depth_slope             (cumulative depth vs level slope, top 20)
  - bid_ask_imbalance       (sum bid_top5 - sum ask_top5) / total
  - mid_range_bp            (intra-day high-low / mid)
  - depth_top5_cv           (std/mean of depth_top5 series)
  - log_mid_drift_bp        (close-open mid)

Output schema (one row per day):
  date, symbol, n_snapshots, mean_spread_bp, mean_depth_top5,
  mid_realized_vol_1m_pct, update_rate_per_sec, depth_slope,
  bid_ask_imbalance, mid_range_bp, depth_top5_cv, log_mid_drift_bp
"""
from __future__ import annotations

import argparse, logging, sys, time, math
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_MARK19_ROOT = Path("/Volumes/PortableSSD/bybit_data/parquet_mark19")
OUT_PATH = Path("/Volumes/PortableSSD/bybit_data/regime_features.parquet")
N_DEPTH_LEVELS_SLOPE = 20  # use bid_0..19 / ask_0..19 for slope
N_TOP = 5


def features_for_day(df: pd.DataFrame) -> dict:
    """Compute regime features for one day. df has Mark19 schema."""
    if len(df) == 0:
        return {}
    df = df.sort_values("timestamp").reset_index(drop=True)
    mid = (df.bid_0_price + df.ask_0_price) / 2
    spread = df.ask_0_price - df.bid_0_price
    spread_bp = spread / mid * 10000

    bid_top5 = sum(df[f"bid_{k}_size"].fillna(0) for k in range(N_TOP))
    ask_top5 = sum(df[f"ask_{k}_size"].fillna(0) for k in range(N_TOP))
    depth_top5 = bid_top5 + ask_top5
    bai = (bid_top5 - ask_top5) / depth_top5.replace(0, np.nan)

    # mid 1-min log-return std
    df1 = pd.DataFrame({"ts": df.timestamp, "mid": mid})
    df1 = df1.set_index("ts").resample("1min").last().dropna()
    log_ret = np.log(df1["mid"]).diff().dropna()
    rv = log_ret.std() * 100  # in %

    # update_rate from update_id deltas
    if df.update_id.dtype.kind in "iu" and len(df) > 1:
        upd_delta = int(df.update_id.iloc[-1]) - int(df.update_id.iloc[0])
        secs = (df.timestamp.iloc[-1] - df.timestamp.iloc[0]).total_seconds()
        upd_rate = upd_delta / secs if secs > 0 else 0.0
    else:
        upd_rate = 0.0

    # depth slope: avg over snapshots of cumulative-depth vs level slope (top 20)
    sample = df.sample(min(1000, len(df)), random_state=0)
    slopes = []
    for _, row in sample.iterrows():
        cum_bid = 0.0
        cum_ask = 0.0
        levels = []
        cums = []
        for k in range(N_DEPTH_LEVELS_SLOPE):
            b = row.get(f"bid_{k}_size", 0)
            a = row.get(f"ask_{k}_size", 0)
            cum_bid += 0 if pd.isna(b) else b
            cum_ask += 0 if pd.isna(a) else a
            levels.append(k)
            cums.append(cum_bid + cum_ask)
        if cums[-1] > 0:
            slope = np.polyfit(levels, cums, 1)[0]
            slopes.append(slope)
    depth_slope = float(np.mean(slopes)) if slopes else 0.0

    # mid range, drift
    mid_high = mid.max(); mid_low = mid.min()
    mid_range_bp = (mid_high - mid_low) / mid.mean() * 10000
    log_mid_drift_bp = (math.log(mid.iloc[-1]) - math.log(mid.iloc[0])) * 10000

    depth_top5_cv = depth_top5.std() / depth_top5.mean() if depth_top5.mean() > 0 else 0.0

    return {
        "n_snapshots": int(len(df)),
        "mean_spread_bp": float(spread_bp.mean()),
        "mean_depth_top5": float(depth_top5.mean()),
        "mid_realized_vol_1m_pct": float(rv),
        "update_rate_per_sec": float(upd_rate),
        "depth_slope": depth_slope,
        "bid_ask_imbalance": float(bai.mean()),
        "mid_range_bp": float(mid_range_bp),
        "depth_top5_cv": float(depth_top5_cv),
        "log_mid_drift_bp": float(log_mid_drift_bp),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SOLUSDT")
    p.add_argument("--limit", type=int, default=0, help="Stop after N days (0 = no limit)")
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--data-root", type=Path, default=DEFAULT_MARK19_ROOT, help="Mark19 parquet root")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    records = []
    n_done = 0
    t0 = time.time()
    for sym in symbols:
        dir_path = args.data_root / sym
        if not dir_path.exists():
            log.warning(f"{sym}: no dir at {dir_path}")
            continue
        files = sorted(dir_path.glob("*.parquet"))
        log.info(f"{sym}: {len(files)} converted days available")
        for f in files:
            date_str = f.stem
            try:
                df = pd.read_parquet(f)
                feats = features_for_day(df)
                rec = {"date": date_str, "symbol": sym, **feats}
                records.append(rec)
                n_done += 1
                if n_done % 10 == 0:
                    log.info(f"  [{n_done}] {sym} {date_str} ... spread_bp={feats['mean_spread_bp']:.2f} rv1m={feats['mid_realized_vol_1m_pct']:.3f}")
                if args.limit > 0 and n_done >= args.limit:
                    break
            except Exception as e:
                log.error(f"  {sym} {date_str}: {type(e).__name__}: {e}")
        if args.limit > 0 and n_done >= args.limit:
            break

    out_df = pd.DataFrame(records).sort_values(["symbol", "date"]).reset_index(drop=True)
    log.info(f"DONE. {len(out_df)} day-rows in {time.time()-t0:.1f}s")
    log.info(f"  Output: {args.out}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(out_df.describe(include="all").to_string())


if __name__ == "__main__":
    main()
