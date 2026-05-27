"""
Compute trades features for last N hours.
"""
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mark19.storage import read_range, write_append
from mark19.features.trades import aggregate_to_1s, compute_rolling_features


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=24)
    args = p.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=args.hours)

    log.info(f"Reading trades from {start} to {end}")
    trades = read_range("trades", "bybit", "ETHUSDT", start, end)
    log.info(f"  {len(trades)} trades loaded")

    if len(trades) < 1000:
        log.warning("not enough trades, exit")
        return

    log.info("Aggregating to 1s buckets (vectorized)")
    agg = aggregate_to_1s(trades)
    log.info(f"  {len(agg)} 1s buckets")

    log.info("Computing rolling features")
    rolling = compute_rolling_features(agg, windows_seconds=[60, 300, 900])
    log.info(f"  {len(rolling)} rows × {len(rolling.columns)} cols")

    combined = pd.merge(agg, rolling, on="timestamp", how="outer").sort_values("timestamp")
    log.info(f"Combined: {len(combined)} rows × {len(combined.columns)} cols")

    write_append(
        combined.dropna(subset=["timestamp"]),
        data_type="trades_features",
        exchange="bybit",
        symbol="ETHUSDT",
    )
    log.info("Saved to data/trades_features/bybit/ETHUSDT/")

    print()
    print("=" * 70)
    print("SANITY CHECK")
    print("=" * 70)
    print(f"\nshape: {combined.shape}")
    print(f"\ntime range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")

    print("\n=== buy_ratio (1s instant) ===")
    print(combined['buy_ratio'].describe())

    print("\n=== buy_ratio_60s (rolling 1min) ===")
    print(combined['buy_ratio_60s'].describe())

    print("\n=== buy_ratio_300s (rolling 5min) ===")
    print(combined['buy_ratio_300s'].describe())

    print("\n=== trades_per_sec_60s ===")
    print(combined['trades_per_sec_60s'].describe())

    print("\n=== large_trade_ratio_60s ===")
    print(combined['large_trade_ratio_60s'].describe())

    print("\n=== tick_buy_ratio_60s ===")
    print(combined['tick_buy_ratio_60s'].describe())

    print("\n=== buy_ratio_60s sign distribution ===")
    br = combined['buy_ratio_60s'].dropna()
    above_55 = (br > 0.55).sum()
    below_45 = (br < 0.45).sum()
    neutral = ((br >= 0.45) & (br <= 0.55)).sum()
    total = len(br)
    if total > 0:
        print(f"  >0.55 (buy dominant): {above_55} ({above_55/total*100:.1f}%)")
        print(f"  <0.45 (sell dominant): {below_45} ({below_45/total*100:.1f}%)")
        print(f"  neutral (0.45-0.55): {neutral} ({neutral/total*100:.1f}%)")

    print("\n=== buy_ratio_60s vs volume_imbalance_60s correlation ===")
    if 'volume_imbalance_60s' in combined.columns:
        valid = combined[['buy_ratio_60s', 'volume_imbalance_60s']].dropna()
        if len(valid) > 1:
            corr = valid.corr().iloc[0, 1]
            print(f"  correlation: {corr:.4f} (expected ~1.0, linear transform)")


if __name__ == "__main__":
    main()
