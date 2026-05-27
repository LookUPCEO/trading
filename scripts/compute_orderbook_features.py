"""
Compute order book features for a date range.
Reads from data/orderbook/, writes to data/orderbook_features/.
"""
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mark19.storage import read_range, write_append
from mark19.features.orderbook import compute_all_pointwise
from mark19.features.orderbook_timeseries import (
    compute_rolling_stats,
    compute_obi_persistence,
)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=24, help="hours to process")
    args = p.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=args.hours)

    log.info(f"Reading orderbook from {start} to {end}")
    df = read_range("orderbook", "bybit", "ETHUSDT", start, end)
    log.info(f"  {len(df)} snapshots loaded")

    if len(df) < 100:
        log.warning("not enough data, exit")
        return

    log.info("Computing point-wise features")
    feat = compute_all_pointwise(df)
    log.info(f"  {len(feat.columns)} pointwise features")

    log.info("Computing rolling stats (mid_price)")
    rolling_mid = compute_rolling_stats(feat, "mid_price", [60, 300, 900])
    log.info(f"  {len(rolling_mid.columns)} rolling features")

    log.info("Computing OBI persistence (obi_top5)")
    obi_persist = compute_obi_persistence(feat, "obi_top5", [60, 300])
    log.info(f"  {len(obi_persist.columns)} persistence features")

    # Combine
    feat_indexed = feat.set_index("timestamp") if "timestamp" in feat.columns else feat
    combined = pd.concat([feat_indexed, rolling_mid, obi_persist], axis=1)
    combined = combined.reset_index().rename(columns={"index": "timestamp"})

    log.info(f"Combined: {len(combined)} rows × {len(combined.columns)} cols")

    # Save
    write_append(
        combined.dropna(subset=["timestamp"]),
        data_type="orderbook_features",
        exchange="bybit",
        symbol="ETHUSDT",
    )
    log.info("Saved to data/orderbook_features/bybit/ETHUSDT/")

    # Sanity check
    print()
    print("=" * 70)
    print("SANITY CHECK")
    print("=" * 70)
    print(f"\nshape: {combined.shape}")
    print(f"\ntime range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")

    print("\n=== Spread stats ===")
    print(combined['spread'].describe())

    print("\n=== Spread pct (bps) ===")
    print((combined['spread_pct'] * 10000).describe())

    print("\n=== OBI top5 distribution ===")
    print(combined['obi_top5'].describe())

    print("\n=== OBI top5 by sign ===")
    obi = combined['obi_top5']
    pos = (obi > 0.1).sum()
    neg = (obi < -0.1).sum()
    neutral = ((obi >= -0.1) & (obi <= 0.1)).sum()
    print(f"  positive (>0.1): {pos} ({pos/len(obi)*100:.1f}%)")
    print(f"  negative (<-0.1): {neg} ({neg/len(obi)*100:.1f}%)")
    print(f"  neutral: {neutral} ({neutral/len(obi)*100:.1f}%)")

    print("\n=== Depth top5 (ETH) ===")
    print(combined[['bid_depth_5', 'ask_depth_5']].describe())

    print("\n=== NaN counts (expected: rolling cols early) ===")
    nan_pct = (combined.isna().sum() / len(combined) * 100).sort_values(ascending=False)
    print(nan_pct.head(10))

    print("\n=== mid_price recent ===")
    print(combined[['timestamp', 'mid_price', 'spread', 'obi_top5']].tail(5).to_string())


if __name__ == "__main__":
    main()
