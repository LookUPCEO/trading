"""Compute cross-exchange features."""
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mark19.storage import read_range, write_append
from mark19.features.cross_exchange import compute_price_features, compute_funding_features


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

    # Price features
    log.info(f"Reading cross-exchange prices ({args.hours}h)")
    prices = read_range("cross_exchange_prices", "combined", "ETHUSDT", start, end)
    log.info(f"  {len(prices)} price snapshots")

    if len(prices) > 0:
        price_feat = compute_price_features(prices)
        log.info(f"  computed {len(price_feat.columns)-1} price features")
        write_append(
            price_feat,
            data_type="cross_exchange_features_price",
            exchange="combined",
            symbol="ETHUSDT",
        )
        log.info("  saved")

    # Funding features
    log.info(f"Reading funding current ({args.hours}h)")
    funding = read_range("funding_current", "combined", "ETHUSDT", start, end)
    log.info(f"  {len(funding)} funding snapshots")

    if len(funding) > 0:
        funding_feat = compute_funding_features(funding)
        log.info(f"  computed {len(funding_feat.columns)-1} funding features")
        write_append(
            funding_feat,
            data_type="cross_exchange_features_funding",
            exchange="combined",
            symbol="ETHUSDT",
        )
        log.info("  saved")

    print()
    print("=" * 70)
    print("PRICE FEATURES")
    print("=" * 70)

    if len(prices) > 0:
        print(f"\nshape: {price_feat.shape}")
        print(f"\n=== max_spread_bps ===")
        print(price_feat["max_spread_bps"].describe())
        print(f"\n=== spread_std_bps ===")
        print(price_feat["spread_std_bps"].describe())
        print(f"\n=== spread_bb_bn_bps (signed) ===")
        print(price_feat["spread_bb_bn_bps"].describe())
        print(f"\n=== bybit_lead_score distribution ===")
        print(price_feat["bybit_lead_score"].value_counts(dropna=False).sort_index())

        # Quick sanity: max_spread should equal max of abs(pairwise)
        sample = price_feat.head(5)
        print(f"\n=== Sample (first 5 rows) ===")
        print(sample[["timestamp", "spread_bb_bn_bps", "spread_bb_ok_bps",
                      "spread_bn_ok_bps", "max_spread_bps", "bybit_lead_score"]].to_string())

    print()
    print("=" * 70)
    print("FUNDING FEATURES")
    print("=" * 70)

    if len(funding) > 0:
        print(f"\nshape: {funding_feat.shape}")
        print(f"\n=== funding rates per exchange ===")
        for col in ["bybit_funding", "binance_funding", "okx_funding"]:
            if col in funding_feat.columns:
                desc = funding_feat[col].describe()
                print(f"  {col}:")
                print(f"    mean={desc['mean']:+.6f}, std={desc['std']:.6f}")
                print(f"    min={desc['min']:+.6f}, max={desc['max']:+.6f}")
        print(f"\n=== funding_max_diff ===")
        print(funding_feat["funding_max_diff"].describe())
        print(f"\n=== funding_mean ===")
        print(funding_feat["funding_mean"].describe())

        # Sign patterns
        print(f"\n=== funding sign patterns (mean across 3 ex) ===")
        fm = funding_feat["funding_mean"].dropna()
        pos = (fm > 0).sum()
        neg = (fm < 0).sum()
        zero = (fm == 0).sum()
        total = len(fm)
        if total > 0:
            print(f"  positive: {pos} ({pos/total*100:.1f}%) - long pays short")
            print(f"  negative: {neg} ({neg/total*100:.1f}%) - short pays long")
            print(f"  zero: {zero}")


if __name__ == "__main__":
    main()
