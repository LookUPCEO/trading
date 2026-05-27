"""Compute liquidation features."""
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mark19.storage import read_range, write_append
from mark19.features.liquidation import compute_liquidation_features


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

    log.info(f"Reading liquidations ({args.hours}h)")
    liq = read_range("liquidation", "bybit", "ETHUSDT", start, end)
    log.info(f"  {len(liq)} liquidations")

    if len(liq) < 10:
        log.warning("not enough liquidations, exit")
        return

    log.info("Computing rolling features (60s, 300s, 3600s)")
    feat = compute_liquidation_features(liq, windows_seconds=[60, 300, 3600])
    log.info(f"  {len(feat)} 1s rows × {len(feat.columns)} cols")

    write_append(
        feat,
        data_type="liquidation_features",
        exchange="bybit",
        symbol="ETHUSDT",
    )
    log.info("Saved")

    print()
    print("=" * 70)
    print("LIQUIDATION FEATURES")
    print("=" * 70)
    print(f"\nshape: {feat.shape}")
    print(f"\ntime range: {feat['timestamp'].min()} to {feat['timestamp'].max()}")

    print(f"\n=== liq_count_60s ===")
    print(feat["liq_count_60s"].describe())
    nonzero = (feat["liq_count_60s"] > 0).sum()
    print(f"  nonzero seconds: {nonzero}/{len(feat)} ({nonzero/len(feat)*100:.1f}%)")

    print(f"\n=== liq_count_300s ===")
    print(feat["liq_count_300s"].describe())

    print(f"\n=== liq_count_3600s ===")
    print(feat["liq_count_3600s"].describe())

    print(f"\n=== liq_notional_60s ===")
    print(feat["liq_notional_60s"].describe())

    print(f"\n=== liq_notional_3600s ===")
    print(feat["liq_notional_3600s"].describe())

    print(f"\n=== liq_buy_ratio_count_300s (when active) ===")
    active_300s = feat[feat["liq_count_300s"] > 0]["liq_buy_ratio_count_300s"]
    print(active_300s.describe())

    print(f"\n=== liq_buy_ratio_notional_300s (when active) ===")
    active_300s_not = feat[feat["liq_count_300s"] > 0]["liq_buy_ratio_notional_300s"]
    print(active_300s_not.describe())

    print(f"\n=== liq_count_60s spike events ===")
    for thresh in [3, 5, 10, 20]:
        n_spikes = (feat["liq_count_60s"] >= thresh).sum()
        pct = n_spikes / len(feat) * 100
        print(f"  count>={thresh}: {n_spikes} seconds ({pct:.2f}%)")

    print(f"\n=== liq_notional_300s spike ===")
    for thresh in [10000, 50000, 100000, 500000]:
        n_spikes = (feat["liq_notional_300s"] >= thresh).sum()
        pct = n_spikes / len(feat) * 100
        print(f"  notional>=${thresh:>7,}: {n_spikes} seconds ({pct:.2f}%)")


if __name__ == "__main__":
    main()
