"""Build integrated feature dataset with targets."""
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mark19.storage import write_append
from mark19.features.integration import (
    load_all_features,
    integrate_to_1s_grid,
    add_targets,
)


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

    log.info(f"Loading streams ({args.hours}h)")
    streams = load_all_features(start, end)
    for name, df in streams.items():
        cols = len(df.columns) if len(df) > 0 else 0
        log.info(f"  {name}: {len(df)} rows × {cols} cols")

    log.info("Integrating to 1s grid")
    integrated = integrate_to_1s_grid(streams)
    log.info(f"  integrated: {len(integrated)} rows × {len(integrated.columns)} cols")

    log.info("Adding targets")
    final = add_targets(integrated, horizons_seconds=[300, 900, 3600])
    log.info(f"  final: {len(final)} rows × {len(final.columns)} cols")

    write_append(
        final,
        data_type="integrated_dataset",
        exchange="bybit",
        symbol="ETHUSDT",
    )
    log.info("Saved")

    print()
    print("=" * 70)
    print("INTEGRATED DATASET")
    print("=" * 70)
    print(f"\nshape: {final.shape}")
    print(f"time range: {final['timestamp'].min()} to {final['timestamp'].max()}")

    # Column inventory by prefix
    prefixes = {"ob": [], "tr": [], "liq": [], "cx": [], "cf": [], "target": [], "other": []}
    for col in final.columns:
        if col == "timestamp":
            prefixes["other"].append(col)
        elif col.startswith("ob_"):
            prefixes["ob"].append(col)
        elif col.startswith("tr_"):
            prefixes["tr"].append(col)
        elif col.startswith("liq_"):
            prefixes["liq"].append(col)
        elif col.startswith("cx_"):
            prefixes["cx"].append(col)
        elif col.startswith("cf_"):
            prefixes["cf"].append(col)
        elif col.startswith("target_"):
            prefixes["target"].append(col)
        else:
            prefixes["other"].append(col)

    print(f"\n=== Column inventory ===")
    for prefix, cols in prefixes.items():
        print(f"  {prefix}: {len(cols)}")

    print(f"\n=== Sample feature columns ===")
    for prefix in ["ob", "tr", "liq", "cx", "cf"]:
        cols = prefixes[prefix]
        if cols:
            sample = cols[:3]
            print(f"  {prefix}: {sample}{'...' if len(cols) > 3 else ''}")

    print(f"\n=== Target stats ===")
    for col in prefixes["target"]:
        valid = final[col].dropna()
        if len(valid) > 0:
            print(f"  {col}:")
            print(f"    n={len(valid)}, mean={valid.mean():+.4f}, std={valid.std():.4f}")
            print(f"    range=[{valid.min():+.4f}, {valid.max():+.4f}]")

    # NaN summary
    print(f"\n=== NaN summary ===")
    nan_pct = (final.isna().sum() / len(final) * 100)
    high_nan = nan_pct[nan_pct > 5].sort_values(ascending=False)
    if len(high_nan) > 0:
        print("  columns with >5% NaN:")
        for col, pct in high_nan.head(20).items():
            print(f"    {col}: {pct:.1f}%")
    else:
        print("  all columns <5% NaN")

    # Memory
    mem_mb = final.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"\n=== Memory ===")
    print(f"  total: {mem_mb:.1f} MB")

    # Sanity: sample row with all targets
    print(f"\n=== Sample row at midpoint ===")
    mid_idx = len(final) // 2
    sample_row = final.iloc[mid_idx]
    print(f"  timestamp: {sample_row['timestamp']}")
    print(f"  ob_mid_price: {sample_row['ob_mid_price']}")
    for col in prefixes["target"]:
        print(f"  {col}: {sample_row[col]:+.4f}" if not pd.isna(sample_row[col]) else f"  {col}: NaN")


if __name__ == "__main__":
    main()
