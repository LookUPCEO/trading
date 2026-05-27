"""
Liquidation cascade analysis.

Cascade = N+ liquidations within W seconds (forward window).
Hypothesis: cascade events mark CHAOS regime onset.
"""
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.storage import read_range


def count_forward(liq: pd.DataFrame, ts: pd.Timestamp, window_sec: int) -> int:
    """Count liquidations in [ts, ts+W]."""
    nearby = liq[
        (liq["timestamp"] >= ts) &
        (liq["timestamp"] <= ts + timedelta(seconds=window_sec))
    ]
    return len(nearby)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=24)

    log.info("Loading...")
    feat = read_range("orderbook_features", "bybit", "ETHUSDT", start, end)
    liq = read_range("liquidation", "bybit", "ETHUSDT", start, end)
    log.info(f"  {len(feat)} feature rows, {len(liq)} liquidations")

    if len(feat) == 0 or len(liq) == 0:
        log.warning("insufficient data")
        return

    feat = feat.set_index("timestamp").sort_index()
    liq = liq.sort_values("timestamp").reset_index(drop=True)

    # ============================================================
    # 1. Cascade detection: forward window count distribution
    # ============================================================
    print()
    print("=" * 70)
    print("1. CASCADE DETECTION (forward window)")
    print("=" * 70)

    windows = [10, 30, 60, 300]

    for W in windows:
        counts = np.array([count_forward(liq, row["timestamp"], W) for _, row in liq.iterrows()])

        print(f"\n--- {W}s forward window ---")
        print(f"  liquidations followed by N or more events:")
        for thresh in [3, 5, 10, 20]:
            n_hits = (counts >= thresh).sum()
            pct = n_hits / len(counts) * 100
            print(f"    N>={thresh:2d}: {n_hits:4d} events ({pct:5.1f}%)")
        print(f"  max: {counts.max()}")
        print(f"  mean: {counts.mean():.2f}")

    # ============================================================
    # 2. Cluster events (60s forward, 3+ events) — used for window analysis
    # ============================================================
    print()
    print("=" * 70)
    print("2. CLUSTER EVENTS (60s forward, 3+ events, deduplicated)")
    print("=" * 70)

    W_CLUSTER = 60
    THRESH_CLUSTER = 3

    cluster_starts = []
    last_cluster_end = None

    for _, row in liq.iterrows():
        ts = row["timestamp"]
        nearby = liq[
            (liq["timestamp"] >= ts) &
            (liq["timestamp"] <= ts + timedelta(seconds=W_CLUSTER))
        ]

        if len(nearby) >= THRESH_CLUSTER:
            if last_cluster_end is None or ts > last_cluster_end:
                cluster_starts.append({
                    "start": ts,
                    "size": len(nearby),
                    "total_notional": float(nearby["notional"].sum()),
                    "side_buy": int((nearby["side"] == "Buy").sum()),
                    "side_sell": int((nearby["side"] == "Sell").sum()),
                })
                last_cluster_end = ts + timedelta(seconds=W_CLUSTER)

    print(f"\nClusters found: {len(cluster_starts)}")

    if len(cluster_starts) == 0:
        print("No clusters in 24h. Exit.")
        return

    cluster_df = pd.DataFrame(cluster_starts)
    print(f"  size dist: min={int(cluster_df['size'].min())}, "
          f"median={int(cluster_df['size'].median())}, "
          f"max={int(cluster_df['size'].max())}")
    print(f"  notional dist: min=${cluster_df['total_notional'].min():,.0f}, "
          f"median=${cluster_df['total_notional'].median():,.0f}, "
          f"max=${cluster_df['total_notional'].max():,.0f}")

    cluster_df["sell_dominant"] = cluster_df["side_sell"] > cluster_df["side_buy"]
    sell_dom = int(cluster_df["sell_dominant"].sum())
    print(f"  sell-dominant clusters: {sell_dom}/{len(cluster_df)}")

    # ============================================================
    # 3. Order book state around clusters
    # ============================================================
    print()
    print("=" * 70)
    print("3. ORDER BOOK STATE AROUND CLUSTERS")
    print("=" * 70)

    PRE_SEC = 60
    DURING_SEC = 60
    POST_SEC = 120

    pre_obi_list = []
    during_obi_list = []
    post_obi_list = []
    pre_spread_list = []
    during_spread_list = []
    pre_mid_list = []
    post_mid_list = []

    for cluster in cluster_starts:
        ts = cluster["start"]

        pre_w = feat[
            (feat.index >= ts - timedelta(seconds=PRE_SEC)) &
            (feat.index < ts)
        ]
        during_w = feat[
            (feat.index >= ts) &
            (feat.index < ts + timedelta(seconds=DURING_SEC))
        ]
        post_w = feat[
            (feat.index >= ts + timedelta(seconds=DURING_SEC)) &
            (feat.index < ts + timedelta(seconds=DURING_SEC + POST_SEC))
        ]

        if len(pre_w) > 0 and len(during_w) > 0 and len(post_w) > 0:
            pre_obi_list.append(pre_w["obi_top5"].mean())
            during_obi_list.append(during_w["obi_top5"].mean())
            post_obi_list.append(post_w["obi_top5"].mean())
            pre_spread_list.append(pre_w["spread"].mean())
            during_spread_list.append(during_w["spread"].mean())
            pre_mid_list.append(pre_w["mid_price"].mean())
            post_mid_list.append(post_w["mid_price"].mean())

    n = len(pre_obi_list)
    print(f"\nClusters with full window data: {n}")

    if n == 0:
        print("No full-window clusters. Exit.")
        return

    print(f"\n  PRE  OBI:    {np.mean(pre_obi_list):+.4f}")
    print(f"  DURING OBI:  {np.mean(during_obi_list):+.4f}")
    print(f"  POST OBI:    {np.mean(post_obi_list):+.4f}")
    print()
    print(f"  PRE  spread: ${np.mean(pre_spread_list):.4f}")
    print(f"  DURING spread: ${np.mean(during_spread_list):.4f}")
    print()

    price_changes = []
    for pre_p, post_p in zip(pre_mid_list, post_mid_list):
        if pre_p > 0:
            price_changes.append((post_p - pre_p) / pre_p * 100)

    if price_changes:
        print(f"  Mid price change (pre → post +{DURING_SEC+POST_SEC}s):")
        print(f"    mean: {np.mean(price_changes):+.4f}%")
        print(f"    median: {np.median(price_changes):+.4f}%")
        print(f"    std: {np.std(price_changes):.4f}%")
        up = sum(1 for x in price_changes if x > 0)
        down = sum(1 for x in price_changes if x < 0)
        print(f"    up: {up}, down: {down}")

    # ============================================================
    # 4. Baseline (random 60s windows)
    # ============================================================
    print()
    print("=" * 70)
    print("4. BASELINE (random 60s windows)")
    print("=" * 70)

    np.random.seed(42)
    baseline_obi = []
    baseline_spread = []

    n_baseline = min(500, len(feat) // 60)
    indices = np.random.choice(len(feat) - 60, size=n_baseline, replace=False)

    for idx in indices:
        window = feat.iloc[idx:idx+60]
        if len(window) > 0:
            baseline_obi.append(window["obi_top5"].mean())
            baseline_spread.append(window["spread"].mean())

    print(f"\n  baseline OBI:    {np.mean(baseline_obi):+.4f} (std {np.std(baseline_obi):.4f})")
    print(f"  baseline spread: ${np.mean(baseline_spread):.4f}")
    print()

    try:
        from scipy import stats as scistats
        t_pre, p_pre = scistats.ttest_ind(pre_obi_list, baseline_obi)
        t_during, p_during = scistats.ttest_ind(during_obi_list, baseline_obi)
        t_spread, p_spread = scistats.ttest_ind(during_spread_list, baseline_spread)
        print(f"  PRE OBI vs baseline:       t={t_pre:+.2f}, p={p_pre:.4f}")
        print(f"  DURING OBI vs baseline:    t={t_during:+.2f}, p={p_during:.4f}")
        print(f"  DURING spread vs baseline: t={t_spread:+.2f}, p={p_spread:.4f}")
    except ImportError:
        print("  (scipy not installed)")
    except Exception as e:
        print(f"  (test failed: {e})")

    print()
    print("=" * 70)
    print("Interpretation:")
    print("  - Cluster count: hours/day with cluster activity.")
    print("  - DURING OBI / spread differing from baseline = CHAOS signal.")
    print("  - Price change direction shows directional cascade effect.")
    print("=" * 70)


if __name__ == "__main__":
    main()
