"""
Analyze order book imbalance around liquidation events.

For each liquidation event, compute OBI/spread in pre/post windows.
Compare to baseline (random non-event samples).

Note: Bybit liquidation 'side' interpretation is ambiguous.
We report raw 'side' (Buy/Sell) without assuming long/short interpretation.
"""
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.storage import read_range

# scipy is optional
try:
    from scipy import stats as scistats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    if not HAVE_SCIPY:
        log.warning("scipy not installed, skipping t-test (pip install scipy)")

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=24)

    log.info("Loading orderbook features...")
    feat = read_range("orderbook_features", "bybit", "ETHUSDT", start, end)
    log.info(f"  {len(feat)} feature rows")

    log.info("Loading liquidations...")
    liq = read_range("liquidation", "bybit", "ETHUSDT", start, end)
    log.info(f"  {len(liq)} liquidations")

    if len(feat) == 0 or len(liq) == 0:
        log.warning("insufficient data")
        return

    feat = feat.set_index("timestamp").sort_index()

    PRE_SEC = 30
    POST_SEC = 30

    print()
    print("=" * 70)
    print(f"OBI/Spread around Liquidations ({len(liq)} events)")
    print(f"Window: -{PRE_SEC}s to +{POST_SEC}s")
    print("=" * 70)

    for side_label, side_filter in [
        ("ALL", liq),
        ("Side=Sell", liq[liq["side"] == "Sell"]),
        ("Side=Buy", liq[liq["side"] == "Buy"]),
    ]:
        if len(side_filter) == 0:
            continue

        print(f"\n--- {side_label} ({len(side_filter)} events) ---")

        pre_obi_list = []
        at_obi_list = []
        post_obi_list = []
        pre_spread_list = []
        post_spread_list = []
        notional_list = []

        for _, event in side_filter.iterrows():
            ts = event["timestamp"]
            notional = event["notional"]

            pre_window = feat[
                (feat.index >= ts - timedelta(seconds=PRE_SEC)) &
                (feat.index < ts)
            ]
            at_window = feat[
                (feat.index >= ts - timedelta(seconds=2)) &
                (feat.index <= ts + timedelta(seconds=2))
            ]
            post_window = feat[
                (feat.index > ts) &
                (feat.index <= ts + timedelta(seconds=POST_SEC))
            ]

            if len(pre_window) > 0 and len(post_window) > 0:
                pre_obi_list.append(pre_window["obi_top5"].mean())
                post_obi_list.append(post_window["obi_top5"].mean())
                pre_spread_list.append(pre_window["spread"].mean())
                post_spread_list.append(post_window["spread"].mean())
                notional_list.append(notional)

                if len(at_window) > 0:
                    at_obi_list.append(at_window["obi_top5"].mean())

        n = len(pre_obi_list)
        if n == 0:
            print("  no matching feature data")
            continue

        print(f"  matched events: {n}")
        print(f"  avg notional:   ${np.mean(notional_list):,.0f}")
        print(f"  pre OBI mean:   {np.mean(pre_obi_list):+.4f} (std {np.std(pre_obi_list):.4f})")
        if at_obi_list:
            print(f"  AT  OBI mean:   {np.mean(at_obi_list):+.4f} (std {np.std(at_obi_list):.4f})")
        print(f"  post OBI mean:  {np.mean(post_obi_list):+.4f} (std {np.std(post_obi_list):.4f})")
        print(f"  pre spread:     ${np.mean(pre_spread_list):.4f}")
        print(f"  post spread:    ${np.mean(post_spread_list):.4f}")

        if HAVE_SCIPY:
            try:
                t_stat, p_value = scistats.ttest_1samp(pre_obi_list, 0)
                print(f"  pre OBI vs 0:   t={t_stat:+.2f}, p={p_value:.4f}")
            except Exception as e:
                print(f"  t-test failed: {e}")

    print()
    print(f"\n--- Baseline (1000 random non-event samples) ---")
    np.random.seed(42)
    n_sample = min(1000, len(feat))
    random_indices = np.random.choice(len(feat), size=n_sample, replace=False)
    baseline_obi = feat.iloc[random_indices]["obi_top5"].dropna()
    if len(baseline_obi) > 0:
        print(f"  mean OBI: {baseline_obi.mean():+.4f}")
        print(f"  std OBI:  {baseline_obi.std():.4f}")

    print()
    print("=" * 70)
    print("Interpretation guide:")
    print("  - Compare each side's pre-OBI mean to baseline (~0).")
    print("  - If significantly different (p<0.05) AND consistent direction,")
    print("    OBI may be predictive of liquidation events.")
    print("  - Sign tells whether side=Sell aligns with neg/pos OBI.")
    print("=" * 70)


if __name__ == "__main__":
    main()
