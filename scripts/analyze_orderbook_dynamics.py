"""Order Book Dynamics Analysis (시도 14 정확 Phase 1)."""
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.storage import read_range
from mark19.ml.data_prep import DATES_TEST


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("ORDER BOOK DYNAMICS ANALYSIS (시도 14 Phase 1)")
    log.info("=" * 70)

    log.info(f"\nLoading order book data for {len(DATES_TEST)} test dates...")

    all_ob_data = []

    for date_str in DATES_TEST:
        y, m, d = map(int, date_str.split('-'))
        start = datetime(y, m, d, tzinfo=timezone.utc)
        end = start + timedelta(days=1)

        try:
            df = read_range("orderbook", "bybit_tardis", "ETHUSDT", start, end)
            if len(df) > 0:
                df["_source_date"] = date_str
                all_ob_data.append(df)
                log.info(f"  {date_str}: {len(df)} rows loaded")
        except Exception as e:
            log.error(f"  {date_str}: ERROR {e}")

    if not all_ob_data:
        log.error("No data loaded")
        return

    ob_df = pd.concat(all_ob_data, ignore_index=True)
    if "timestamp" in ob_df.columns:
        ob_df["timestamp"] = pd.to_datetime(ob_df["timestamp"], utc=True)
        ob_df = ob_df.sort_values(["_source_date", "timestamp"]).reset_index(drop=True)

    log.info(f"\nTotal order book rows: {len(ob_df)}")

    # ============================================================
    # Spread + mid (Tardis format: bid_0_price, ask_0_price)
    # ============================================================
    ob_df["spread"] = ob_df["ask_0_price"] - ob_df["bid_0_price"]
    ob_df["mid"] = (ob_df["ask_0_price"] + ob_df["bid_0_price"]) / 2
    ob_df["spread_pct"] = ob_df["spread"] / ob_df["mid"] * 100  # in percent

    # ============================================================
    # 1. SPREAD DISTRIBUTION
    # ============================================================
    print()
    print("=" * 80)
    print("1. SPREAD DISTRIBUTION")
    print("=" * 80)

    print(f"\nOverall spread distribution (in %):")
    print(ob_df["spread_pct"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

    print(f"\nPer-date spread:")
    print(f"  {'Date':<14} {'Median (%)':<14} {'Mean (%)':<14} {'P90 (%)':<14}")
    print("  " + "-" * 56)
    for date_str in DATES_TEST:
        sub = ob_df[ob_df["_source_date"] == date_str]
        if len(sub) > 0:
            print(f"  {date_str:<14} {sub['spread_pct'].median():<14.6f} {sub['spread_pct'].mean():<14.6f} {sub['spread_pct'].quantile(0.9):<14.6f}")

    # ============================================================
    # 2. ORDER BOOK DEPTH (Top 5 levels)
    # ============================================================
    print()
    print("=" * 80)
    print("2. ORDER BOOK DEPTH (Top 5 levels)")
    print("=" * 80)

    bid_5 = [f"bid_{i}_size" for i in range(5)]
    ask_5 = [f"ask_{i}_size" for i in range(5)]

    print(f"\n{'Level':<14} {'Median':<14} {'Mean':<14} {'P90':<14}")
    print("-" * 56)
    for col in bid_5 + ask_5:
        if col in ob_df.columns:
            stats = ob_df[col].describe(percentiles=[0.5, 0.9])
            print(f"{col:<14} {stats['50%']:<14.2f} {stats['mean']:<14.2f} {stats['90%']:<14.2f}")

    # Best bid/ask depth concentration
    bid_total_5 = ob_df[bid_5].sum(axis=1)
    ask_total_5 = ob_df[ask_5].sum(axis=1)
    print(f"\nTop 5 total depth:")
    print(f"  Bid total median: {bid_total_5.median():.2f}, mean {bid_total_5.mean():.2f}")
    print(f"  Ask total median: {ask_total_5.median():.2f}, mean {ask_total_5.mean():.2f}")

    bid_0_concentration = ob_df["bid_0_size"] / bid_total_5.replace(0, np.nan)
    ask_0_concentration = ob_df["ask_0_size"] / ask_total_5.replace(0, np.nan)
    print(f"\nBest level concentration (level 0 / top 5):")
    print(f"  Bid_0: median {bid_0_concentration.median():.3f}")
    print(f"  Ask_0: median {ask_0_concentration.median():.3f}")

    # ============================================================
    # 3. MID-PRICE DYNAMICS (1-second changes)
    # ============================================================
    print()
    print("=" * 80)
    print("3. MID-PRICE DYNAMICS (1-second changes)")
    print("=" * 80)

    print(f"\n{'Date':<14} {'1s Δ med':<14} {'1s Δ% med':<14} {'1s Δ% P90':<14} {'1s Δ% P99':<14}")
    print("-" * 70)
    for date_str in DATES_TEST:
        sub = ob_df[ob_df["_source_date"] == date_str].sort_values("timestamp")
        if len(sub) < 100:
            continue

        mid_changes = sub["mid"].diff().abs()
        mid_pct_changes = mid_changes / sub["mid"]

        med_abs = mid_changes.median()
        med_pct = mid_pct_changes.median() * 100
        p90_pct = mid_pct_changes.quantile(0.9) * 100
        p99_pct = mid_pct_changes.quantile(0.99) * 100

        print(f"{date_str:<14} {med_abs:<14.4f} {med_pct:<14.6f}% {p90_pct:<14.6f}% {p99_pct:<14.6f}%")

    # ============================================================
    # 4. MAKER FILL ESTIMATION (heuristic baseline)
    # ============================================================
    print()
    print("=" * 80)
    print("4. MAKER FILL ESTIMATION (1-min window)")
    print("=" * 80)
    print()
    print("Long limit at best bid: fill if mid drops by >= spread/2 within 1 min")
    print("Short limit at best ask: fill if mid rises by >= spread/2 within 1 min")
    print()
    print(f"{'Date':<14} {'Long fill':<14} {'Short fill':<14} {'Long unfilled':<16} {'Short unfilled':<16}")
    print("-" * 76)

    overall_long_fill = []
    overall_short_fill = []

    for date_str in DATES_TEST:
        sub = ob_df[ob_df["_source_date"] == date_str].sort_values("timestamp").reset_index(drop=True).copy()
        if len(sub) < 100:
            continue

        sub["mid_1min_later"] = sub["mid"].shift(-60)  # 60s = 1 min (1Hz data)
        sub["mid_change_1min"] = (sub["mid_1min_later"] - sub["mid"]) / sub["mid"]

        spread_half = sub["spread_pct"] / 2 / 100  # convert % to fraction

        # Long fill: price drops by ≥ half-spread (touches the bid)
        prob_long_fill = (sub["mid_change_1min"] <= -spread_half * 0.5).mean()
        # Short fill: price rises by ≥ half-spread
        prob_short_fill = (sub["mid_change_1min"] >= spread_half * 0.5).mean()
        # Drift away (significantly unfilled)
        prob_long_unfilled = (sub["mid_change_1min"] >= spread_half * 2).mean()
        prob_short_unfilled = (sub["mid_change_1min"] <= -spread_half * 2).mean()

        overall_long_fill.append(prob_long_fill)
        overall_short_fill.append(prob_short_fill)

        print(f"{date_str:<14} {prob_long_fill:<14.3f} {prob_short_fill:<14.3f} {prob_long_unfilled:<16.3f} {prob_short_unfilled:<16.3f}")

    avg_long = np.mean(overall_long_fill) if overall_long_fill else 0
    avg_short = np.mean(overall_short_fill) if overall_short_fill else 0

    print()
    print(f"Avg long fill prob (heuristic): {avg_long:.3f}")
    print(f"Avg short fill prob (heuristic): {avg_short:.3f}")

    # ============================================================
    # 5. PHASE 2 DECISION
    # ============================================================
    print()
    print("=" * 80)
    print("PHASE 1 SUMMARY")
    print("=" * 80)

    overall_spread_median = ob_df["spread_pct"].median()
    overall_spread_mean = ob_df["spread_pct"].mean()

    print(f"\nOverall spread: median {overall_spread_median:.6f}%, mean {overall_spread_mean:.6f}%")
    print(f"Overall 1-min long fill prob: {avg_long:.3f}")
    print(f"Overall 1-min short fill prob: {avg_short:.3f}")
    print()
    print(f"Phase 2 결정:")
    avg_fill = (avg_long + avg_short) / 2
    if avg_fill > 0.50:
        print(f"  ✅ Avg fill prob {avg_fill:.3f} > 0.50 → 진짜 시뮬 가치 큼 (Phase 2 진행)")
    elif avg_fill > 0.30:
        print(f"  ⚠️  Avg fill prob {avg_fill:.3f} ∈ [0.30, 0.50] → 시도 14 simple 결과 신뢰, 시도 17 운영 권장")
    else:
        print(f"  ❌ Avg fill prob {avg_fill:.3f} < 0.30 → Maker 어려움, Mixed/Taker 운영 강제")

    log.info("Phase 1 complete")


if __name__ == "__main__":
    main()
