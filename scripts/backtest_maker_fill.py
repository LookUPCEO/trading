"""Maker Fill Rate 추정 (시도 14 simple)."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.ml.data_prep import (
    DATES_TRAIN, DATES_VAL, DATES_TEST,
    build_split, get_feature_columns,
)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("MAKER FILL RATE 추정 (시도 14 simple)")
    log.info("=" * 70)

    # Build datasets
    log.info("\nBuilding datasets...")
    train_df_full = build_split(DATES_TRAIN, log)
    val_df_full = build_split(DATES_VAL, log)
    test_df_full = build_split(DATES_TEST, log)

    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"

    for df in [train_df_full, val_df_full, test_df_full]:
        df.dropna(subset=[vol_target, dir_target], inplace=True)

    feature_cols = get_feature_columns(train_df_full)

    X_train_raw = train_df_full.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    train_feature_medians = X_train_raw.median(numeric_only=True)

    def make_X(df, feat_cols, train_medians):
        X = df.reindex(columns=feat_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X_filled = X.fillna(train_medians).fillna(0)
        return X_filled

    X_train = make_X(train_df_full, feature_cols, train_feature_medians)
    X_val = make_X(val_df_full, feature_cols, train_feature_medians)
    X_test = make_X(test_df_full, feature_cols, train_feature_medians)

    # Train models (시도 6 reproduction)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    log.info("Training models (시도 6 baseline)...")
    train_vol_median = train_df_full[vol_target].median()
    y_vol_train = (train_df_full[vol_target] > train_vol_median).astype(int).values

    scaler_vol = StandardScaler()
    X_train_vol_scaled = scaler_vol.fit_transform(X_train)
    X_test_vol_scaled = scaler_vol.transform(X_test)

    lr_vol = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_vol.fit(X_train_vol_scaled, y_vol_train)
    vol_proba_test = lr_vol.predict_proba(X_test_vol_scaled)[:, 1]

    T = 0.20
    train_dir_mask = train_df_full[dir_target].abs() > T
    X_train_dir = X_train[train_dir_mask].values
    y_dir_train = (train_df_full.loc[train_dir_mask, dir_target] > 0).astype(int).values

    scaler_dir = StandardScaler()
    X_train_dir_scaled = scaler_dir.fit_transform(X_train_dir)
    X_test_dir_scaled = scaler_dir.transform(X_test.values)

    lr_dir = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_dir.fit(X_train_dir_scaled, y_dir_train)
    dir_proba_test = lr_dir.predict_proba(X_test_dir_scaled)[:, 1]

    test_df = test_df_full.copy().reset_index(drop=True)
    test_df["vol_proba"] = vol_proba_test
    test_df["dir_proba"] = dir_proba_test
    test_df["actual_return"] = test_df[dir_target].values

    # Spread 통계 (test)
    spread_col = "ob_spread_pct"
    if spread_col not in test_df.columns:
        log.error(f"{spread_col} not found!")
        return

    print()
    print("=" * 80)
    print("Spread distribution (test)")
    print("=" * 80)
    spread_q = test_df[spread_col].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    print(spread_q)

    # Spread bin 정의 (training 분포 기반)
    train_spread = train_df_full[spread_col]
    SPREAD_TIGHT = train_spread.quantile(0.33)  # 좁은 spread (lower 33%)
    SPREAD_LOOSE = train_spread.quantile(0.67)  # 넓은 spread (upper 33%)

    log.info(f"\nSpread thresholds (train-based):")
    log.info(f"  Tight: < {SPREAD_TIGHT:.6f}")
    log.info(f"  Mid:   {SPREAD_TIGHT:.6f} - {SPREAD_LOOSE:.6f}")
    log.info(f"  Loose: > {SPREAD_LOOSE:.6f}")

    # ============================================================
    # Maker fill rate 모델 (heuristic)
    # ============================================================
    def get_maker_fill_rate(spread, model="conservative"):
        """
        Maker limit order fill rate 추정.
        - Tight spread: 경쟁 많음, fill 어려움
        - Loose spread: 경쟁 적음, fill 쉬움
        """
        if model == "conservative":
            if spread < SPREAD_TIGHT: return 0.40
            elif spread < SPREAD_LOOSE: return 0.65
            else: return 0.85
        elif model == "moderate":
            if spread < SPREAD_TIGHT: return 0.55
            elif spread < SPREAD_LOOSE: return 0.75
            else: return 0.90
        elif model == "optimistic":
            if spread < SPREAD_TIGHT: return 0.70
            elif spread < SPREAD_LOOSE: return 0.85
            else: return 0.95
        else:
            return 1.0  # baseline (no fill issues)

    ts_col = None
    for c in ["_ts", "ts", "timestamp"]:
        if c in test_df.columns:
            ts_col = c
            break
    test_df = test_df.sort_values(["_source_date", ts_col]).reset_index(drop=True)

    DIR_THRESH = 0.65
    VOL_THRESH = 0.6

    # Fee scenarios with maker fill considerations
    SCENARIOS = [
        # (name, fill_model, unfilled_action, taker_fee)
        ("ideal (baseline, 100% maker)",     "ideal",       "miss",    0.11),
        ("optimistic (70-95% maker)",         "optimistic",  "miss",    0.11),
        ("optimistic + force taker",          "optimistic",  "taker",   0.11),
        ("moderate (55-90% maker)",           "moderate",    "miss",    0.11),
        ("moderate + force taker",            "moderate",    "taker",   0.11),
        ("conservative (40-85% maker)",       "conservative","miss",    0.11),
        ("conservative + force taker",        "conservative","taker",   0.11),
    ]

    MAKER_FEE = -0.05

    print()
    print("=" * 100)
    print("MAKER FILL RATE SIMULATION")
    print("=" * 100)
    print(f"\n{'Scenario':<40} {'Trades(M+T)':<15} {'Daily':<10} {'Sharpe':<10} {'MaxDD':<10}")
    print("-" * 100)

    np.random.seed(42)
    all_results = {}

    for scenario_name, fill_model, unfilled_action, taker_fee in SCENARIOS:
        date_pnls = []
        all_trades = []
        n_maker = 0
        n_taker = 0
        n_miss = 0

        for date_str in DATES_TEST:
            date_df = test_df[test_df["_source_date"] == date_str].copy().sort_values(ts_col).reset_index(drop=True)
            if len(date_df) == 0:
                continue

            if len(date_df) > 2:
                ts_diffs = date_df[ts_col].diff().dropna()
                median_diff = ts_diffs.median()
                if hasattr(median_diff, 'total_seconds'):
                    interval_sec = median_diff.total_seconds()
                else:
                    interval_sec = median_diff / 1e9 if median_diff > 1e9 else median_diff
                lockout_rows = max(int(3600 / interval_sec), 1)
            else:
                lockout_rows = 60

            i = 0
            n = len(date_df)
            date_trades = []

            while i < n:
                row = date_df.iloc[i]
                vol_proba = row["vol_proba"]
                dir_proba = row["dir_proba"]
                actual = row["actual_return"]
                spread = row[spread_col] if not pd.isna(row[spread_col]) else SPREAD_LOOSE

                if pd.isna(actual):
                    i += 1
                    continue

                trade = False
                direction = 0

                if vol_proba > VOL_THRESH:
                    if dir_proba > DIR_THRESH:
                        trade = True; direction = 1
                    elif dir_proba < (1 - DIR_THRESH):
                        trade = True; direction = -1

                if trade:
                    # Maker fill 시뮬
                    if fill_model == "ideal":
                        # Always fill at maker
                        fee = MAKER_FEE
                        pnl = direction * actual - fee
                        date_trades.append({"pnl": pnl, "type": "maker"})
                        n_maker += 1
                    else:
                        fill_rate = get_maker_fill_rate(spread, fill_model)
                        filled = np.random.random() < fill_rate

                        if filled:
                            fee = MAKER_FEE
                            pnl = direction * actual - fee
                            date_trades.append({"pnl": pnl, "type": "maker"})
                            n_maker += 1
                        else:
                            if unfilled_action == "miss":
                                # Skip trade
                                n_miss += 1
                            elif unfilled_action == "taker":
                                # Force taker
                                fee = taker_fee
                                pnl = direction * actual - fee
                                date_trades.append({"pnl": pnl, "type": "taker"})
                                n_taker += 1

                    i += lockout_rows
                else:
                    i += 1

            if date_trades:
                pnl_sum = sum(t["pnl"] for t in date_trades)
                wins = sum(1 for t in date_trades if t["pnl"] > 0)
                date_pnls.append({
                    "date": date_str,
                    "n_trades": len(date_trades),
                    "pnl_sum": pnl_sum,
                    "win_rate": wins / len(date_trades),
                })
                all_trades.extend(date_trades)
            else:
                date_pnls.append({"date": date_str, "n_trades": 0, "pnl_sum": 0, "win_rate": 0})

        # Aggregate
        total_trades = sum(d["n_trades"] for d in date_pnls)
        daily_pnls = [d["pnl_sum"] for d in date_pnls]
        daily_avg = np.mean(daily_pnls) if daily_pnls else 0
        daily_std = np.std(daily_pnls) if len(daily_pnls) > 1 else 0
        sharpe = daily_avg / max(daily_std, 0.001)

        cum_pnl = np.cumsum([d["pnl_sum"] for d in date_pnls])
        peak = np.maximum.accumulate(cum_pnl) if len(cum_pnl) > 0 else [0]
        drawdowns = peak - cum_pnl if len(cum_pnl) > 0 else [0]
        max_dd = max(drawdowns) if len(drawdowns) > 0 else 0

        all_results[scenario_name] = {
            "total_trades": total_trades,
            "n_maker": n_maker,
            "n_taker": n_taker,
            "n_miss": n_miss,
            "daily_avg": daily_avg,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "date_pnls": date_pnls,
        }

        trades_str = f"{n_maker}M+{n_taker}T (-{n_miss}miss)"
        print(f"{scenario_name:<40} {trades_str:<15} {daily_avg:<+10.3f}% {sharpe:<10.2f} {max_dd:<10.3f}%")

    # ============================================================
    # 일 1% 평가
    # ============================================================
    print()
    print("=" * 80)
    print("일 1% 달성 평가")
    print("=" * 80)
    print()
    print(f"{'Scenario':<40} {'Daily':<10} {'Sharpe':<10} {'1% 달성':<12}")
    print("-" * 80)
    for name, r in all_results.items():
        achieve = "OK" if r["daily_avg"] >= 1.0 else f"NO ({r['daily_avg']:+.2f}%)"
        print(f"{name:<40} {r['daily_avg']:<+10.3f}% {r['sharpe']:<10.2f} {achieve:<12}")

    print()
    print("=" * 80)
    print("Per-date breakdown - Best scenario")
    print("=" * 80)
    best = max(all_results.items(), key=lambda x: x[1]["sharpe"])
    print(f"\nBest: {best[0]}")
    for d in best[1]["date_pnls"]:
        print(f"  {d['date']}: trades={d['n_trades']:<3} pnl={d['pnl_sum']:+.3f}% winrate={d['win_rate']:.3f}")

    print()
    print("=" * 80)
    print("결론")
    print("=" * 80)
    print()
    print("Realistic 시나리오 (force taker on unfilled):")
    print(f"  - Conservative + taker: {all_results['conservative + force taker']['daily_avg']:+.2f}% (Sh {all_results['conservative + force taker']['sharpe']:.2f})")
    print(f"  - Moderate + taker:     {all_results['moderate + force taker']['daily_avg']:+.2f}% (Sh {all_results['moderate + force taker']['sharpe']:.2f})")
    print(f"  - Optimistic + taker:   {all_results['optimistic + force taker']['daily_avg']:+.2f}% (Sh {all_results['optimistic + force taker']['sharpe']:.2f})")
    print()
    print("Miss 시나리오 (skip unfilled):")
    print(f"  - Conservative: {all_results['conservative (40-85% maker)']['daily_avg']:+.2f}% (Sh {all_results['conservative (40-85% maker)']['sharpe']:.2f})")
    print(f"  - Moderate:     {all_results['moderate (55-90% maker)']['daily_avg']:+.2f}% (Sh {all_results['moderate (55-90% maker)']['sharpe']:.2f})")
    print()
    print("Decision:")
    print("  - Realistic 시나리오 daily > 1.0 -> 시도 14 정확 버전")
    print("  - 0.5-1.0 -> 시도 9 (Ensemble) 또는 14 정확")
    print("  - < 0.5 -> AUC 향상이 더 시급, 시도 9 또는 10")

    log.info("Done")


if __name__ == "__main__":
    main()
