"""Vol + Direction Combined Strategy with Trading Simulation."""
from __future__ import annotations

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
    log.info("VOL + DIRECTION COMBINED STRATEGY (시도 12)")
    log.info("=" * 70)

    log.info(f"\nBuilding datasets...")
    train_df_full = build_split(DATES_TRAIN, log)
    val_df_full = build_split(DATES_VAL, log)
    test_df_full = build_split(DATES_TEST, log)

    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"

    if vol_target not in train_df_full.columns or dir_target not in train_df_full.columns:
        log.error("Targets missing")
        return

    for df in [train_df_full, val_df_full, test_df_full]:
        before = len(df)
        df.dropna(subset=[vol_target, dir_target], inplace=True)
        log.info(f"  Drop NaN: {before} → {len(df)}")

    feature_cols = get_feature_columns(train_df_full)
    log.info(f"\nFeatures: {len(feature_cols)}")

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

    print()
    print("=" * 80)
    print("MODEL 1: Vol Classifier")
    print("=" * 80)

    train_vol_median = train_df_full[vol_target].median()
    log.info(f"  Vol threshold: {train_vol_median:.6f}")

    y_vol_train = (train_df_full[vol_target] > train_vol_median).astype(int).values
    y_vol_val = (val_df_full[vol_target] > train_vol_median).astype(int).values
    y_vol_test = (test_df_full[vol_target] > train_vol_median).astype(int).values

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score

    scaler_vol = StandardScaler()
    X_train_vol_scaled = scaler_vol.fit_transform(X_train)
    X_val_vol_scaled = scaler_vol.transform(X_val)
    X_test_vol_scaled = scaler_vol.transform(X_test)

    lr_vol = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_vol.fit(X_train_vol_scaled, y_vol_train)

    vol_proba_test = lr_vol.predict_proba(X_test_vol_scaled)[:, 1]
    vol_auc = roc_auc_score(y_vol_test, vol_proba_test)
    log.info(f"  Vol model test AUC: {vol_auc:.3f}")

    print()
    print("=" * 80)
    print("MODEL 2: Direction Classifier (1h, T=0.20)")
    print("=" * 80)

    T = 0.20

    train_dir_mask = train_df_full[dir_target].abs() > T
    val_dir_mask = val_df_full[dir_target].abs() > T

    X_train_dir = X_train[train_dir_mask].values
    X_val_dir = X_val[val_dir_mask].values

    y_dir_train = (train_df_full.loc[train_dir_mask, dir_target] > 0).astype(int).values
    y_dir_val = (val_df_full.loc[val_dir_mask, dir_target] > 0).astype(int).values

    log.info(f"  Train dir samples: {len(X_train_dir)}")
    log.info(f"  Val   dir samples: {len(X_val_dir)}")

    scaler_dir = StandardScaler()
    X_train_dir_scaled = scaler_dir.fit_transform(X_train_dir)
    X_val_dir_scaled = scaler_dir.transform(X_val_dir)
    X_test_dir_scaled = scaler_dir.transform(X_test.values)

    lr_dir = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_dir.fit(X_train_dir_scaled, y_dir_train)

    dir_proba_test = lr_dir.predict_proba(X_test_dir_scaled)[:, 1]

    test_dir_mask = test_df_full[dir_target].abs() > T
    y_dir_test_subset = (test_df_full.loc[test_dir_mask, dir_target] > 0).astype(int).values
    dir_proba_test_subset = dir_proba_test[test_dir_mask.values]

    dir_auc = roc_auc_score(y_dir_test_subset, dir_proba_test_subset)
    log.info(f"  Direction model test AUC (T>0.20 subset): {dir_auc:.3f}")

    print()
    print("=" * 80)
    print("PHASE 3: TRADING SIMULATION")
    print("=" * 80)

    test_df = test_df_full.copy().reset_index(drop=True)
    test_df["vol_proba"] = vol_proba_test
    test_df["dir_proba"] = dir_proba_test
    test_df["actual_return"] = test_df[dir_target].values

    STRATEGIES = [
        ("baseline_dir_only_0.55", 0.0, 0.05, "fixed"),
        ("baseline_dir_only_0.55_vol_filter", 0.6, 0.05, "fixed"),
        ("dir_strong_only", 0.0, 0.10, "fixed"),
        ("dir_strong + vol", 0.6, 0.10, "fixed"),
        ("dir_very_strong", 0.0, 0.15, "fixed"),
        ("dir_very_strong + vol", 0.6, 0.15, "fixed"),
        ("kelly_sizing", 0.6, 0.05, "kelly"),
    ]

    TRADING_FEE = 0.11

    print(f"\n{'Strategy':<40} {'Trades':<10} {'WinRate':<10} {'AvgPnL%':<10} {'TotalPnL%':<12} {'DailyAvg%':<12} {'Sharpe':<10}")
    print("-" * 110)

    all_strategies = {}

    for strat_name, vol_min, dir_strength, sizing in STRATEGIES:
        df = test_df.copy()

        df["long_signal"] = df["dir_proba"] > (0.5 + dir_strength)
        df["short_signal"] = df["dir_proba"] < (0.5 - dir_strength)

        if vol_min > 0:
            df["vol_pass"] = df["vol_proba"] > vol_min
        else:
            df["vol_pass"] = True

        df["trade_long"] = df["long_signal"] & df["vol_pass"]
        df["trade_short"] = df["short_signal"] & df["vol_pass"]
        df["trade_any"] = df["trade_long"] | df["trade_short"]

        if df["trade_any"].sum() == 0:
            print(f"{strat_name:<40} 0 trades")
            continue

        if sizing == "kelly":
            df["size"] = (df["dir_proba"] - 0.5).abs() * 4
            df["size"] = df["size"].clip(0, 1)
        else:
            df["size"] = 1.0

        df["pnl_pct"] = 0.0
        df.loc[df["trade_long"], "pnl_pct"] = df.loc[df["trade_long"], "actual_return"] * df.loc[df["trade_long"], "size"] - TRADING_FEE * df.loc[df["trade_long"], "size"]
        df.loc[df["trade_short"], "pnl_pct"] = -df.loc[df["trade_short"], "actual_return"] * df.loc[df["trade_short"], "size"] - TRADING_FEE * df.loc[df["trade_short"], "size"]

        trades_df = df[df["trade_any"]].copy()
        n_trades = len(trades_df)
        wins = (trades_df["pnl_pct"] > 0).sum()
        win_rate = wins / max(n_trades, 1)
        avg_pnl = trades_df["pnl_pct"].mean()
        total_pnl = trades_df["pnl_pct"].sum()

        daily_pnls = []
        for date_str in DATES_TEST:
            sub = trades_df[trades_df["_source_date"] == date_str]
            if len(sub) > 0:
                daily_pnls.append({
                    "date": date_str,
                    "n_trades": len(sub),
                    "pnl_sum": sub["pnl_pct"].sum(),
                    "win_rate": (sub["pnl_pct"] > 0).mean(),
                })

        if daily_pnls:
            daily_avg = np.mean([d["pnl_sum"] for d in daily_pnls])
            daily_std = np.std([d["pnl_sum"] for d in daily_pnls])
            sharpe = daily_avg / max(daily_std, 0.001)
        else:
            daily_avg = 0
            sharpe = 0

        print(f"{strat_name:<40} {n_trades:<10} {win_rate:<10.3f} {avg_pnl:<10.3f} {total_pnl:<12.2f} {daily_avg:<12.2f} {sharpe:<10.2f}")

        all_strategies[strat_name] = {
            "n_trades": n_trades, "win_rate": win_rate,
            "avg_pnl": avg_pnl, "total_pnl": total_pnl,
            "daily_avg": daily_avg, "sharpe": sharpe,
            "daily_pnls": daily_pnls,
        }

    print()
    print("=" * 80)
    print("BEST STRATEGY BREAKDOWN (per-date)")
    print("=" * 80)

    if all_strategies:
        best_strat = max(all_strategies.items(), key=lambda x: x[1]["daily_avg"])
        print(f"\nBest: {best_strat[0]}")
        print(f"  Daily avg PnL: {best_strat[1]['daily_avg']:+.3f}%")
        print(f"  Win rate: {best_strat[1]['win_rate']:.3f}")
        print(f"  Total trades: {best_strat[1]['n_trades']}")
        print(f"\nPer-date:")
        for d in best_strat[1]["daily_pnls"]:
            print(f"  {d['date']}: trades={d['n_trades']:<5} pnl={d['pnl_sum']:+.3f}% winrate={d['win_rate']:.3f}")

    print()
    print("=" * 80)
    print("일 1% 달성 가능성 평가")
    print("=" * 80)
    print()
    print(f"{'Strategy':<40} {'Daily avg':<12} {'1% 달성?':<15}")
    print("-" * 80)

    for name, r in all_strategies.items():
        achieve_1pct = "PASS" if r["daily_avg"] >= 1.0 else f"FAIL ({r['daily_avg']:+.2f}%)"
        print(f"{name:<40} {r['daily_avg']:<+12.3f}% {achieve_1pct:<15}")

    print()
    print("핵심 결론:")
    print("  - Daily avg > 1.0 -> 일 1% 목표 달성")
    print("  - Daily avg 0.3-1.0 -> 큰 진전")
    print("  - Daily avg 0-0.3 -> 약함")
    print("  - Daily avg < 0 -> 거래비용 못 이김")
    print()
    print("거래비용 0.11% (Bybit taker x 2) 가 중요한 변수.")
    print("Maker fee (-0.025%) 사용 시 비용 ~0.005% -> 큰 향상 가능.")

    log.info("\nAnalysis complete")


if __name__ == "__main__":
    main()
