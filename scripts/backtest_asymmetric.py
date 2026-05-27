"""Asymmetric Exit (Stop Loss + Take Profit). Uses target_max_drawdown/runup for path."""
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
    log.info("ASYMMETRIC EXIT (시도 16)")
    log.info("=" * 70)

    log.info("\nBuilding datasets...")
    train_df_full = build_split(DATES_TRAIN, log)
    val_df_full = build_split(DATES_VAL, log)
    test_df_full = build_split(DATES_TEST, log)

    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"
    min_col = "target_max_drawdown_3600s"  # 최대 drawdown (negative)
    max_col = "target_max_runup_3600s"     # 최대 runup (positive)

    has_path = (min_col in train_df_full.columns) and (max_col in train_df_full.columns)

    if has_path:
        log.info("PATH OK: target_max_drawdown_3600s / target_max_runup_3600s exist")
    else:
        log.error("Path missing - cannot proceed")
        return

    drop_cols = [vol_target, dir_target, min_col, max_col]

    for df in [train_df_full, val_df_full, test_df_full]:
        df.dropna(subset=drop_cols, inplace=True)

    feature_cols = get_feature_columns(train_df_full)

    X_train_raw = train_df_full.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    train_feature_medians = X_train_raw.median(numeric_only=True)

    def make_X(df, feat_cols, train_medians):
        X = df.reindex(columns=feat_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X_filled = X.fillna(train_medians).fillna(0)
        return X_filled

    X_train = make_X(train_df_full, feature_cols, train_feature_medians)
    X_test = make_X(test_df_full, feature_cols, train_feature_medians)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    log.info("Training Vol model...")
    train_vol_median = train_df_full[vol_target].median()
    y_vol_train = (train_df_full[vol_target] > train_vol_median).astype(int).values

    scaler_vol = StandardScaler()
    X_train_vol_scaled = scaler_vol.fit_transform(X_train)
    X_test_vol_scaled = scaler_vol.transform(X_test)

    lr_vol = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_vol.fit(X_train_vol_scaled, y_vol_train)
    vol_proba_test = lr_vol.predict_proba(X_test_vol_scaled)[:, 1]
    y_vol_test = (test_df_full[vol_target] > train_vol_median).astype(int).values
    vol_auc = roc_auc_score(y_vol_test, vol_proba_test)
    log.info(f"  Vol AUC: {vol_auc:.3f}")

    log.info("Training Direction model...")
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

    test_dir_mask = test_df_full[dir_target].abs() > T
    y_dir_test_subset = (test_df_full.loc[test_dir_mask, dir_target] > 0).astype(int).values
    dir_proba_test_subset = dir_proba_test[test_dir_mask.values]
    dir_auc = roc_auc_score(y_dir_test_subset, dir_proba_test_subset)
    log.info(f"  Direction AUC: {dir_auc:.3f}")

    test_df = test_df_full.copy().reset_index(drop=True)
    test_df["vol_proba"] = vol_proba_test
    test_df["dir_proba"] = dir_proba_test
    test_df["actual_return"] = test_df[dir_target].values
    test_df["min_return"] = test_df[min_col].values
    test_df["max_return"] = test_df[max_col].values

    ts_col = None
    for c in ["_ts", "ts", "timestamp"]:
        if c in test_df.columns:
            ts_col = c
            break
    test_df = test_df.sort_values(["_source_date", ts_col]).reset_index(drop=True)

    def simulate_pnl(direction, actual, min_ret, max_ret, sl_pct, tp_pct, fee):
        """
        min_ret = target_max_drawdown_3600s (negative)
        max_ret = target_max_runup_3600s (positive)
        """
        if direction == 1:  # Long
            hit_sl = min_ret <= -sl_pct
            hit_tp = max_ret >= tp_pct
            if hit_sl and hit_tp:
                raw_pnl = -sl_pct  # 보수적: SL 우선
            elif hit_sl:
                raw_pnl = -sl_pct
            elif hit_tp:
                raw_pnl = tp_pct
            else:
                raw_pnl = actual
        else:  # Short
            hit_sl = max_ret >= sl_pct
            hit_tp = min_ret <= -tp_pct
            if hit_sl and hit_tp:
                raw_pnl = -sl_pct
            elif hit_sl:
                raw_pnl = -sl_pct
            elif hit_tp:
                raw_pnl = tp_pct
            else:
                raw_pnl = -actual
        return raw_pnl - fee

    DIR_THRESH = 0.65
    VOL_THRESH = 0.6

    SL_TP_PAIRS = [
        ("baseline (no SL/TP)", 100.0, 100.0),
        ("SL 0.10, TP 0.30 (1:3)", 0.10, 0.30),
        ("SL 0.15, TP 0.30 (1:2)", 0.15, 0.30),
        ("SL 0.10, TP 0.20 (1:2)", 0.10, 0.20),
        ("SL 0.20, TP 0.40 (1:2)", 0.20, 0.40),
        ("SL 0.05, TP 0.20 (1:4)", 0.05, 0.20),
        ("SL 0.15, TP 0.45 (1:3)", 0.15, 0.45),
    ]

    FEE_SCENARIOS = {
        "Maker (-0.05%)": -0.05,
        "Mixed (0.03%)": 0.03,
        "Taker (0.11%)": 0.11,
    }

    print()
    print("=" * 80)
    print(f"Path-based simulation: {has_path}")
    print("=" * 80)

    all_results = {}

    for sl_tp_name, sl, tp in SL_TP_PAIRS:
        for fee_name, fee in FEE_SCENARIOS.items():
            date_pnls = []
            all_trades = []

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
                    lockout_rows = max(int(3600 / max(interval_sec, 1)), 1)
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

                    if pd.isna(actual):
                        i += 1
                        continue

                    trade = False
                    direction = 0

                    if vol_proba > VOL_THRESH:
                        if dir_proba > DIR_THRESH:
                            trade = True
                            direction = 1
                        elif dir_proba < (1 - DIR_THRESH):
                            trade = True
                            direction = -1

                    if trade:
                        min_ret = row["min_return"]
                        max_ret = row["max_return"]
                        pnl = simulate_pnl(direction, actual, min_ret, max_ret, sl, tp, fee)
                        date_trades.append({
                            "ts": row[ts_col],
                            "direction": direction,
                            "pnl": pnl,
                            "actual": actual,
                        })
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

            total_trades = sum(d["n_trades"] for d in date_pnls)
            daily_pnls = [d["pnl_sum"] for d in date_pnls]
            daily_avg = np.mean(daily_pnls) if daily_pnls else 0
            daily_std = np.std(daily_pnls) if len(daily_pnls) > 1 else 0
            sharpe = daily_avg / max(daily_std, 0.001)

            cum_pnl = np.cumsum([d["pnl_sum"] for d in date_pnls])
            peak = np.maximum.accumulate(cum_pnl) if len(cum_pnl) > 0 else [0]
            drawdowns = peak - cum_pnl if len(cum_pnl) > 0 else [0]
            max_dd = max(drawdowns) if len(drawdowns) > 0 else 0

            wins = [t["pnl"] for t in all_trades if t["pnl"] > 0]
            losses = [t["pnl"] for t in all_trades if t["pnl"] <= 0]
            win_rate = len(wins) / max(len(all_trades), 1)
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0

            all_results[(sl_tp_name, fee_name)] = {
                "total_trades": total_trades,
                "daily_avg": daily_avg,
                "sharpe": sharpe,
                "max_dd": max_dd,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "date_pnls": date_pnls,
            }

    print()
    print(f"{'SL/TP':<32} {'Fee':<18} {'Trades':<8} {'Win%':<8} {'AvgW':<9} {'AvgL':<9} {'Daily':<11} {'Sharpe':<9} {'MaxDD':<9}")
    print("-" * 130)

    for sl_tp_name, _, _ in SL_TP_PAIRS:
        for fee_name in FEE_SCENARIOS:
            r = all_results[(sl_tp_name, fee_name)]
            print(f"{sl_tp_name:<32} {fee_name:<18} {r['total_trades']:<8} {r['win_rate']:<8.3f} {r['avg_win']:<+9.3f} {r['avg_loss']:<+9.3f} {r['daily_avg']:<+11.3f}% {r['sharpe']:<9.2f} {r['max_dd']:<9.3f}%")
        print()

    print()
    print("=" * 80)
    print("BEST SL/TP PER FEE (by Sharpe)")
    print("=" * 80)

    for fee_name in FEE_SCENARIOS:
        best = max(SL_TP_PAIRS, key=lambda x: all_results[(x[0], fee_name)]["sharpe"])
        r = all_results[(best[0], fee_name)]
        print(f"\n{fee_name}:")
        print(f"  Best: {best[0]}")
        print(f"  Daily: {r['daily_avg']:+.3f}%, Sharpe: {r['sharpe']:.2f}, Max DD: {r['max_dd']:.3f}%")
        for d in r["date_pnls"]:
            print(f"    {d['date']}: trades={d['n_trades']:<3} pnl={d['pnl_sum']:+.3f}% winrate={d['win_rate']:.3f}")

    print()
    print("=" * 80)
    print("일 1% 달성 평가")
    print("=" * 80)
    print()
    print(f"{'SL/TP':<32} {'Maker':<24} {'Mixed':<24}")
    print("-" * 90)
    for sl_tp_name, _, _ in SL_TP_PAIRS:
        maker_r = all_results[(sl_tp_name, "Maker (-0.05%)")]
        mixed_r = all_results[(sl_tp_name, "Mixed (0.03%)")]
        maker_str = f"{maker_r['daily_avg']:+.2f}% (Sh={maker_r['sharpe']:.2f})"
        mixed_str = f"{mixed_r['daily_avg']:+.2f}% (Sh={mixed_r['sharpe']:.2f})"
        print(f"{sl_tp_name:<32} {maker_str:<24} {mixed_str:<24}")

    print()
    print("Decision guide:")
    print("  - Sharpe > 1.5 -> 시도 14 (Maker fill)")
    print("  - Sharpe 1.0-1.5 -> 시도 8 (Regime conditional)")
    print("  - Sharpe < 1.0 -> tweak more")

    log.info("Done")


if __name__ == "__main__":
    main()
