"""Realistic Backtest: 1h cycle, no position overlap, maker/taker comparison."""
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
    log.info("REALISTIC BACKTEST: 1h cycle, no overlap (시도 13)")
    log.info("=" * 70)

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

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    log.info("\nTraining Vol model...")
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

    ts_col = None
    for c in ["_ts", "ts", "timestamp"]:
        if c in test_df.columns:
            ts_col = c
            break
    if ts_col is None:
        log.error(f"No timestamp col. Cols: {test_df.columns[:10].tolist()}")
        return

    log.info(f"Using timestamp col: {ts_col}")

    test_df = test_df.sort_values(["_source_date", ts_col]).reset_index(drop=True)

    print()
    print("=" * 80)
    print("1H CYCLE BACKTEST (no overlap)")
    print("=" * 80)

    FEE_SCENARIOS = {
        "Taker (0.055% × 2 = 0.11%)": 0.11,
        "Maker (-0.025% × 2 = -0.05%)": -0.05,
        "Mixed (Taker entry, Maker exit)": 0.03,
        "No fee (theoretical)": 0.0,
    }

    DIR_THRESH = 0.65
    VOL_THRESH = 0.6

    print(f"\nStrategy: dir_proba > {DIR_THRESH} OR < {1-DIR_THRESH}, vol_proba > {VOL_THRESH}")
    print(f"1h cycle: trade 후 60분 lockout")
    print()

    all_fee_results = {}

    for fee_name, fee_pct in FEE_SCENARIOS.items():
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
                    pnl = direction * actual - fee_pct
                    date_trades.append({
                        "ts": row[ts_col],
                        "direction": direction,
                        "dir_proba": dir_proba,
                        "vol_proba": vol_proba,
                        "actual_return": actual,
                        "pnl": pnl,
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
                    "lockout_rows": lockout_rows,
                })
                all_trades.extend(date_trades)
            else:
                date_pnls.append({
                    "date": date_str,
                    "n_trades": 0,
                    "pnl_sum": 0.0,
                    "win_rate": 0,
                    "lockout_rows": lockout_rows,
                })

        total_trades = sum(d["n_trades"] for d in date_pnls)
        total_pnl = sum(d["pnl_sum"] for d in date_pnls)
        daily_pnls = [d["pnl_sum"] for d in date_pnls]
        daily_avg = np.mean(daily_pnls) if daily_pnls else 0
        daily_std = np.std(daily_pnls) if len(daily_pnls) > 1 else 0
        sharpe = daily_avg / max(daily_std, 0.001)

        cum_pnl = np.cumsum([d["pnl_sum"] for d in date_pnls])
        peak = np.maximum.accumulate(cum_pnl) if len(cum_pnl) > 0 else [0]
        drawdowns = peak - cum_pnl if len(cum_pnl) > 0 else [0]
        max_dd = max(drawdowns) if len(drawdowns) > 0 else 0

        if all_trades:
            wins = [t["pnl"] for t in all_trades if t["pnl"] > 0]
            losses = [t["pnl"] for t in all_trades if t["pnl"] <= 0]
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            win_rate = len(wins) / len(all_trades)
            wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = avg_win = avg_loss = wl_ratio = 0

        all_fee_results[fee_name] = {
            "date_pnls": date_pnls,
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "daily_avg": daily_avg,
            "daily_std": daily_std,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "wl_ratio": wl_ratio,
        }

        print(f"\n--- {fee_name} ---")
        print(f"  Total trades: {total_trades}")
        print(f"  Win rate: {win_rate:.3f}")
        print(f"  Avg win: {avg_win:+.3f}%, Avg loss: {avg_loss:+.3f}%, W/L ratio: {wl_ratio:.2f}")
        print(f"  Daily avg: {daily_avg:+.3f}%, Std: {daily_std:.3f}%, Sharpe: {sharpe:.2f}")
        print(f"  Max drawdown: {max_dd:.3f}%")
        print(f"  Per-date:")
        for d in date_pnls:
            print(f"    {d['date']}: trades={d['n_trades']:<3} pnl={d['pnl_sum']:+.3f}% winrate={d['win_rate']:.3f}")

    print()
    print("=" * 80)
    print("일 1% 달성 가능성 평가 (4 test dates 평균)")
    print("=" * 80)
    print()
    print(f"{'Fee Scenario':<40} {'Daily Avg':<12} {'Sharpe':<10} {'Max DD':<10} {'1% 달성':<12}")
    print("-" * 90)
    for fee_name, r in all_fee_results.items():
        achieve = "PASS" if r["daily_avg"] >= 1.0 else f"FAIL ({r['daily_avg']:.2f}%)"
        print(f"{fee_name:<40} {r['daily_avg']:<+12.3f}% {r['sharpe']:<10.2f} {r['max_dd']:<10.3f}% {achieve:<12}")

    print()
    print("핵심 결론:")
    print(f"  Taker fee (0.11%) 시: daily_avg 가 가장 보수적")
    print(f"  Maker fee (-0.05%) 시: 가장 낙관적, 그러나 maker 어려움")
    print(f"  Mixed (0.03%): 현실적 시나리오")
    print()
    print("일 1% 달성 시:")
    print(f"  - Daily avg >= 1.0% AND Sharpe >= 1.0 -> 안정적 일 1% 달성")
    print(f"  - Daily avg >= 1.0% AND Sharpe < 1.0 -> 변동성 큰 1% (high risk)")
    print(f"  - Daily avg < 1.0% -> 추가 시도 필요")

    log.info("\nBacktest complete")


if __name__ == "__main__":
    main()
