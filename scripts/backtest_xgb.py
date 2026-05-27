"""Adaptive Features + XGBoost vs LR (시도 18)."""
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
    log.info("XGBoost vs LR with Adaptive Features (시도 18)")
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
    log.info(f"Features: {len(feature_cols)}")

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
    import xgboost as xgb

    # Vol classifier (LR, 시도 6 동일)
    log.info("\nTraining Vol model (LR)...")
    train_vol_median = train_df_full[vol_target].median()
    y_vol_train = (train_df_full[vol_target] > train_vol_median).astype(int).values

    scaler_vol = StandardScaler()
    X_train_vol_scaled = scaler_vol.fit_transform(X_train)
    X_test_vol_scaled = scaler_vol.transform(X_test)

    lr_vol = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_vol.fit(X_train_vol_scaled, y_vol_train)
    vol_proba_test = lr_vol.predict_proba(X_test_vol_scaled)[:, 1]

    y_vol_test = (test_df_full[vol_target] > train_vol_median).astype(int).values
    log.info(f"  Vol AUC: {roc_auc_score(y_vol_test, vol_proba_test):.3f}")

    # Direction Model 1: LR (시도 17 reproduction)
    log.info("\nTraining Direction model (LR, 시도 17)...")
    T = 0.20
    train_dir_mask = train_df_full[dir_target].abs() > T
    val_dir_mask = val_df_full[dir_target].abs() > T

    X_train_dir = X_train[train_dir_mask].values
    X_val_dir = X_val[val_dir_mask].values

    y_dir_train = (train_df_full.loc[train_dir_mask, dir_target] > 0).astype(int).values
    y_dir_val = (val_df_full.loc[val_dir_mask, dir_target] > 0).astype(int).values

    log.info(f"  Train sample: {len(X_train_dir)}")
    log.info(f"  Val sample: {len(X_val_dir)}")

    scaler_dir = StandardScaler()
    X_train_dir_scaled = scaler_dir.fit_transform(X_train_dir)
    X_val_dir_scaled = scaler_dir.transform(X_val_dir)
    X_test_dir_scaled = scaler_dir.transform(X_test.values)

    lr_dir = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_dir.fit(X_train_dir_scaled, y_dir_train)

    lr_proba_test = lr_dir.predict_proba(X_test_dir_scaled)[:, 1]

    # Direction Model 2: XGBoost (시도 18)
    log.info("\nTraining Direction model (XGBoost, 시도 18)...")

    xgb_dir = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0, gamma=0.1,
        random_state=42, eval_metric="logloss",
        early_stopping_rounds=30,
    )

    xgb_dir.fit(X_train_dir, y_dir_train, eval_set=[(X_val_dir, y_dir_val)], verbose=False)
    log.info(f"  XGBoost best_iter: {xgb_dir.best_iteration}")

    if xgb_dir.best_iteration < 5:
        log.warning("  XGBoost early stop - mode collapse risk")

    xgb_proba_test = xgb_dir.predict_proba(X_test.values)[:, 1]

    # AUC 비교
    print()
    print("=" * 80)
    print("Direction AUC 비교")
    print("=" * 80)

    test_dir_mask = test_df_full[dir_target].abs() > T
    y_dir_test_subset = (test_df_full.loc[test_dir_mask, dir_target] > 0).astype(int).values

    lr_auc = roc_auc_score(y_dir_test_subset, lr_proba_test[test_dir_mask.values])
    xgb_auc = roc_auc_score(y_dir_test_subset, xgb_proba_test[test_dir_mask.values])

    ensemble_proba = (lr_proba_test + xgb_proba_test) / 2
    ensemble_auc = roc_auc_score(y_dir_test_subset, ensemble_proba[test_dir_mask.values])

    print(f"\n  LR (시도 17):           {lr_auc:.3f}")
    print(f"  XGBoost (시도 18):       {xgb_auc:.3f}")
    print(f"  Ensemble (LR + XGB)/2:   {ensemble_auc:.3f}")

    if xgb_dir.best_iteration >= 5:
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": xgb_dir.feature_importances_,
        }).sort_values("importance", ascending=False).head(20)

        print(f"\n  XGBoost Top 20 features:")
        for _, row in importance.iterrows():
            print(f"    {row['feature']:<46} {row['importance']:.4f}")

    # Trading simulation (3 models × 3 fee scenarios)
    print()
    print("=" * 80)
    print("Trading Simulation (3 models × 3 fees)")
    print("=" * 80)

    test_df = test_df_full.copy().reset_index(drop=True)
    test_df["vol_proba"] = vol_proba_test
    test_df["lr_proba"] = lr_proba_test
    test_df["xgb_proba"] = xgb_proba_test
    test_df["ensemble_proba"] = ensemble_proba
    test_df["actual_return"] = test_df[dir_target].values

    ts_col = None
    for c in ["_ts", "ts", "timestamp"]:
        if c in test_df.columns:
            ts_col = c
            break
    test_df = test_df.sort_values(["_source_date", ts_col]).reset_index(drop=True)

    DIR_THRESH = 0.65
    VOL_THRESH = 0.6

    FEE_SCENARIOS = {
        "Maker (-0.05%)": -0.05,
        "Mixed (0.03%)": 0.03,
        "Taker (0.11%)": 0.11,
    }

    MODELS = ["lr_proba", "xgb_proba", "ensemble_proba"]
    MODEL_LABELS = {"lr_proba": "LR (시도 17)", "xgb_proba": "XGBoost", "ensemble_proba": "Ensemble"}

    def backtest(proba_col, fee):
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
                lockout_rows = max(int(3600 / interval_sec), 1)
            else:
                lockout_rows = 60

            i = 0
            n = len(date_df)
            date_trades = []

            while i < n:
                row = date_df.iloc[i]
                vol_proba = row["vol_proba"]
                dir_proba = row[proba_col]
                actual = row["actual_return"]

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
                    pnl = direction * actual - fee
                    date_trades.append({"pnl": pnl, "actual": actual})
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
        wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        return {
            "total_trades": total_trades,
            "daily_avg": daily_avg,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "win_rate": win_rate,
            "wl_ratio": wl_ratio,
            "date_pnls": date_pnls,
        }

    print(f"\n{'Model':<20} {'Fee':<20} {'Trades':<8} {'WR':<8} {'W/L':<8} {'Daily':<10} {'Sharpe':<10} {'MaxDD':<10}")
    print("-" * 110)

    all_results = {}

    for model in MODELS:
        for fee_name, fee in FEE_SCENARIOS.items():
            r = backtest(model, fee)
            all_results[(model, fee_name)] = r
            label = MODEL_LABELS[model]
            print(f"{label:<20} {fee_name:<20} {r['total_trades']:<8} {r['win_rate']:<8.3f} {r['wl_ratio']:<8.2f} {r['daily_avg']:<+10.3f}% {r['sharpe']:<10.2f} {r['max_dd']:<10.3f}%")
        print()

    print()
    print("=" * 80)
    print("Per-date breakdown (Maker)")
    print("=" * 80)
    for model in MODELS:
        r = all_results[(model, "Maker (-0.05%)")]
        print(f"\n{MODEL_LABELS[model]} (Maker):")
        for d in r["date_pnls"]:
            print(f"  {d['date']}: trades={d['n_trades']:<3} pnl={d['pnl_sum']:+.3f}% winrate={d['win_rate']:.3f}")

    print()
    print("=" * 80)
    print("시도 17 (LR) vs 시도 18 (XGBoost / Ensemble) 비교")
    print("=" * 80)

    print(f"\nDirection AUC: LR {lr_auc:.3f}, XGBoost {xgb_auc:.3f}, Ensemble {ensemble_auc:.3f}")
    print()
    print(f"{'Fee':<20} {'LR':<25} {'XGBoost':<25} {'Ensemble':<25}")
    print("-" * 100)

    for fee_name in FEE_SCENARIOS:
        lr_r = all_results[("lr_proba", fee_name)]
        xgb_r = all_results[("xgb_proba", fee_name)]
        ens_r = all_results[("ensemble_proba", fee_name)]

        lr_str = f"{lr_r['daily_avg']:+.2f}% (Sh={lr_r['sharpe']:.2f})"
        xgb_str = f"{xgb_r['daily_avg']:+.2f}% (Sh={xgb_r['sharpe']:.2f})"
        ens_str = f"{ens_r['daily_avg']:+.2f}% (Sh={ens_r['sharpe']:.2f})"

        print(f"{fee_name:<20} {lr_str:<25} {xgb_str:<25} {ens_str:<25}")

    print()
    print("=" * 80)
    print("Best Model per Fee")
    print("=" * 80)
    for fee_name in FEE_SCENARIOS:
        best_model = max(MODELS, key=lambda m: all_results[(m, fee_name)]["sharpe"])
        r = all_results[(best_model, fee_name)]
        print(f"\n{fee_name}:")
        print(f"  Best: {MODEL_LABELS[best_model]} (Sharpe {r['sharpe']:.2f}, daily {r['daily_avg']:+.2f}%, MaxDD {r['max_dd']:.3f}%)")

    print()
    print("=" * 80)
    print("결론")
    print("=" * 80)
    print()
    print("Decision:")
    print("  - XGBoost Sharpe > LR Sharpe -> 비선형 효과 입증, 시도 18 성공")
    print("  - Ensemble Sharpe > 둘 다 -> 결합 효과")
    print("  - LR 이 best -> XGBoost smoothing 또는 overfit (시도 7 패턴)")
    print()
    print("일 1% 달성:")
    for fee_name in FEE_SCENARIOS:
        for model in MODELS:
            r = all_results[(model, fee_name)]
            if r["daily_avg"] >= 1.0:
                print(f"  OK {MODEL_LABELS[model]} ({fee_name}): {r['daily_avg']:+.2f}% (Sh {r['sharpe']:.2f})")

    log.info("Done")


if __name__ == "__main__":
    main()
