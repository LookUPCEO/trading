"""1-hour Direction Classifier + OBI Strength Filter."""
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
    log.info("1-HOUR DIRECTION + OBI STRENGTH (시도 4)")
    log.info("=" * 70)

    log.info(f"\nBuilding datasets...")
    train_df_full = build_split(DATES_TRAIN, log)
    val_df_full = build_split(DATES_VAL, log)
    test_df_full = build_split(DATES_TEST, log)

    target_col = "target_return_3600s"

    if target_col not in train_df_full.columns:
        log.error(f"{target_col} not in dataset")
        return

    for df in [train_df_full, val_df_full, test_df_full]:
        before = len(df)
        df.dropna(subset=[target_col], inplace=True)
        log.info(f"  Drop NaN: {before} → {len(df)}")

    feature_cols = get_feature_columns(train_df_full)
    log.info(f"\nFeatures: {len(feature_cols)}")

    obi_col = "ob_obi_top1"
    if obi_col not in train_df_full.columns:
        log.warning(f"{obi_col} not found, using ob_obi_top5")
        obi_col = "ob_obi_top5"
    log.info(f"OBI col: {obi_col}")

    X_train_raw = train_df_full.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    train_feature_medians = X_train_raw.median(numeric_only=True)

    def make_xy(df, feat_cols, train_medians, target):
        X = df.reindex(columns=feat_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X_filled = X.fillna(train_medians).fillna(0)
        y = df[target].values
        return X, X_filled, y

    def per_class(y_true, y_pred):
        c0 = y_true == 0; c1 = y_true == 1
        c0_acc = (y_pred[c0] == 0).mean() if c0.sum() > 0 else 0
        c1_acc = (y_pred[c1] == 1).mean() if c1.sum() > 0 else 0
        return c0_acc, c1_acc, (c0_acc + c1_acc) / 2

    T = 0.20

    VARIANTS = [
        ("1h + T=0.20 (no OBI)",     0.0),
        ("1h + T=0.20 + |OBI|>0.2",  0.2),
        ("1h + T=0.20 + |OBI|>0.3",  0.3),
        ("1h + T=0.20 + |OBI|>0.4",  0.4),
        ("1h + T=0.20 + |OBI|>0.5",  0.5),
    ]

    all_results = {}

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    for variant_name, obi_thresh in VARIANTS:
        print()
        print("=" * 80)
        print(f"VARIANT: {variant_name}")
        print("=" * 80)

        def filter_df(df):
            mask = df[target_col].abs() > T
            if obi_thresh > 0:
                mask &= df[obi_col].abs() >= obi_thresh
            return df[mask].copy()

        train_df = filter_df(train_df_full)
        val_df = filter_df(val_df_full)
        test_df = filter_df(test_df_full)

        train_retention = len(train_df) / len(train_df_full) * 100
        val_retention = len(val_df) / len(val_df_full) * 100
        test_retention = len(test_df) / len(test_df_full) * 100

        print(f"\n  Sample retention:")
        print(f"    Train: {len(train_df):>6} ({train_retention:.1f}%)")
        print(f"    Val:   {len(val_df):>6} ({val_retention:.1f}%)")
        print(f"    Test:  {len(test_df):>6} ({test_retention:.1f}%)")

        if len(train_df) < 300 or len(val_df) < 50 or len(test_df) < 50:
            print(f"    WARN: Sample too small, skipping")
            continue

        train_df["target_binary"] = (train_df[target_col] > 0).astype(int)
        val_df["target_binary"] = (val_df[target_col] > 0).astype(int)
        test_df["target_binary"] = (test_df[target_col] > 0).astype(int)

        if test_df["target_binary"].nunique() < 2 or val_df["target_binary"].nunique() < 2:
            print(f"    WARN: Single class, skipping")
            continue

        print(f"  Up fraction: train {train_df['target_binary'].mean():.3f}, val {val_df['target_binary'].mean():.3f}, test {test_df['target_binary'].mean():.3f}")

        X_train, X_train_filled, y_train = make_xy(train_df, feature_cols, train_feature_medians, "target_binary")
        X_val, X_val_filled, y_val = make_xy(val_df, feature_cols, train_feature_medians, "target_binary")
        X_test, X_test_filled, y_test = make_xy(test_df, feature_cols, train_feature_medians, "target_binary")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
        X_val_scaled = scaler.transform(X_val_filled)
        X_test_scaled = scaler.transform(X_test_filled)

        lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lr.fit(X_train_scaled, y_train)

        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=2.0, gamma=0.1,
            random_state=42, eval_metric="logloss", early_stopping_rounds=30,
        )
        xgb_model.fit(X_train_filled, y_train, eval_set=[(X_val_filled, y_val)], verbose=False)

        lr_proba_train = lr.predict_proba(X_train_scaled)[:, 1]
        lr_proba_val = lr.predict_proba(X_val_scaled)[:, 1]
        lr_proba_test = lr.predict_proba(X_test_scaled)[:, 1]
        xgb_proba_test = xgb_model.predict_proba(X_test_filled)[:, 1]

        lr_pred_train = (lr_proba_train > 0.5).astype(int)
        lr_pred_test = (lr_proba_test > 0.5).astype(int)
        xgb_pred_test = (xgb_proba_test > 0.5).astype(int)

        lr_auc = roc_auc_score(y_test, lr_proba_test)
        xgb_auc = roc_auc_score(y_test, xgb_proba_test)
        c0_lr, c1_lr, lr_bal = per_class(y_test, lr_pred_test)
        c0_xgb, c1_xgb, xgb_bal = per_class(y_test, xgb_pred_test)
        _, _, lr_train_bal = per_class(y_train, lr_pred_train)

        best_thresh = 0.5
        best_bal_val = 0
        for t in np.arange(0.30, 0.70, 0.01):
            pred = (lr_proba_val > t).astype(int)
            _, _, bal = per_class(y_val, pred)
            if bal > best_bal_val:
                best_bal_val = bal
                best_thresh = t

        lr_pred_test_best = (lr_proba_test > best_thresh).astype(int)
        _, _, lr_test_bal_at_val = per_class(y_test, lr_pred_test_best)

        print(f"\n  XGBoost best_iter: {xgb_model.best_iteration}")
        if xgb_model.best_iteration < 5:
            print(f"  WARN: XGBoost early stop")

        print(f"  LR  test AUC: {lr_auc:.3f}, bal_acc: {lr_bal:.3f} (Down {c0_lr:.3f}, Up {c1_lr:.3f})")
        print(f"  XGB test AUC: {xgb_auc:.3f}, bal_acc: {xgb_bal:.3f} (Down {c0_xgb:.3f}, Up {c1_xgb:.3f})")
        print(f"  Val-selected thresh: {best_thresh:.2f}, test bal_acc: {lr_test_bal_at_val:.3f}")
        print(f"  Train-test gap (LR bal): {lr_train_bal - lr_bal:+.3f}")

        print(f"\n  Per-date Test AUC:")
        test_df_copy = test_df.copy().reset_index(drop=True)
        test_df_copy["lr_proba"] = lr_proba_test
        test_df_copy["xgb_proba"] = xgb_proba_test

        for date_str in DATES_TEST:
            sub = test_df_copy[test_df_copy["_source_date"] == date_str]
            if len(sub) > 5 and sub["target_binary"].nunique() > 1:
                try:
                    lr_auc_d = roc_auc_score(sub["target_binary"], sub["lr_proba"])
                    xgb_auc_d = roc_auc_score(sub["target_binary"], sub["xgb_proba"])
                    print(f"    {date_str}: n={len(sub):<4}, up={sub['target_binary'].mean():.2f}, LR {lr_auc_d:.3f}, XGB {xgb_auc_d:.3f}")
                except:
                    pass
            else:
                print(f"    {date_str}: insufficient data")

        all_results[variant_name] = {
            "obi_thresh": obi_thresh,
            "lr_auc": lr_auc,
            "xgb_auc": xgb_auc,
            "lr_bal": lr_bal,
            "xgb_bal": xgb_bal,
            "lr_test_bal_at_val": lr_test_bal_at_val,
            "best_iter": xgb_model.best_iteration,
            "best_val_thresh": best_thresh,
            "test_size": len(test_df),
            "test_retention": test_retention,
            "train_test_gap": lr_train_bal - lr_bal,
        }

    print()
    print("=" * 100)
    print("FINAL COMPARISON: 1h Direction + OBI Strength")
    print("=" * 100)
    print()
    print(f"{'Variant':<35} {'Test n':<10} {'Retain':<10} {'LR AUC':<10} {'XGB AUC':<10} {'@val_thr':<10} {'Best iter':<10}")
    print("-" * 100)

    print(f"{'시도1: 5m T=0.20':<35} {'876':<10} {'12.2%':<10} {'0.580':<10} {'0.580':<10} {'-':<10} {'36':<10}")
    print(f"{'시도2: 5m + OBI>0.3':<35} {'661':<10} {'11.5%':<10} {'0.565':<10} {'0.611':<10} {'0.546':<10} {'105':<10}")
    print(f"{'시도3: 1h T=0.20':<35} {'3290':<10} {'59.6%':<10} {'0.610':<10} {'0.620':<10} {'-':<10} {'0':<10}")
    print()

    for name, r in all_results.items():
        print(f"{name:<35} {r['test_size']:<10} {r['test_retention']:<10.1f}% {r['lr_auc']:<10.3f} {r['xgb_auc']:<10.3f} {r['lr_test_bal_at_val']:<10.3f} {r['best_iter']:<10}")

    print()
    print("=" * 80)
    print("INTERPRETATION + 다음 단계")
    print("=" * 80)
    print()
    print("AUC 향상 시나리오:")
    print("  > 0.62 -> 시도 3 baseline 유지")
    print("  > 0.65 -> 시도 4 큰 성공")
    print("  > 0.68 -> 시도 4 매우 큰 성공")
    print()
    print("Best variant 선정:")
    print("  1. AUC 가장 높음")
    print("  2. Test sample >= 200")
    print("  3. Val-thresh test bal_acc 높음")
    print("  4. xgb best_iter > 5")
    print("  5. Train-test gap < 0.07")

    log.info("\nAnalysis complete")


if __name__ == "__main__":
    main()
