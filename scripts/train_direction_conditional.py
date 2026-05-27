"""Direction Classifier with Conditional Filtering (Triple-barrier + Vol regime + OBI strength)."""
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
    log.info("CONDITIONAL DIRECTION CLASSIFIER")
    log.info("(Triple-barrier + Vol regime + OBI strength)")
    log.info("=" * 70)

    log.info(f"\nBuilding datasets...")
    train_df_full = build_split(DATES_TRAIN, log)
    val_df_full = build_split(DATES_VAL, log)
    test_df_full = build_split(DATES_TEST, log)

    return_target = "target_return_300s"
    vol_target = "target_volatility_300s"

    for df in [train_df_full, val_df_full, test_df_full]:
        df.dropna(subset=[return_target, vol_target], inplace=True)

    feature_cols = get_feature_columns(train_df_full)
    log.info(f"Features: {len(feature_cols)}")

    X_train_full_raw = train_df_full.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    train_feature_medians = X_train_full_raw.median(numeric_only=True)

    def make_xy(df, feat_cols, train_medians, target):
        X = df.reindex(columns=feat_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X_filled = X.fillna(train_medians).fillna(0)
        y = df[target].values
        return X, X_filled, y

    print()
    print("=" * 80)
    print("PHASE 1: Vol regime classifier (V2 reproduction)")
    print("=" * 80)

    train_median_vol = train_df_full[vol_target].median()
    train_df_full["vol_binary"] = (train_df_full[vol_target] > train_median_vol).astype(int)
    val_df_full["vol_binary"] = (val_df_full[vol_target] > train_median_vol).astype(int)
    test_df_full["vol_binary"] = (test_df_full[vol_target] > train_median_vol).astype(int)

    X_train_v, X_train_v_filled, _ = make_xy(train_df_full, feature_cols, train_feature_medians, "vol_binary")
    X_val_v, X_val_v_filled, _ = make_xy(val_df_full, feature_cols, train_feature_medians, "vol_binary")
    X_test_v, X_test_v_filled, _ = make_xy(test_df_full, feature_cols, train_feature_medians, "vol_binary")

    y_vol_train = train_df_full["vol_binary"].values
    y_vol_val = val_df_full["vol_binary"].values
    y_vol_test = test_df_full["vol_binary"].values

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    import xgboost as xgb

    scaler_vol = StandardScaler()
    X_train_v_scaled = scaler_vol.fit_transform(X_train_v_filled)
    X_val_v_scaled = scaler_vol.transform(X_val_v_filled)
    X_test_v_scaled = scaler_vol.transform(X_test_v_filled)

    lr_vol = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr_vol.fit(X_train_v_scaled, y_vol_train)

    train_df_full["vol_proba"] = lr_vol.predict_proba(X_train_v_scaled)[:, 1]
    val_df_full["vol_proba"] = lr_vol.predict_proba(X_val_v_scaled)[:, 1]
    test_df_full["vol_proba"] = lr_vol.predict_proba(X_test_v_scaled)[:, 1]

    log.info(f"  Vol classifier trained")
    log.info(f"  Test vol_proba > 0.6 fraction: {(test_df_full['vol_proba'] > 0.6).mean():.3f}")

    print()
    print("=" * 80)
    print("PHASE 2: Conditional Filter Variants")
    print("=" * 80)

    obi_col = "ob_obi_top1"
    if obi_col not in train_df_full.columns:
        log.warning(f"{obi_col} not in features, fallback to ob_obi_top5")
        obi_col = "ob_obi_top5"

    log.info(f"  Using OBI col: {obi_col}")

    VARIANTS = [
        ("baseline_T0.10",       0.10, 0.0, 0.0),
        ("T0.10 + vol>0.5",      0.10, 0.5, 0.0),
        ("T0.10 + vol>0.6",      0.10, 0.6, 0.0),
        ("T0.10 + |OBI|>0.3",    0.10, 0.0, 0.3),
        ("T0.10 + |OBI|>0.5",    0.10, 0.0, 0.5),
        ("T0.10 + vol>0.5 + |OBI|>0.3", 0.10, 0.5, 0.3),
        ("T0.10 + vol>0.6 + |OBI|>0.5", 0.10, 0.6, 0.5),
        ("T0.20 + vol>0.5",      0.20, 0.5, 0.0),
        ("T0.20 + |OBI|>0.3",    0.20, 0.0, 0.3),
        ("T0.20 + vol>0.5 + |OBI|>0.3", 0.20, 0.5, 0.3),
    ]

    all_results = {}

    for variant_name, T, vol_thresh, obi_thresh in VARIANTS:
        print()
        print("-" * 80)
        print(f"VARIANT: {variant_name}")
        print(f"  T={T}, vol_proba >= {vol_thresh}, |OBI| >= {obi_thresh}")
        print("-" * 80)

        def filter_df(df):
            mask = df[return_target].abs() > T
            if vol_thresh > 0:
                mask &= df["vol_proba"] >= vol_thresh
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
        print(f"    Train: {len(train_df):>5} / {len(train_df_full):>6} = {train_retention:.1f}%")
        print(f"    Val:   {len(val_df):>5} / {len(val_df_full):>6} = {val_retention:.1f}%")
        print(f"    Test:  {len(test_df):>5} / {len(test_df_full):>6} = {test_retention:.1f}%")

        if len(train_df) < 300 or len(val_df) < 50 or len(test_df) < 50:
            print(f"    WARN: Sample too small, skipping")
            continue

        train_df["target_binary"] = (train_df[return_target] > 0).astype(int)
        val_df["target_binary"] = (val_df[return_target] > 0).astype(int)
        test_df["target_binary"] = (test_df[return_target] > 0).astype(int)

        if train_df["target_binary"].nunique() < 2 or val_df["target_binary"].nunique() < 2 or test_df["target_binary"].nunique() < 2:
            print(f"    WARN: Single class, skipping")
            continue

        print(f"  Up fraction: train={train_df['target_binary'].mean():.3f}, val={val_df['target_binary'].mean():.3f}, test={test_df['target_binary'].mean():.3f}")

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

        lr_proba_val = lr.predict_proba(X_val_scaled)[:, 1]
        lr_proba_test = lr.predict_proba(X_test_scaled)[:, 1]
        xgb_proba_val = xgb_model.predict_proba(X_val_filled)[:, 1]
        xgb_proba_test = xgb_model.predict_proba(X_test_filled)[:, 1]

        lr_pred_test = (lr_proba_test > 0.5).astype(int)
        xgb_pred_test = (xgb_proba_test > 0.5).astype(int)

        def per_class(y_true, y_pred):
            c0 = y_true == 0; c1 = y_true == 1
            c0_acc = (y_pred[c0] == 0).mean() if c0.sum() > 0 else 0
            c1_acc = (y_pred[c1] == 1).mean() if c1.sum() > 0 else 0
            return c0_acc, c1_acc, (c0_acc + c1_acc) / 2

        lr_test_auc = roc_auc_score(y_test, lr_proba_test) if len(set(y_test)) > 1 else 0.5
        xgb_test_auc = roc_auc_score(y_test, xgb_proba_test) if len(set(y_test)) > 1 else 0.5

        c0_te, c1_te, lr_test_bal = per_class(y_test, lr_pred_test)
        _, _, xgb_test_bal = per_class(y_test, xgb_pred_test)

        best_thresh_val = 0.5
        best_bal_val = 0
        for t in np.arange(0.30, 0.70, 0.01):
            pred = (lr_proba_val > t).astype(int)
            _, _, bal = per_class(y_val, pred)
            if bal > best_bal_val:
                best_bal_val = bal
                best_thresh_val = t

        lr_pred_test_best = (lr_proba_test > best_thresh_val).astype(int)
        _, _, lr_test_bal_best = per_class(y_test, lr_pred_test_best)

        print(f"\n  XGBoost best_iter: {xgb_model.best_iteration}")
        if xgb_model.best_iteration < 5:
            print(f"  WARN: XGBoost early stop - learning issue")
        print(f"  LogReg test AUC: {lr_test_auc:.3f}, bal_acc: {lr_test_bal:.3f}")
        print(f"  XGBoost test AUC: {xgb_test_auc:.3f}, bal_acc: {xgb_test_bal:.3f}")
        print(f"  Val-selected threshold: {best_thresh_val:.2f}")
        print(f"  LogReg test bal_acc at val-threshold: {lr_test_bal_best:.3f}  (no leakage)")

        print(f"\n  Per-date Test AUC:")
        test_df_copy = test_df.copy().reset_index(drop=True)
        test_df_copy["lr_proba"] = lr_proba_test
        for date_str in DATES_TEST:
            sub = test_df_copy[test_df_copy["_source_date"] == date_str]
            if len(sub) > 5 and sub["target_binary"].nunique() > 1:
                try:
                    auc_d = roc_auc_score(sub["target_binary"], sub["lr_proba"])
                    print(f"    {date_str}: n={len(sub):<4}, up={sub['target_binary'].mean():.2f}, AUC={auc_d:.3f}")
                except:
                    pass

        all_results[variant_name] = {
            "T": T, "vol_thresh": vol_thresh, "obi_thresh": obi_thresh,
            "lr_test_auc": lr_test_auc,
            "lr_test_bal": lr_test_bal,
            "lr_test_bal_at_val_thresh": lr_test_bal_best,
            "xgb_test_auc": xgb_test_auc,
            "xgb_test_bal": xgb_test_bal,
            "xgb_best_iter": xgb_model.best_iteration,
            "best_val_threshold": best_thresh_val,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_retention": train_retention,
            "test_retention": test_retention,
        }

    print()
    print("=" * 100)
    print("FINAL COMPARISON: Conditional Variants")
    print("=" * 100)
    print()
    print(f"{'Variant':<35} {'Test n':<8} {'Retain':<8} {'LR AUC':<8} {'LR bal':<8} {'@val_thr':<10} {'Best thr':<10} {'XGB iter':<10}")
    print("-" * 100)

    print(f"{'시도1: T=0.10':<35} {'2551':<8} {'44.4%':<8} {'0.543':<8} {'0.529':<8} {'~0.538':<10} {'0.53 (test)':<10} {'6':<10}")
    print(f"{'시도1: T=0.20':<35} {'876':<8} {'15.3%':<8} {'0.580':<8} {'0.534':<8} {'~0.576':<10} {'0.62 (test)':<10} {'36':<10}")

    print()
    for name, r in all_results.items():
        print(f"{name:<35} {r['test_size']:<8} {r['test_retention']:<8.1f}% {r['lr_test_auc']:<8.3f} {r['lr_test_bal']:<8.3f} {r['lr_test_bal_at_val_thresh']:<10.3f} {r['best_val_threshold']:<10.2f} {r['xgb_best_iter']:<10}")

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("Conditional 효과 평가:")
    print("  - Vol regime + OBI strength 결합으로 AUC 추가 향상 여부")
    print("  - Sample 줄어드는 trade-off")
    print()
    print("Trading 가능 시나리오:")
    print("  AUC > 0.60 -> 강한 신호. Conditional 성공")
    print("  AUC > 0.62 -> 매우 강함")
    print("  AUC > 0.65 -> 시도 2 큰 성공")
    print()
    print("Best variant 선정 기준:")
    print("  1. AUC 가장 높음")
    print("  2. Test sample >= 100 (variance 제어)")
    print("  3. Val-selected threshold 의 test bal_acc 높음")
    print("  4. XGB best_iter > 5 (mode collapse 아님)")

    log.info("\nConditional analysis complete")


if __name__ == "__main__":
    main()
