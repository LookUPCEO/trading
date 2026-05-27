"""V2 모델 검증: Train/Val/Test per-class, calibration, threshold."""
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
    log.info("V2 Validation Analysis")
    log.info("=" * 70)

    log.info(f"\nBuilding TRAIN ({len(DATES_TRAIN)} dates)")
    train_df = build_split(DATES_TRAIN, log)
    log.info(f"\nBuilding VAL ({len(DATES_VAL)} dates)")
    val_df = build_split(DATES_VAL, log)
    log.info(f"\nBuilding TEST ({len(DATES_TEST)} dates)")
    test_df = build_split(DATES_TEST, log)

    target_col = "target_volatility_300s"
    train_median = train_df[target_col].median()

    train_df["target_binary"] = (train_df[target_col] > train_median).astype(int)
    val_df["target_binary"] = (val_df[target_col] > train_median).astype(int)
    test_df["target_binary"] = (test_df[target_col] > train_median).astype(int)

    feature_cols = get_feature_columns(train_df)

    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.dropna(subset=[target_col, "target_binary"], inplace=True)

    X_train_raw = train_df.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    train_feature_medians = X_train_raw.median(numeric_only=True)

    def make_xy(df, feat_cols, train_medians):
        X = df.reindex(columns=feat_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X_filled = X.fillna(train_medians).fillna(0)
        y = df["target_binary"].values
        return X, X_filled, y

    X_train, X_train_filled, y_train = make_xy(train_df, feature_cols, train_feature_medians)
    X_val, X_val_filled, y_val = make_xy(val_df, feature_cols, train_feature_medians)
    X_test, X_test_filled, y_test = make_xy(test_df, feature_cols, train_feature_medians)

    log.info(f"\nClass balance:")
    log.info(f"  Train: {y_train.mean():.3f}")
    log.info(f"  Val:   {y_val.mean():.3f}")
    log.info(f"  Test:  {y_test.mean():.3f}")

    log.info("\nTraining V2 models...")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled)
    X_val_scaled = scaler.transform(X_val_filled)
    X_test_scaled = scaler.transform(X_test_filled)

    lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lr.fit(X_train_scaled, y_train)

    xgb_default = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0, gamma=0.1,
        random_state=42, eval_metric="logloss", early_stopping_rounds=30,
    )
    xgb_default.fit(X_train_filled, y_train, eval_set=[(X_val_filled, y_val)], verbose=False)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    log.info(f"  scale_pos_weight (train-based): {scale_pos_weight:.3f}")

    xgb_balanced = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0, gamma=0.1,
        random_state=42, eval_metric="logloss", early_stopping_rounds=30,
        scale_pos_weight=scale_pos_weight,
    )
    xgb_balanced.fit(X_train_filled, y_train, eval_set=[(X_val_filled, y_val)], verbose=False)

    log.info(f"  Default XGB best_iter: {xgb_default.best_iteration}")
    log.info(f"  Balanced XGB best_iter: {xgb_balanced.best_iteration}")

    lr_proba = {
        "train": lr.predict_proba(X_train_scaled)[:, 1],
        "val": lr.predict_proba(X_val_scaled)[:, 1],
        "test": lr.predict_proba(X_test_scaled)[:, 1],
    }
    xgb_proba = {
        "train": xgb_default.predict_proba(X_train_filled)[:, 1],
        "val": xgb_default.predict_proba(X_val_filled)[:, 1],
        "test": xgb_default.predict_proba(X_test_filled)[:, 1],
    }
    xgb_balanced_proba = {
        "train": xgb_balanced.predict_proba(X_train_filled)[:, 1],
        "val": xgb_balanced.predict_proba(X_val_filled)[:, 1],
        "test": xgb_balanced.predict_proba(X_test_filled)[:, 1],
    }

    y_dict = {"train": y_train, "val": y_val, "test": y_test}

    lr_pred = {k: (v > 0.5).astype(int) for k, v in lr_proba.items()}
    xgb_pred = {k: (v > 0.5).astype(int) for k, v in xgb_proba.items()}
    xgb_balanced_pred = {k: (v > 0.5).astype(int) for k, v in xgb_balanced_proba.items()}

    print()
    print("=" * 80)
    print("A. PER-CLASS ACCURACY (Train + Val + Test) — covariate shift 진단")
    print("=" * 80)

    def per_class(y_true, y_pred):
        c0_mask = y_true == 0
        c1_mask = y_true == 1
        c0_acc = (y_pred[c0_mask] == 0).mean() if c0_mask.sum() > 0 else 0
        c1_acc = (y_pred[c1_mask] == 1).mean() if c1_mask.sum() > 0 else 0
        bal = (c0_acc + c1_acc) / 2
        return c0_acc, c1_acc, bal, c0_mask.sum(), c1_mask.sum()

    print(f"\n{'Model':<20} {'Split':<8} {'C0 acc':<10} {'C1 acc':<10} {'Bal acc':<10} {'C0 n':<8} {'C1 n':<8}")
    print("-" * 80)

    for model_name, pred_dict in [("LogReg", lr_pred), ("XGB default", xgb_pred), ("XGB balanced", xgb_balanced_pred)]:
        for split in ["train", "val", "test"]:
            c0, c1, bal, n0, n1 = per_class(y_dict[split], pred_dict[split])
            print(f"{model_name:<20} {split:<8} {c0:<10.3f} {c1:<10.3f} {bal:<10.3f} {n0:<8} {n1:<8}")
        print()

    print()
    print("=" * 80)
    print("B. MODE COLLAPSE 진단 (Test pred=1 ratio)")
    print("=" * 80)
    print(f"\nTest set high_vol baseline: {y_test.mean():.3f}")
    print(f"\nIf pred=1 ratio == 1.0 (or close): full mode collapse")
    print(f"If pred=1 ratio == high_vol baseline: smart majority predictor")
    print(f"If pred=1 ratio close to 0.5: balanced predictor\n")
    print(f"  LogReg:        pred=1 ratio = {lr_pred['test'].mean():.3f}")
    print(f"  XGB default:   pred=1 ratio = {xgb_pred['test'].mean():.3f}")
    print(f"  XGB balanced:  pred=1 ratio = {xgb_balanced_pred['test'].mean():.3f}")

    print()
    print("=" * 80)
    print("C. THRESHOLD 분석 (LogReg on test)")
    print("=" * 80)
    print(f"\n{'Threshold':<10} {'Pred=1 %':<10} {'Acc':<8} {'C0 acc':<8} {'C1 acc':<8} {'Bal acc':<10}")
    print("-" * 60)

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred = (lr_proba["test"] > threshold).astype(int)
        pred1_pct = pred.mean()
        c0, c1, bal, _, _ = per_class(y_test, pred)
        acc = (pred == y_test).mean()
        print(f"{threshold:<10} {pred1_pct:<10.3f} {acc:<8.3f} {c0:<8.3f} {c1:<8.3f} {bal:<10.3f}")

    print(f"\n--- Best threshold for TEST balanced acc (LogReg) ---")
    best_thresh = 0.5
    best_bal = 0
    for t in np.arange(0.05, 0.95, 0.02):
        pred = (lr_proba["test"] > t).astype(int)
        _, _, bal, _, _ = per_class(y_test, pred)
        if bal > best_bal:
            best_bal = bal
            best_thresh = t
    print(f"  Best threshold: {best_thresh:.2f}, Balanced acc: {best_bal:.3f}")

    print()
    print("=" * 80)
    print("D. CALIBRATION (LogReg on test)")
    print("=" * 80)
    print(f"\n{'Bin':<15} {'Pred avg':<12} {'Actual':<12} {'Diff':<10} {'N':<8}")
    print("-" * 60)

    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = (lr_proba["test"] >= low) & (lr_proba["test"] < high)
        n = mask.sum()
        if n > 5:
            pred_avg = lr_proba["test"][mask].mean()
            actual = y_test[mask].mean()
            diff = pred_avg - actual
            warn = " WARN" if abs(diff) > 0.15 else ""
            print(f"{low:.1f}-{high:.1f}        {pred_avg:<12.3f} {actual:<12.3f} {diff:<+10.3f} {n:<8}{warn}")

    print()
    print("=" * 80)
    print("E. PER-DATE AUC")
    print("=" * 80)

    from sklearn.metrics import roc_auc_score

    test_df_copy = test_df.copy()
    test_df_copy["lr_proba"] = lr_proba["test"]
    test_df_copy["xgb_proba"] = xgb_proba["test"]

    print(f"\n{'Date':<14} {'N':<6} {'High_vol':<10} {'LogReg AUC':<12} {'XGB AUC':<12}")
    print("-" * 60)

    for date_str in DATES_TEST:
        sub = test_df_copy[test_df_copy["_source_date"] == date_str]
        if len(sub) > 0:
            if sub["target_binary"].nunique() > 1:
                try:
                    lr_auc = roc_auc_score(sub["target_binary"], sub["lr_proba"])
                    xgb_auc = roc_auc_score(sub["target_binary"], sub["xgb_proba"])
                    hv = sub["target_binary"].mean()
                    print(f"{date_str:<14} {len(sub):<6} {hv:<10.3f} {lr_auc:<12.3f} {xgb_auc:<12.3f}")
                except Exception as e:
                    print(f"{date_str:<14} ERROR: {e}")
            else:
                hv = sub["target_binary"].mean()
                print(f"{date_str:<14} {len(sub):<6} {hv:<10.3f} {'N/A (single class)':<12}")

    print()
    print("=" * 80)
    print("F. SUMMARY + DECISION GUIDE")
    print("=" * 80)

    print(f"\n--- Overview ---")
    print(f"Train high_vol: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")

    _, _, train_bal_lr, _, _ = per_class(y_train, lr_pred["train"])
    _, _, test_bal_lr, _, _ = per_class(y_test, lr_pred["test"])
    _, _, train_bal_xgb, _, _ = per_class(y_train, xgb_pred["train"])
    _, _, test_bal_xgb, _, _ = per_class(y_test, xgb_pred["test"])

    print()
    print(f"--- Balanced accuracy summary (random=0.5) ---")
    print(f"               Train      Test       Gap")
    print(f"  LogReg       {train_bal_lr:.3f}      {test_bal_lr:.3f}      {train_bal_lr-test_bal_lr:+.3f}")
    print(f"  XGB default  {train_bal_xgb:.3f}      {test_bal_xgb:.3f}      {train_bal_xgb-test_bal_xgb:+.3f}")

    print()
    print(f"--- Best threshold result (LogReg on test) ---")
    print(f"  Best balanced acc: {best_bal:.3f}  (at threshold {best_thresh:.2f})")

    print()
    print(f"--- DECISION GUIDE ---")
    print(f"")
    print(f"Case 1: Train bal_acc >> Test bal_acc")
    print(f"  -> Covariate shift; V2.5 (rolling threshold) needed")
    print(f"")
    print(f"Case 2: Both train and test bal_acc < 0.55")
    print(f"  -> Model can't learn vol signal; pivot to direction target")
    print(f"")
    print(f"Case 3: Both train and test bal_acc > 0.60")
    print(f"  -> Real signal; B safe to proceed")
    print(f"")
    print(f"Case 4: Test best_threshold bal_acc > 0.60")
    print(f"  -> Threshold tuning fixes it; B proceed")

    log.info("\nValidation analysis complete")


if __name__ == "__main__":
    main()
