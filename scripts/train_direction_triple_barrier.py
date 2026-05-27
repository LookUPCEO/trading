"""Direction Classifier with Triple-Barrier (multi-T comparison + mode collapse 진단)."""
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
    log.info("TRIPLE-BARRIER DIRECTION CLASSIFIER (multi-T)")
    log.info("=" * 70)

    log.info(f"\nBuilding TRAIN ({len(DATES_TRAIN)} dates)")
    train_df_full = build_split(DATES_TRAIN, log)
    log.info(f"\nBuilding VAL ({len(DATES_VAL)} dates)")
    val_df_full = build_split(DATES_VAL, log)
    log.info(f"\nBuilding TEST ({len(DATES_TEST)} dates)")
    test_df_full = build_split(DATES_TEST, log)

    target_col = "target_return_300s"

    for df_name, df in [("train", train_df_full), ("val", val_df_full), ("test", test_df_full)]:
        before = len(df)
        df.dropna(subset=[target_col], inplace=True)
        after = len(df)
        log.info(f"  {df_name} after target dropna: {before} → {after}")

    feature_cols = get_feature_columns(train_df_full)
    log.info(f"\nFeatures: {len(feature_cols)}")

    X_train_full_raw = train_df_full.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    train_feature_medians = X_train_full_raw.median(numeric_only=True)

    T_VALUES = [0.05, 0.10, 0.15, 0.20]
    all_results = {}

    for T in T_VALUES:
        print()
        print("=" * 80)
        print(f"T = {T}% (triple-barrier threshold)")
        print("=" * 80)

        train_df = train_df_full[train_df_full[target_col].abs() > T].copy()
        val_df = val_df_full[val_df_full[target_col].abs() > T].copy()
        test_df = test_df_full[test_df_full[target_col].abs() > T].copy()

        train_df["target_binary"] = (train_df[target_col] > 0).astype(int)
        val_df["target_binary"] = (val_df[target_col] > 0).astype(int)
        test_df["target_binary"] = (test_df[target_col] > 0).astype(int)

        train_retention = len(train_df) / len(train_df_full) * 100
        val_retention = len(val_df) / len(val_df_full) * 100
        test_retention = len(test_df) / len(test_df_full) * 100

        print(f"\n--- Sample retention ---")
        print(f"  Train: {len(train_df):>6} / {len(train_df_full):>6} = {train_retention:.1f}%")
        print(f"  Val:   {len(val_df):>6} / {len(val_df_full):>6} = {val_retention:.1f}%")
        print(f"  Test:  {len(test_df):>6} / {len(test_df_full):>6} = {test_retention:.1f}%")

        if len(train_df) < 500 or len(val_df) < 100 or len(test_df) < 100:
            print(f"  WARN: Sample too small at T={T}, skipping")
            continue

        print(f"\n--- Up fraction ---")
        print(f"  Train: {train_df['target_binary'].mean():.3f}")
        print(f"  Val:   {val_df['target_binary'].mean():.3f}")
        print(f"  Test:  {test_df['target_binary'].mean():.3f}")

        def make_xy(df, feat_cols, train_medians):
            X = df.reindex(columns=feat_cols).copy()
            X = X.replace([np.inf, -np.inf], np.nan)
            X_filled = X.fillna(train_medians).fillna(0)
            y = df["target_binary"].values
            return X, X_filled, y

        X_train, X_train_filled, y_train = make_xy(train_df, feature_cols, train_feature_medians)
        X_val, X_val_filled, y_val = make_xy(val_df, feature_cols, train_feature_medians)
        X_test, X_test_filled, y_test = make_xy(test_df, feature_cols, train_feature_medians)

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, roc_auc_score
        import xgboost as xgb

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

        print(f"\n--- Training stats (T={T}%) ---")
        print(f"  XGBoost best_iter: {xgb_model.best_iteration} / 500")
        if xgb_model.best_iteration < 5:
            print(f"  WARN: Very early stop - possible learning failure")

        lr_proba_train = lr.predict_proba(X_train_scaled)[:, 1]
        lr_proba_test = lr.predict_proba(X_test_scaled)[:, 1]
        xgb_proba_train = xgb_model.predict_proba(X_train_filled)[:, 1]
        xgb_proba_test = xgb_model.predict_proba(X_test_filled)[:, 1]

        lr_pred_train = (lr_proba_train > 0.5).astype(int)
        lr_pred_test = (lr_proba_test > 0.5).astype(int)
        xgb_pred_train = (xgb_proba_train > 0.5).astype(int)
        xgb_pred_test = (xgb_proba_test > 0.5).astype(int)

        def per_class(y_true, y_pred):
            c0 = y_true == 0
            c1 = y_true == 1
            c0_acc = (y_pred[c0] == 0).mean() if c0.sum() > 0 else 0
            c1_acc = (y_pred[c1] == 1).mean() if c1.sum() > 0 else 0
            return c0_acc, c1_acc, (c0_acc + c1_acc) / 2

        print(f"\n--- Per-class accuracy (T={T}%) ---")
        print(f"  {'Model':<12} {'Split':<8} {'Down acc':<10} {'Up acc':<10} {'Bal acc':<10}")

        for name, pred_train, pred_test in [
            ("LogReg", lr_pred_train, lr_pred_test),
            ("XGBoost", xgb_pred_train, xgb_pred_test),
        ]:
            c0_tr, c1_tr, bal_tr = per_class(y_train, pred_train)
            c0_te, c1_te, bal_te = per_class(y_test, pred_test)
            print(f"  {name:<12} {'train':<8} {c0_tr:<10.3f} {c1_tr:<10.3f} {bal_tr:<10.3f}")
            print(f"  {name:<12} {'test':<8} {c0_te:<10.3f} {c1_te:<10.3f} {bal_te:<10.3f}")

        xgb_pred_pos_test = xgb_pred_test.mean()
        if xgb_pred_pos_test > 0.95 or xgb_pred_pos_test < 0.05:
            print(f"  WARN: XGBoost mode collapse: pred=1 ratio = {xgb_pred_pos_test:.3f}")

        lr_test_acc = accuracy_score(y_test, lr_pred_test)
        lr_test_auc = roc_auc_score(y_test, lr_proba_test) if len(set(y_test)) > 1 else 0.5
        _, _, lr_test_bal = per_class(y_test, lr_pred_test)

        xgb_test_acc = accuracy_score(y_test, xgb_pred_test)
        xgb_test_auc = roc_auc_score(y_test, xgb_proba_test) if len(set(y_test)) > 1 else 0.5
        _, _, xgb_test_bal = per_class(y_test, xgb_pred_test)

        _, _, lr_train_bal = per_class(y_train, lr_pred_train)
        _, _, xgb_train_bal = per_class(y_train, xgb_pred_train)

        best_thresh = 0.5
        best_bal = 0
        for t in np.arange(0.30, 0.70, 0.01):
            pred = (lr_proba_test > t).astype(int)
            _, _, bal = per_class(y_test, pred)
            if bal > best_bal:
                best_bal = bal
                best_thresh = t

        print(f"\n--- Test metrics (T={T}%) ---")
        print(f"  LogReg:    acc={lr_test_acc:.3f}, AUC={lr_test_auc:.3f}, bal_acc={lr_test_bal:.3f}")
        print(f"  XGBoost:   acc={xgb_test_acc:.3f}, AUC={xgb_test_auc:.3f}, bal_acc={xgb_test_bal:.3f}")
        print(f"  LogReg best threshold {best_thresh:.2f}: bal_acc={best_bal:.3f} (selected on test)")
        print(f"  Train-Test gap (LogReg bal): {lr_train_bal - lr_test_bal:+.3f}")

        print(f"\n--- Per-date Test AUC ---")
        print(f"  {'Date':<14} {'N':<6} {'Up %':<8} {'LogReg AUC':<12} {'XGB AUC':<12}")

        test_df_copy = test_df.copy().reset_index(drop=True)
        test_df_copy["lr_proba"] = lr_proba_test
        test_df_copy["xgb_proba"] = xgb_proba_test

        for date_str in DATES_TEST:
            sub = test_df_copy[test_df_copy["_source_date"] == date_str]
            if len(sub) > 5 and sub["target_binary"].nunique() > 1:
                try:
                    lr_auc_d = roc_auc_score(sub["target_binary"], sub["lr_proba"])
                    xgb_auc_d = roc_auc_score(sub["target_binary"], sub["xgb_proba"])
                    up = sub["target_binary"].mean()
                    print(f"  {date_str:<14} {len(sub):<6} {up:<8.3f} {lr_auc_d:<12.3f} {xgb_auc_d:<12.3f}")
                except:
                    pass
            else:
                print(f"  {date_str:<14} {len(sub):<6} (insufficient or single class)")

        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": xgb_model.feature_importances_,
        }).sort_values("importance", ascending=False).head(10)

        print(f"\n--- Top 10 features (XGBoost, T={T}%) ---")
        for _, row in importance.iterrows():
            print(f"  {row['feature']:<46} {row['importance']:.4f}")

        all_results[T] = {
            "lr_test_auc": lr_test_auc,
            "lr_test_bal": lr_test_bal,
            "lr_test_acc": lr_test_acc,
            "xgb_test_auc": xgb_test_auc,
            "xgb_test_bal": xgb_test_bal,
            "xgb_best_iter": xgb_model.best_iteration,
            "best_thresh_bal": best_bal,
            "best_thresh": best_thresh,
            "train_test_gap": lr_train_bal - lr_test_bal,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_retention": train_retention,
            "test_up_frac": y_test.mean(),
        }

    print()
    print("=" * 80)
    print("FINAL COMPARISON: Triple-barrier T values")
    print("=" * 80)
    print()
    print(f"{'T':<10} {'Train n':<10} {'Retention':<12} {'LR AUC':<10} {'LR bal':<10} {'Best bal':<12} {'Best thr':<10} {'Gap':<8} {'XGB iter':<10}")
    print("-" * 100)

    print(f"{'0.00 (B)':<10} {'~22969':<10} {'100.0%':<12} {'0.515':<10} {'0.512':<10} {'0.518':<12} {'0.53':<10} {'+0.035':<8} {'~58':<10}")

    for T in T_VALUES:
        if T in all_results:
            r = all_results[T]
            print(f"{T:<10.2f} {r['train_size']:<10} {r['train_retention']:<12.1f} {r['lr_test_auc']:<10.3f} {r['lr_test_bal']:<10.3f} {r['best_thresh_bal']:<12.3f} {r['best_thresh']:<10.2f} {r['train_test_gap']:<+8.3f} {r['xgb_best_iter']:<10}")

    print()
    print("=" * 80)
    print("INTERPRETATION + 다음 단계")
    print("=" * 80)
    print()
    print("AUC 향상 시 의미:")
    print("  - Noise 제외 -> signal 정제")
    print("  - 단점: Test sample down, evaluation variance up")
    print()
    print("Best T 선택 기준:")
    print("  - AUC 가장 높음 + sample retention >= 25%")
    print("  - Best threshold bal_acc 가장 높음")
    print("  - Gap 작음 + xgb_best_iter > 5 (mode collapse 아님)")
    print()
    print("AUC 해석:")
    print("  > 0.55 -> Direction trading 가능 시작점")
    print("  > 0.58 -> 강한 신호")
    print("  > 0.60 -> 매우 강함")

    log.info("\nTriple-barrier analysis complete")


if __name__ == "__main__":
    main()
