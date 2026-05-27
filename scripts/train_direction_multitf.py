"""Direction Classifier with Multi-Timeframe Consensus."""
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
    log.info("MULTI-TIMEFRAME DIRECTION CLASSIFIER")
    log.info("=" * 70)

    log.info(f"\nBuilding datasets...")
    train_df_full = build_split(DATES_TRAIN, log)
    val_df_full = build_split(DATES_VAL, log)
    test_df_full = build_split(DATES_TEST, log)

    available_targets = [c for c in train_df_full.columns if c.startswith("target_return_")]
    log.info(f"\nAvailable return targets: {available_targets}")

    candidate_tfs = [60, 180, 300, 600, 900, 1800, 3600]
    available_tfs = []
    for tf in candidate_tfs:
        if f"target_return_{tf}s" in available_targets:
            available_tfs.append(tf)

    log.info(f"Will use timeframes: {available_tfs}")

    if len(available_tfs) < 2:
        log.error("Need at least 2 timeframes.")
        return

    return_cols = [f"target_return_{tf}s" for tf in available_tfs]

    for df in [train_df_full, val_df_full, test_df_full]:
        before = len(df)
        df.dropna(subset=return_cols, inplace=True)
        log.info(f"  Drop NaN: {before} → {len(df)}")

    feature_cols = get_feature_columns(train_df_full)
    log.info(f"\nFeatures: {len(feature_cols)}")

    X_train_raw = train_df_full.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    train_feature_medians = X_train_raw.median(numeric_only=True)

    def make_xy(df, feat_cols, train_medians, target):
        X = df.reindex(columns=feat_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X_filled = X.fillna(train_medians).fillna(0)
        y = df[target].values
        return X, X_filled, y

    print()
    print("=" * 80)
    print("PHASE 1: Single-Timeframe baselines (T=0.20)")
    print("=" * 80)

    T = 0.20

    single_tf_results = {}

    for tf in available_tfs:
        col = f"target_return_{tf}s"
        train_df = train_df_full[train_df_full[col].abs() > T].copy()
        val_df = val_df_full[val_df_full[col].abs() > T].copy()
        test_df = test_df_full[test_df_full[col].abs() > T].copy()

        if len(train_df) < 500 or len(test_df) < 100:
            print(f"\n  TF {tf}s: insufficient samples (train {len(train_df)}, test {len(test_df)})")
            continue

        train_df["target_binary"] = (train_df[col] > 0).astype(int)
        val_df["target_binary"] = (val_df[col] > 0).astype(int)
        test_df["target_binary"] = (test_df[col] > 0).astype(int)

        if test_df["target_binary"].nunique() < 2:
            continue

        X_train, X_train_filled, y_train = make_xy(train_df, feature_cols, train_feature_medians, "target_binary")
        X_val, X_val_filled, y_val = make_xy(val_df, feature_cols, train_feature_medians, "target_binary")
        X_test, X_test_filled, y_test = make_xy(test_df, feature_cols, train_feature_medians, "target_binary")

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
        import xgboost as xgb

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
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

        lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
        xgb_proba = xgb_model.predict_proba(X_test_filled)[:, 1]

        try:
            lr_auc = roc_auc_score(y_test, lr_proba)
            xgb_auc = roc_auc_score(y_test, xgb_proba)
        except:
            lr_auc = xgb_auc = 0.5

        retention = len(test_df) / len(test_df_full) * 100
        print(f"\n  TF {tf}s: test n={len(test_df)}, retention {retention:.1f}%, LR AUC {lr_auc:.3f}, XGB AUC {xgb_auc:.3f}, best_iter {xgb_model.best_iteration}")

        single_tf_results[tf] = {
            "lr_auc": lr_auc, "xgb_auc": xgb_auc,
            "best_iter": xgb_model.best_iteration,
            "test_n": len(test_df), "retention": retention,
        }

    print()
    print("=" * 80)
    print("PHASE 2: Multi-Timeframe Consensus")
    print("=" * 80)

    T_filter = 0.20

    main_tf = 300
    if main_tf not in available_tfs:
        main_tf = available_tfs[len(available_tfs) // 2]

    main_col = f"target_return_{main_tf}s"
    log.info(f"\nUsing main TF: {main_tf}s")

    def add_consensus(df, return_cols, main_col, T):
        signs = df[return_cols].apply(np.sign)
        consensus = signs.abs().sum(axis=1) == len(return_cols)
        movement = df[main_col].abs() > T
        df["multi_tf_strong"] = consensus & movement
        df["target_binary"] = (df[main_col] > 0).astype(int)
        return df

    train_df = add_consensus(train_df_full.copy(), return_cols, main_col, T_filter)
    val_df = add_consensus(val_df_full.copy(), return_cols, main_col, T_filter)
    test_df = add_consensus(test_df_full.copy(), return_cols, main_col, T_filter)

    train_df = train_df[train_df["multi_tf_strong"]].copy()
    val_df = val_df[val_df["multi_tf_strong"]].copy()
    test_df = test_df[test_df["multi_tf_strong"]].copy()

    train_retention = len(train_df) / len(train_df_full) * 100
    val_retention = len(val_df) / len(val_df_full) * 100
    test_retention = len(test_df) / len(test_df_full) * 100

    print(f"\nConsensus filter retention:")
    print(f"  Train: {len(train_df)} ({train_retention:.1f}%)")
    print(f"  Val:   {len(val_df)} ({val_retention:.1f}%)")
    print(f"  Test:  {len(test_df)} ({test_retention:.1f}%)")

    if len(train_df) < 300 or len(test_df) < 50:
        print("Sample too small")
        return

    print(f"\n  Up fraction: train {train_df['target_binary'].mean():.3f}, test {test_df['target_binary'].mean():.3f}")

    if test_df["target_binary"].nunique() < 2:
        print("Single class in test")
        return

    X_train, X_train_filled, y_train = make_xy(train_df, feature_cols, train_feature_medians, "target_binary")
    X_val, X_val_filled, y_val = make_xy(val_df, feature_cols, train_feature_medians, "target_binary")
    X_test, X_test_filled, y_test = make_xy(test_df, feature_cols, train_feature_medians, "target_binary")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
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

    lr_proba_val = lr.predict_proba(X_val_scaled)[:, 1]
    lr_proba_test = lr.predict_proba(X_test_scaled)[:, 1]
    xgb_proba_val = xgb_model.predict_proba(X_val_filled)[:, 1]
    xgb_proba_test = xgb_model.predict_proba(X_test_filled)[:, 1]

    lr_auc = roc_auc_score(y_test, lr_proba_test)
    xgb_auc = roc_auc_score(y_test, xgb_proba_test)

    def per_class(y_true, y_pred):
        c0 = y_true == 0; c1 = y_true == 1
        c0_acc = (y_pred[c0] == 0).mean() if c0.sum() > 0 else 0
        c1_acc = (y_pred[c1] == 1).mean() if c1.sum() > 0 else 0
        return c0_acc, c1_acc, (c0_acc + c1_acc) / 2

    lr_pred_test = (lr_proba_test > 0.5).astype(int)
    xgb_pred_test = (xgb_proba_test > 0.5).astype(int)

    c0_lr, c1_lr, lr_bal = per_class(y_test, lr_pred_test)
    c0_xgb, c1_xgb, xgb_bal = per_class(y_test, xgb_pred_test)

    best_thresh = 0.5
    best_bal_val = 0
    for t in np.arange(0.30, 0.70, 0.01):
        pred = (lr_proba_val > t).astype(int)
        _, _, bal = per_class(y_val, pred)
        if bal > best_bal_val:
            best_bal_val = bal
            best_thresh = t

    lr_pred_test_best = (lr_proba_test > best_thresh).astype(int)
    _, _, lr_bal_test_at_val = per_class(y_test, lr_pred_test_best)

    print(f"\n--- Multi-TF Consensus Results ---")
    print(f"  XGBoost best_iter: {xgb_model.best_iteration}")
    if xgb_model.best_iteration < 5:
        print(f"  WARN: Mode collapse risk")
    print(f"  LR  AUC: {lr_auc:.3f}, bal_acc: {lr_bal:.3f} (Down {c0_lr:.3f}, Up {c1_lr:.3f})")
    print(f"  XGB AUC: {xgb_auc:.3f}, bal_acc: {xgb_bal:.3f} (Down {c0_xgb:.3f}, Up {c1_xgb:.3f})")
    print(f"  Val-selected threshold: {best_thresh:.2f}")
    print(f"  LR test bal_acc at val-thresh: {lr_bal_test_at_val:.3f}")

    print(f"\n--- Per-date Test AUC ---")
    test_df_copy = test_df.copy().reset_index(drop=True)
    test_df_copy["lr_proba"] = lr_proba_test
    test_df_copy["xgb_proba"] = xgb_proba_test

    for date_str in DATES_TEST:
        sub = test_df_copy[test_df_copy["_source_date"] == date_str]
        if len(sub) > 5 and sub["target_binary"].nunique() > 1:
            try:
                lr_auc_d = roc_auc_score(sub["target_binary"], sub["lr_proba"])
                xgb_auc_d = roc_auc_score(sub["target_binary"], sub["xgb_proba"])
                print(f"  {date_str}: n={len(sub):<4}, up={sub['target_binary'].mean():.2f}, LR {lr_auc_d:.3f}, XGB {xgb_auc_d:.3f}")
            except:
                pass

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False).head(15)

    print(f"\n--- Top 15 features (Multi-TF Consensus) ---")
    for _, row in importance.iterrows():
        print(f"  {row['feature']:<46} {row['importance']:.4f}")

    print()
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Approach':<40} {'Test n':<10} {'LR AUC':<10} {'XGB AUC':<10}")
    print("-" * 80)
    print(f"{'시도1: T=0.20 (single 300s TF)':<40} {'876':<10} {'0.580':<10} {'0.580':<10}")
    print(f"{'시도2: T=0.20 + |OBI|>0.3':<40} {'661':<10} {'0.565':<10} {'0.611':<10}")

    for tf, r in single_tf_results.items():
        label = f"Single TF {tf}s + T=0.20"
        print(f"{label:<40} {r['test_n']:<10} {r['lr_auc']:<10.3f} {r['xgb_auc']:<10.3f}")

    label = f"Multi-TF consensus (T={T_filter})"
    print(f"{label:<40} {len(test_df):<10} {lr_auc:<10.3f} {xgb_auc:<10.3f}")

    print()
    print("=" * 80)
    print("INTERPRETATION + 다음 단계")
    print("=" * 80)
    print()
    print("AUC 향상 시나리오:")
    print("  > 0.62 -> Multi-TF 효과 검증")
    print("  > 0.65 -> 매우 강함")
    print("  > 0.68 -> 시도 3 큰 성공")
    print()
    print("Per-date 안정성:")
    print("  4 dates 평균 + 최저 AUC 모두 중요")
    print("  최저 AUC > 0.55 = 일관 성공")

    log.info("\nMulti-TF analysis complete")


if __name__ == "__main__":
    main()
