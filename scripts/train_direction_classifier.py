"""ML Direction Classifier - 5min 후 가격 up/down."""
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
    log.info("DIRECTION CLASSIFIER (5min up/down)")
    log.info("=" * 70)

    log.info(f"\nBuilding TRAIN ({len(DATES_TRAIN)} dates)")
    train_df = build_split(DATES_TRAIN, log)
    log.info(f"\nBuilding VAL ({len(DATES_VAL)} dates)")
    val_df = build_split(DATES_VAL, log)
    log.info(f"\nBuilding TEST ({len(DATES_TEST)} dates)")
    test_df = build_split(DATES_TEST, log)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        log.error("Missing data")
        return

    target_col = "target_return_300s"

    log.info(f"\n--- Target distribution ---")
    log.info(f"  Train return mean: {train_df[target_col].mean():.4f}%")
    log.info(f"  Train return std:  {train_df[target_col].std():.4f}%")
    log.info(f"  Val return mean:   {val_df[target_col].mean():.4f}%")
    log.info(f"  Test return mean:  {test_df[target_col].mean():.4f}%")

    train_df["target_binary"] = (train_df[target_col] > 0).astype(int)
    val_df["target_binary"] = (val_df[target_col] > 0).astype(int)
    test_df["target_binary"] = (test_df[target_col] > 0).astype(int)

    log.info(f"\n--- Up fraction (binary balance) ---")
    log.info(f"  Train: {train_df['target_binary'].mean():.3f}")
    log.info(f"  Val:   {val_df['target_binary'].mean():.3f}")
    log.info(f"  Test:  {test_df['target_binary'].mean():.3f}")

    feature_cols = get_feature_columns(train_df)
    log.info(f"\nFeatures: {len(feature_cols)}")

    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        before = len(df)
        df.dropna(subset=[target_col, "target_binary"], inplace=True)
        after = len(df)
        log.info(f"  {df_name}: {before} → {after}")

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

    log.info(f"\nFinal shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    log.info("\nTraining models...")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
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
    log.info(f"  XGBoost best_iter: {xgb_model.best_iteration}")

    lr_proba = {
        "train": lr.predict_proba(X_train_scaled)[:, 1],
        "val": lr.predict_proba(X_val_scaled)[:, 1],
        "test": lr.predict_proba(X_test_scaled)[:, 1],
    }
    xgb_proba = {
        "train": xgb_model.predict_proba(X_train_filled)[:, 1],
        "val": xgb_model.predict_proba(X_val_filled)[:, 1],
        "test": xgb_model.predict_proba(X_test_filled)[:, 1],
    }
    y_dict = {"train": y_train, "val": y_val, "test": y_test}

    lr_pred = {k: (v > 0.5).astype(int) for k, v in lr_proba.items()}
    xgb_pred = {k: (v > 0.5).astype(int) for k, v in xgb_proba.items()}

    print()
    print("=" * 80)
    print("DIRECTION CLASSIFIER EVALUATION")
    print("=" * 80)

    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

    def evaluate(name, y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            auc = 0.5
        print(f"  {name:<25}: acc={acc:.3f} auc={auc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")
        return {"name": name, "acc": acc, "auc": auc}

    print(f"\n--- 1. Random ---")
    np.random.seed(42)
    rand_pred = np.random.randint(0, 2, size=len(y_test))
    rand_proba = np.random.random(size=len(y_test))
    evaluate("Random (test)", y_test, rand_pred, rand_proba)

    print(f"\n--- 2. Logistic Regression ---")
    evaluate("LogReg (train)", y_train, lr_pred["train"], lr_proba["train"])
    evaluate("LogReg (val)", y_val, lr_pred["val"], lr_proba["val"])
    evaluate("LogReg (test)", y_test, lr_pred["test"], lr_proba["test"])

    print(f"\n--- 3. XGBoost ---")
    evaluate("XGBoost (train)", y_train, xgb_pred["train"], xgb_proba["train"])
    evaluate("XGBoost (val)", y_val, xgb_pred["val"], xgb_proba["val"])
    evaluate("XGBoost (test)", y_test, xgb_pred["test"], xgb_proba["test"])

    print()
    print("=" * 80)
    print("PER-CLASS ACCURACY (covariate shift 진단)")
    print("=" * 80)

    def per_class(y_true, y_pred):
        c0_mask = y_true == 0
        c1_mask = y_true == 1
        c0_acc = (y_pred[c0_mask] == 0).mean() if c0_mask.sum() > 0 else 0
        c1_acc = (y_pred[c1_mask] == 1).mean() if c1_mask.sum() > 0 else 0
        bal = (c0_acc + c1_acc) / 2
        return c0_acc, c1_acc, bal, c0_mask.sum(), c1_mask.sum()

    print(f"\n{'Model':<20} {'Split':<8} {'Down acc':<10} {'Up acc':<10} {'Bal acc':<10} {'Down/Up n':<15}")
    print("-" * 80)

    for model_name, pred_dict in [("LogReg", lr_pred), ("XGBoost", xgb_pred)]:
        for split in ["train", "val", "test"]:
            c0, c1, bal, n0, n1 = per_class(y_dict[split], pred_dict[split])
            print(f"{model_name:<20} {split:<8} {c0:<10.3f} {c1:<10.3f} {bal:<10.3f} {n0}/{n1}")
        print()

    print()
    print("=" * 80)
    print("THRESHOLD 분석 (LogReg on test)")
    print("=" * 80)
    print(f"\n{'Threshold':<10} {'Pred=1 %':<10} {'Acc':<8} {'Down acc':<10} {'Up acc':<10} {'Bal':<8}")
    print("-" * 60)

    for threshold in [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60]:
        pred = (lr_proba["test"] > threshold).astype(int)
        c0, c1, bal, _, _ = per_class(y_test, pred)
        acc = (pred == y_test).mean()
        print(f"{threshold:<10} {pred.mean():<10.3f} {acc:<8.3f} {c0:<10.3f} {c1:<10.3f} {bal:<8.3f}")

    best_thresh = 0.5
    best_bal = 0
    for t in np.arange(0.30, 0.70, 0.01):
        pred = (lr_proba["test"] > t).astype(int)
        _, _, bal, _, _ = per_class(y_test, pred)
        if bal > best_bal:
            best_bal = bal
            best_thresh = t
    print(f"\nBest threshold: {best_thresh:.2f}, Bal acc: {best_bal:.3f}")

    print()
    print("=" * 80)
    print("XGBoost Feature Importance (Top 25 for Direction)")
    print("=" * 80)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False).head(25)

    for _, row in importance.iterrows():
        print(f"  {row['feature']:<46} {row['importance']:.4f}")

    print()
    print("=" * 80)
    print("PER-DATE AUC (Test 4 dates)")
    print("=" * 80)

    test_df_copy = test_df.copy()
    test_df_copy["lr_proba"] = lr_proba["test"]
    test_df_copy["xgb_proba"] = xgb_proba["test"]

    print(f"\n{'Date':<14} {'N':<6} {'Up %':<8} {'LogReg AUC':<12} {'XGB AUC':<12}")
    print("-" * 60)

    for date_str in DATES_TEST:
        sub = test_df_copy[test_df_copy["_source_date"] == date_str]
        if len(sub) > 0:
            if sub["target_binary"].nunique() > 1:
                try:
                    lr_auc = roc_auc_score(sub["target_binary"], sub["lr_proba"])
                    xgb_auc = roc_auc_score(sub["target_binary"], sub["xgb_proba"])
                    up = sub["target_binary"].mean()
                    print(f"{date_str:<14} {len(sub):<6} {up:<8.3f} {lr_auc:<12.3f} {xgb_auc:<12.3f}")
                except Exception as e:
                    print(f"{date_str:<14} ERROR: {e}")
            else:
                print(f"{date_str:<14} {len(sub):<6} single class")

    print()
    print("=" * 80)
    print("CALIBRATION (LogReg on test)")
    print("=" * 80)
    print(f"\n{'Bin':<15} {'Pred avg':<12} {'Actual':<12} {'Diff':<10} {'N':<8}")
    print("-" * 60)

    bins = [0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0]
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = (lr_proba["test"] >= low) & (lr_proba["test"] < high)
        n = mask.sum()
        if n > 5:
            pred_avg = lr_proba["test"][mask].mean()
            actual = y_test[mask].mean()
            diff = pred_avg - actual
            warn = " WARN" if abs(diff) > 0.10 else ""
            print(f"{low:.2f}-{high:.2f}      {pred_avg:<12.3f} {actual:<12.3f} {diff:<+10.3f} {n:<8}{warn}")

    print()
    print("=" * 80)
    print("SUMMARY + INTERPRETATION")
    print("=" * 80)

    test_lr_auc = roc_auc_score(y_test, lr_proba["test"])
    test_xgb_auc = roc_auc_score(y_test, xgb_proba["test"])
    test_lr_acc = accuracy_score(y_test, lr_pred["test"])
    test_xgb_acc = accuracy_score(y_test, xgb_pred["test"])

    _, _, train_bal_lr, _, _ = per_class(y_train, lr_pred["train"])
    _, _, test_bal_lr, _, _ = per_class(y_test, lr_pred["test"])

    print(f"\n--- Test set metrics ---")
    print(f"  Up baseline: {y_test.mean():.3f}")
    print(f"  LogReg:    acc={test_lr_acc:.3f}, AUC={test_lr_auc:.3f}, bal_acc={test_bal_lr:.3f}")
    print(f"  XGBoost:   acc={test_xgb_acc:.3f}, AUC={test_xgb_auc:.3f}")
    print(f"  Best threshold (LogReg): {best_thresh:.2f}, bal_acc={best_bal:.3f}")

    print(f"\n--- Train vs Test (overfit check) ---")
    print(f"  LogReg train bal_acc: {train_bal_lr:.3f}")
    print(f"  LogReg test bal_acc:  {test_bal_lr:.3f}")
    print(f"  Gap: {train_bal_lr - test_bal_lr:+.3f}")

    print(f"\n--- DECISION GUIDE ---")
    print(f"AUC < 0.51 -> Direction signal absent. Vol-only system.")
    print(f"AUC 0.51-0.53 -> Weak signal. Trading risky.")
    print(f"AUC 0.53-0.56 -> Meaningful. Risk mgmt + edge possible.")
    print(f"AUC > 0.56 -> Strong. Trading possible start.")
    print(f"AUC > 0.60 -> Very strong. 5%+ daily PnL hypothesis.")
    print(f"")
    print(f"Bal acc + threshold tuning:")
    print(f"  > 0.55 -> trading possible")
    print(f"  0.52-0.55 -> weak, combo strategy")
    print(f"  < 0.52 -> vol only")

    print(f"\n--- V2 (Vol) vs Direction comparison ---")
    print(f"  V2 (Vol):     LogReg test bal_acc 0.629, AUC 0.70 (per-date)")
    print(f"  Direction:    LogReg test bal_acc {test_bal_lr:.3f}, AUC {test_lr_auc:.3f}")

    out_path = Path("data/analysis_results")
    out_path.mkdir(exist_ok=True, parents=True)
    importance.to_csv(out_path / "xgb_direction_feature_importance.csv", index=False)

    log.info("\nDirection classifier analysis complete")


if __name__ == "__main__":
    main()
