"""ML Volatility Classifier."""
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
    log.info("Building train/val/test sets")
    log.info("=" * 70)

    log.info(f"\nTRAIN ({len(DATES_TRAIN)} dates):")
    train_df = build_split(DATES_TRAIN, log)

    log.info(f"\nVAL ({len(DATES_VAL)} dates):")
    val_df = build_split(DATES_VAL, log)

    log.info(f"\nTEST ({len(DATES_TEST)} dates):")
    test_df = build_split(DATES_TEST, log)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        log.error("Missing data")
        return

    target_col = "target_volatility_300s"
    train_median = train_df[target_col].median()
    log.info(f"\nTrain median volatility_300s: {train_median:.6f}")

    train_df["target_binary"] = (train_df[target_col] > train_median).astype(int)
    val_df["target_binary"] = (val_df[target_col] > train_median).astype(int)
    test_df["target_binary"] = (test_df[target_col] > train_median).astype(int)

    log.info(f"\nClass balance (high vol fraction):")
    log.info(f"  Train: {train_df['target_binary'].mean():.3f}")
    log.info(f"  Val:   {val_df['target_binary'].mean():.3f}")
    log.info(f"  Test:  {test_df['target_binary'].mean():.3f}")

    feature_cols = get_feature_columns(train_df)
    log.info(f"\nFeatures: {len(feature_cols)}")
    log.info(f"Sample: {feature_cols[:5]}")

    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        before = len(df)
        df.dropna(subset=[target_col, "target_binary"], inplace=True)
        after = len(df)
        log.info(f"  {df_name}: {before} → {after} (NaN target dropped)")

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

    log.info(f"\nFinal shapes:")
    log.info(f"  Train: X={X_train.shape}, y={y_train.shape}")
    log.info(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    log.info(f"  Test:  X={X_test.shape}, y={y_test.shape}")

    print()
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

    def evaluate(name, y_true, y_pred, y_proba=None):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
        if auc is not None:
            print(f"  {name:<25}: acc={acc:.3f} auc={auc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")
        else:
            print(f"  {name:<25}: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")
        return {"name": name, "acc": acc, "auc": auc}

    results = {}

    print(f"\n--- 1. Random baseline ---")
    np.random.seed(42)
    rand_pred_val = np.random.randint(0, 2, size=len(y_val))
    rand_pred_test = np.random.randint(0, 2, size=len(y_test))
    rand_proba_val = np.random.random(size=len(y_val))
    rand_proba_test = np.random.random(size=len(y_test))
    results["random_val"] = evaluate("Random (val)", y_val, rand_pred_val, rand_proba_val)
    results["random_test"] = evaluate("Random (test)", y_test, rand_pred_test, rand_proba_test)

    print(f"\n--- 2. Persistence baseline ---")
    PERSISTENCE_COL = "ob_mid_price_std_300s"
    if PERSISTENCE_COL in train_df.columns:
        train_persist_median = train_df[PERSISTENCE_COL].median()
        log.info(f"  Persistence threshold (train median {PERSISTENCE_COL}): {train_persist_median:.6f}")

        pers_val_vals = val_df[PERSISTENCE_COL].fillna(train_persist_median).values
        pers_test_vals = test_df[PERSISTENCE_COL].fillna(train_persist_median).values

        pers_pred_val = (pers_val_vals > train_persist_median).astype(int)
        pers_pred_test = (pers_test_vals > train_persist_median).astype(int)

        max_val = max(pers_val_vals.max(), pers_test_vals.max(), 1e-10)
        pers_proba_val = pers_val_vals / max_val
        pers_proba_test = pers_test_vals / max_val

        results["pers_val"] = evaluate("Persistence (val)", y_val, pers_pred_val, pers_proba_val)
        results["pers_test"] = evaluate("Persistence (test)", y_test, pers_pred_test, pers_proba_test)
    else:
        print(f"  {PERSISTENCE_COL} not in train_df — skipping")

    print(f"\n--- 3. Logistic Regression ---")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled)
    X_val_scaled = scaler.transform(X_val_filled)
    X_test_scaled = scaler.transform(X_test_filled)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    lr_pred_train = lr.predict(X_train_scaled)
    lr_pred_val = lr.predict(X_val_scaled)
    lr_pred_test = lr.predict(X_test_scaled)
    lr_proba_train = lr.predict_proba(X_train_scaled)[:, 1]
    lr_proba_val = lr.predict_proba(X_val_scaled)[:, 1]
    lr_proba_test = lr.predict_proba(X_test_scaled)[:, 1]

    results["lr_train"] = evaluate("LogReg (train)", y_train, lr_pred_train, lr_proba_train)
    results["lr_val"] = evaluate("LogReg (val)", y_val, lr_pred_val, lr_proba_val)
    results["lr_test"] = evaluate("LogReg (test)", y_test, lr_pred_test, lr_proba_test)

    print(f"\n--- 4. XGBoost ---")
    import xgboost as xgb

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=20,
    )

    xgb_model.fit(
        X_train_filled, y_train,
        eval_set=[(X_val_filled, y_val)],
    )
    log.info(f"  Best iteration: {xgb_model.best_iteration}")

    xgb_pred_train = xgb_model.predict(X_train_filled)
    xgb_pred_val = xgb_model.predict(X_val_filled)
    xgb_pred_test = xgb_model.predict(X_test_filled)
    xgb_proba_train = xgb_model.predict_proba(X_train_filled)[:, 1]
    xgb_proba_val = xgb_model.predict_proba(X_val_filled)[:, 1]
    xgb_proba_test = xgb_model.predict_proba(X_test_filled)[:, 1]

    results["xgb_train"] = evaluate("XGBoost (train)", y_train, xgb_pred_train, xgb_proba_train)
    results["xgb_val"] = evaluate("XGBoost (val)", y_val, xgb_pred_val, xgb_proba_val)
    results["xgb_test"] = evaluate("XGBoost (test)", y_test, xgb_pred_test, xgb_proba_test)

    print()
    print("=" * 70)
    print("XGBoost Feature Importance (Top 20)")
    print("=" * 70)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False).head(20)

    for _, row in importance.iterrows():
        print(f"  {row['feature']:<46} {row['importance']:.4f}")

    print()
    print("=" * 70)
    print("Per-date Test Performance (XGBoost)")
    print("=" * 70)

    test_df_copy = test_df.copy()
    test_df_copy["xgb_pred"] = xgb_pred_test

    for date_str in DATES_TEST:
        date_subset = test_df_copy[test_df_copy["_source_date"] == date_str]
        if len(date_subset) > 0:
            acc = (date_subset["target_binary"] == date_subset["xgb_pred"]).mean()
            print(f"  {date_str}: n={len(date_subset)}, accuracy={acc:.3f}")

    print()
    print("=" * 70)
    print("SUMMARY: Test set accuracy")
    print("=" * 70)
    print(f"  Random:       {results.get('random_test', {}).get('acc', 0):.3f}")
    if 'pers_test' in results:
        print(f"  Persistence:  {results['pers_test']['acc']:.3f}")
    print(f"  LogReg:       {results['lr_test']['acc']:.3f}")
    print(f"  XGBoost:      {results['xgb_test']['acc']:.3f}")

    print()
    print("Train vs Test gap (overfit check):")
    print(f"  LogReg:  train={results['lr_train']['acc']:.3f} test={results['lr_test']['acc']:.3f} gap={results['lr_train']['acc']-results['lr_test']['acc']:+.3f}")
    print(f"  XGBoost: train={results['xgb_train']['acc']:.3f} test={results['xgb_test']['acc']:.3f} gap={results['xgb_train']['acc']-results['xgb_test']['acc']:+.3f}")

    out_path = Path("data/analysis_results")
    out_path.mkdir(exist_ok=True, parents=True)
    importance.to_csv(out_path / "xgb_feature_importance.csv", index=False)
    log.info(f"Feature importance saved")


if __name__ == "__main__":
    main()
