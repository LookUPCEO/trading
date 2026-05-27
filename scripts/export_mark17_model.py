"""Mark17 (시도 17) 모델 → joblib export.

저장 내용:
- vol_lr: LogisticRegression for volatility
- vol_scaler: StandardScaler for vol model
- dir_lr: LogisticRegression for direction (Triple-barrier T=0.20% filtered)
- dir_scaler: StandardScaler for dir model
- feature_cols: list of feature column names (~170)
- train_medians: dict of {col: median} for fillna
- vol_threshold: 0.6
- dir_threshold: 0.65
- vol_target_median: train median of target_volatility_300s
- model_version: "mark17_v1"
- training_dates: DATES_TRAIN list
- exported_at: datetime
"""
import logging
import sys
import joblib
from datetime import datetime, timezone
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
    log.info("MARK17 MODEL EXPORT (시도 17)")
    log.info("=" * 70)

    # 1. Build datasets
    log.info("\nBuilding datasets...")
    train_df = build_split(DATES_TRAIN, log)
    val_df = build_split(DATES_VAL, log)
    test_df = build_split(DATES_TEST, log)

    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"

    for df in [train_df, val_df, test_df]:
        df.dropna(subset=[vol_target, dir_target], inplace=True)

    log.info(f"  Train rows: {len(train_df)}")
    log.info(f"  Val rows:   {len(val_df)}")
    log.info(f"  Test rows:  {len(test_df)}")

    feature_cols = get_feature_columns(train_df)
    log.info(f"Features: {len(feature_cols)}")

    cross_count = sum(1 for c in feature_cols if c.startswith("cross_"))
    adapt_count = sum(1 for c in feature_cols if c.startswith("adapt_"))
    log.info(f"  Cross features: {cross_count}")
    log.info(f"  Adaptive features: {adapt_count}")

    # 2. Train medians
    log.info("\nComputing train medians...")
    X_train_raw = train_df.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    train_medians = X_train_raw.median(numeric_only=True)

    def make_X(df, feat_cols, medians):
        X = df.reindex(columns=feat_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.fillna(medians).fillna(0)

    X_train = make_X(train_df, feature_cols, train_medians)
    X_val = make_X(val_df, feature_cols, train_medians)
    X_test = make_X(test_df, feature_cols, train_medians)

    # 3. Vol model
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    log.info("\nTraining Vol model...")
    train_vol_median = float(train_df[vol_target].median())
    y_vol_train = (train_df[vol_target] > train_vol_median).astype(int).values

    vol_scaler = StandardScaler()
    X_train_vol_scaled = vol_scaler.fit_transform(X_train)
    X_test_vol_scaled = vol_scaler.transform(X_test)

    vol_lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    vol_lr.fit(X_train_vol_scaled, y_vol_train)

    vol_proba_test = vol_lr.predict_proba(X_test_vol_scaled)[:, 1]
    y_vol_test = (test_df[vol_target] > train_vol_median).astype(int).values
    vol_auc = float(roc_auc_score(y_vol_test, vol_proba_test))
    log.info(f"  Vol AUC (test): {vol_auc:.3f} (expected ~0.762)")

    # 4. Direction model (Triple-barrier T=0.20)
    log.info("\nTraining Direction model...")
    T = 0.20
    train_dir_mask = train_df[dir_target].abs() > T

    X_train_dir = X_train[train_dir_mask].values
    y_dir_train = (train_df.loc[train_dir_mask, dir_target] > 0).astype(int).values

    log.info(f"  Train sample (filtered): {len(X_train_dir)}")

    dir_scaler = StandardScaler()
    X_train_dir_scaled = dir_scaler.fit_transform(X_train_dir)
    X_test_dir_scaled = dir_scaler.transform(X_test.values)

    dir_lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    dir_lr.fit(X_train_dir_scaled, y_dir_train)

    dir_proba_test = dir_lr.predict_proba(X_test_dir_scaled)[:, 1]
    test_dir_mask = test_df[dir_target].abs() > T
    y_dir_test_subset = (test_df.loc[test_dir_mask, dir_target] > 0).astype(int).values
    dir_auc = float(roc_auc_score(y_dir_test_subset, dir_proba_test[test_dir_mask.values]))
    log.info(f"  Direction AUC (test): {dir_auc:.3f} (expected ~0.545)")

    # 5. Save
    log.info("\nSaving model to joblib...")

    model_data = {
        "vol_lr": vol_lr,
        "vol_scaler": vol_scaler,
        "dir_lr": dir_lr,
        "dir_scaler": dir_scaler,

        "feature_cols": feature_cols,
        "train_medians": train_medians.to_dict(),
        "vol_target_median": train_vol_median,

        "vol_threshold": 0.6,
        "dir_threshold": 0.65,
        "triple_barrier_T": T,

        "model_version": "mark17_v1",
        "training_dates": DATES_TRAIN,
        "val_dates": DATES_VAL,
        "test_dates": DATES_TEST,
        "n_features": len(feature_cols),
        "n_cross_features": cross_count,
        "n_adaptive_features": adapt_count,
        "vol_auc": vol_auc,
        "dir_auc": dir_auc,
        "exported_at": datetime.now(timezone.utc).isoformat(),

        "sample_test_row_0": X_test.iloc[0].to_dict(),
        "sample_vol_proba_0": float(vol_proba_test[0]),
        "sample_dir_proba_0": float(dir_proba_test[0]),
    }

    output_path = Path("/Users/dohun/Desktop/Mark/mark19/models/mark17_v1.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_data, output_path, compress=3)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    log.info(f"  Saved to: {output_path}")
    log.info(f"  Size: {file_size_mb:.2f} MB")

    # 6. Reproducibility check
    log.info("\nReproducibility check (load + verify)...")

    loaded = joblib.load(output_path)

    expected_keys = ["vol_lr", "vol_scaler", "dir_lr", "dir_scaler",
                     "feature_cols", "train_medians", "vol_target_median",
                     "vol_threshold", "dir_threshold", "triple_barrier_T",
                     "model_version", "n_features",
                     "sample_test_row_0", "sample_vol_proba_0", "sample_dir_proba_0"]
    missing = [k for k in expected_keys if k not in loaded]
    if missing:
        log.error(f"  Missing keys: {missing}")
    else:
        log.info(f"  All keys present ({len(loaded)} total)")

    # Verify reproducibility
    test_row_0 = pd.Series(loaded["sample_test_row_0"]).reindex(loaded["feature_cols"])
    test_row_0_filled = test_row_0.fillna(pd.Series(loaded["train_medians"])).fillna(0)
    test_row_0_arr = test_row_0_filled.values.reshape(1, -1)

    vol_scaled = loaded["vol_scaler"].transform(test_row_0_arr)
    vol_proba_loaded = float(loaded["vol_lr"].predict_proba(vol_scaled)[0, 1])

    dir_scaled = loaded["dir_scaler"].transform(test_row_0_arr)
    dir_proba_loaded = float(loaded["dir_lr"].predict_proba(dir_scaled)[0, 1])

    expected_vol = loaded["sample_vol_proba_0"]
    expected_dir = loaded["sample_dir_proba_0"]

    vol_diff = abs(vol_proba_loaded - expected_vol)
    dir_diff = abs(dir_proba_loaded - expected_dir)

    log.info(f"\n  Vol proba: expected {expected_vol:.6f}, loaded {vol_proba_loaded:.6f}, diff {vol_diff:.2e}")
    log.info(f"  Dir proba: expected {expected_dir:.6f}, loaded {dir_proba_loaded:.6f}, diff {dir_diff:.2e}")

    if vol_diff < 1e-6 and dir_diff < 1e-6:
        log.info(f"  Reproducibility OK (diff < 1e-6)")
    else:
        log.warning(f"  Reproducibility drift > 1e-6 - check feature_cols/medians order")

    # 7. Summary
    print()
    print("=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    print(f"Model: {loaded['model_version']}")
    print(f"Path: {output_path}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Features: {loaded['n_features']} ({loaded['n_cross_features']} cross + {loaded['n_adaptive_features']} adaptive)")
    print(f"Vol AUC: {loaded['vol_auc']:.3f}")
    print(f"Direction AUC: {loaded['dir_auc']:.3f}")
    print(f"Vol threshold: {loaded['vol_threshold']}")
    print(f"Dir threshold: {loaded['dir_threshold']}")
    print(f"Triple-barrier T: {loaded['triple_barrier_T']}")
    print(f"Exported: {loaded['exported_at']}")
    print()
    print("Reproducibility:")
    print(f"  Vol diff: {vol_diff:.2e}")
    print(f"  Dir diff: {dir_diff:.2e}")
    print()

    if vol_diff < 1e-6 and dir_diff < 1e-6:
        print("Model export OK - ready for live trading")
    else:
        print("Model export has drift - investigate")

    log.info("Done")


if __name__ == "__main__":
    main()
