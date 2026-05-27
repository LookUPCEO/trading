"""Save mark36_v1.joblib — norm_only XGB n1000 d6 mcw100, full Tardis + Self 9d train."""
import sys, logging, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from mark19.ml.data_prep import DATES_TRAIN, DATES_VAL, build_split, get_feature_columns

import importlib.util
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_bsd", _HERE / "backtest_self_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_self_date_dataset = _mod.build_self_date_dataset

_spec36 = importlib.util.spec_from_file_location("_sido36", _HERE / "sido36_regime_invariant.py")
_mod36 = importlib.util.module_from_spec(_spec36)
_spec36.loader.exec_module(_mod36)
add_normalized_features = _mod36.add_normalized_features
HIGH_SHIFT_FEATURES = _mod36.HIGH_SHIFT_FEATURES


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("Save mark36_v1: norm_only XGB n1000 d6 mcw100 full retrain")
    log.info("=" * 70)
    np.random.seed(42)

    log.info("\nBuilding...")
    tardis_train = build_split(DATES_TRAIN, log)
    tardis_val = build_split(DATES_VAL, log)
    feat_pre = get_feature_columns(tardis_train)
    tt_clean = tardis_train.dropna(subset=["target_volatility_300s", "target_return_3600s"])
    tardis_medians = tt_clean.reindex(columns=feat_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    SELF_ALL = [f"2026-04-{d:02d}" for d in range(21, 31)] + ["2026-05-01"]
    self_dfs = []
    for d in SELF_ALL:
        try:
            df = build_self_date_dataset(d, log, train_medians=tardis_medians)
            df = df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
            self_dfs.append(df)
        except Exception as e:
            log.warning(f"  {d}: {e}")
    self_all = pd.concat(self_dfs, ignore_index=True) if self_dfs else pd.DataFrame()

    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
    tardis_train.dropna(subset=[vol_target, dir_target], inplace=True)
    tardis_val.dropna(subset=[vol_target, dir_target], inplace=True)

    tardis_train = add_normalized_features(tardis_train, log)
    tardis_val = add_normalized_features(tardis_val, log)
    self_all = add_normalized_features(self_all, log)

    norm_features = [c for c in tardis_train.columns if c.endswith("_norm")]
    canonical = get_feature_columns(tardis_train)
    feat_set = [f for f in canonical if f not in HIGH_SHIFT_FEATURES] + norm_features
    log.info(f"\nFeature set (norm_only): {len(feat_set)}")

    # Combined train: Tardis full + Self all
    train_df = pd.concat([tardis_train, self_all], ignore_index=True)
    val_df = tardis_val
    log.info(f"Train: {len(train_df)}  Val: {len(val_df)}")

    meds = train_df.reindex(columns=feat_set).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    def mx(df):
        X = df.reindex(columns=feat_set).copy().replace([np.inf, -np.inf], np.nan)
        return X.fillna(meds).fillna(0)
    Xt = mx(train_df); Xv = mx(val_df)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb_lib

    # Vol LR
    vol_med = float(train_df[vol_target].median())
    y_vt = (train_df[vol_target] > vol_med).astype(int).values
    sv = StandardScaler(); X_tv = sv.fit_transform(Xt)
    lrv = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lrv.fit(X_tv, y_vt)
    y_vv = (val_df[vol_target] > vol_med).astype(int).values
    vol_auc_val = roc_auc_score(y_vv, lrv.predict_proba(sv.transform(Xv))[:, 1])
    log.info(f"\nVol LR AUC val: {vol_auc_val:.3f}")

    # Direction XGB n1000 d6 mcw100
    T = 0.20
    tm = train_df[dir_target].abs() > T
    vm = val_df[dir_target].abs() > T
    Xt_f = Xt[tm].values; Xv_f = Xv[vm].values
    y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
    y_dv = (val_df.loc[vm, dir_target] > 0).astype(int).values

    log.info(f"Direction samples: train {len(y_dt)}  val {len(y_dv)}")
    clf = xgb_lib.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.03,
                                  min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
                                  reg_alpha=1.0, reg_lambda=5.0,
                                  random_state=42, n_jobs=4, eval_metric="auc",
                                  early_stopping_rounds=30)
    clf.fit(Xt_f, y_dt, eval_set=[(Xv_f, y_dv)], verbose=False)
    auc_train = roc_auc_score(y_dt, clf.predict_proba(Xt_f)[:, 1])
    auc_val = roc_auc_score(y_dv, clf.predict_proba(Xv_f)[:, 1])
    log.info(f"\nDir XGB AUC: train {auc_train:.3f}  val {auc_val:.3f}  best_iter {clf.best_iteration}")

    # Save
    out = {
        "lr_vol": lrv, "scaler_vol": sv,
        "xgb_dir": clf,
        "feature_cols": feat_set,
        "high_shift_features": HIGH_SHIFT_FEATURES,
        "norm_features": norm_features,
        "train_medians": meds.to_dict(),
        "vol_med": vol_med, "T": T,
        "metadata": {
            "approach": "norm_only XGB n1000 d6 mcw100 (sido36)",
            "n_train_total": int(len(train_df)),
            "n_dir_train": int(len(y_dt)),
            "n_dir_val": int(len(y_dv)),
            "vol_auc_val": float(vol_auc_val),
            "dir_auc_train": float(auc_train),
            "dir_auc_val": float(auc_val),
            "best_iter": int(clf.best_iteration if clf.best_iteration else 1000),
            "5seed_walk_forward_mean_auc": 0.608,
            "5seed_walk_forward_std_auc": 0.021,
            "9day_backtest_daily_pnl_mean_pct": 0.590,
            "9day_backtest_dir_th": 0.55,
            "self_dates_used": SELF_ALL,
        },
    }
    model_path = Path("/Users/dohun/Desktop/Mark/mark19/models/mark36_v1.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, model_path, compress=3)
    log.info(f"\nSaved: {model_path}")

    # Save metadata as JSON
    md_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/mark36_v1_metadata.json")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        json.dump(out["metadata"], f, indent=2, default=str)
    log.info(f"Metadata: {md_path}")


if __name__ == "__main__":
    main()
