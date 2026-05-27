"""시도 37: Regime classifier + per-regime sub-models, walk-forward 9 days."""
import sys, logging, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

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


def label_regime(df, vol_target="target_volatility_300s", dir_target="target_return_3600s",
                 vol_q_low=0.33, vol_q_high=0.67, trend_thr=0.30):
    """Per-day regime label: combine vol level (low/mid/high) + trend (chop/up/down).

    Returns Series of integer labels:
      0: low_vol_chop, 1: low_vol_trend
      2: mid_vol_chop, 3: mid_vol_trend
      4: high_vol_chop, 5: high_vol_trend
    """
    if "_source_date" not in df.columns:
        return pd.Series(2, index=df.index)  # default mid_vol_chop

    # Per-day vol mean
    day_vol = df.groupby("_source_date")[vol_target].transform("mean")
    # Per-day trend strength = sum(direction) / sum(|direction|)
    df_copy = df[[dir_target, "_source_date"]].copy()
    sign_sum = df_copy.groupby("_source_date")[dir_target].transform("sum")
    abs_sum = df_copy.groupby("_source_date")[dir_target].transform(lambda s: s.abs().sum())
    trend = (sign_sum / abs_sum.replace(0, np.nan)).fillna(0)

    # Vol percentiles (within all dates)
    vol_low = day_vol.quantile(vol_q_low)
    vol_high = day_vol.quantile(vol_q_high)

    vol_class = pd.Series(1, index=df.index)  # mid
    vol_class[day_vol < vol_low] = 0  # low
    vol_class[day_vol > vol_high] = 2  # high

    is_trend = (trend.abs() > trend_thr).astype(int)

    label = vol_class * 2 + is_trend
    return label.astype(int)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("시도 37: Regime classifier + per-regime sub-models")
    log.info("=" * 70)
    np.random.seed(42)

    log.info("\nBuilding...")
    tardis_train = build_split(DATES_TRAIN, log)
    tardis_val = build_split(DATES_VAL, log)
    feat_pre = get_feature_columns(tardis_train)
    tt_clean = tardis_train.dropna(subset=["target_volatility_300s", "target_return_3600s"])
    medians = tt_clean.reindex(columns=feat_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    SELF_ALL = [f"2026-04-{d:02d}" for d in range(21, 31)]
    self_dfs = {}
    for d in SELF_ALL:
        df = build_self_date_dataset(d, log, train_medians=medians)
        df = df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
        self_dfs[d] = df

    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
    tardis_train.dropna(subset=[vol_target, dir_target], inplace=True)
    tardis_val.dropna(subset=[vol_target, dir_target], inplace=True)
    tardis_train = add_normalized_features(tardis_train, log)
    tardis_val = add_normalized_features(tardis_val, log)
    for d in self_dfs:
        self_dfs[d] = add_normalized_features(self_dfs[d], log)

    norm_features = [c for c in tardis_train.columns if c.endswith("_norm")]
    canonical = get_feature_columns(tardis_train)
    feat_set = [f for f in canonical if f not in HIGH_SHIFT_FEATURES] + norm_features
    log.info(f"\nFeature set (norm_only): {len(feat_set)}")

    # Label regimes for full Tardis train (used by sub-models)
    tardis_train["_regime"] = label_regime(tardis_train, vol_target, dir_target)
    log.info(f"\nTardis train regime distribution:")
    log.info(tardis_train["_regime"].value_counts().sort_index().to_dict())

    test_dates = SELF_ALL[1:]
    T = 0.20

    from sklearn.metrics import roc_auc_score
    import xgboost as xgb_lib

    walk_results = []
    for step_idx, test_date in enumerate(test_dates, 1):
        test_dt_idx = SELF_ALL.index(test_date)
        train_self_dates = SELF_ALL[:test_dt_idx]
        log.info(f"\n=== STEP {step_idx}/9: train Self {len(train_self_dates)}d, test {test_date} ===")

        self_train_df = pd.concat([self_dfs[d] for d in train_self_dates], ignore_index=True)
        self_train_df["_regime"] = label_regime(self_train_df, vol_target, dir_target)
        self_test_df = self_dfs[test_date].copy()
        self_test_df["_regime"] = label_regime(self_test_df, vol_target, dir_target)
        train_df = pd.concat([tardis_train, self_train_df], ignore_index=True)
        val_df = tardis_val

        meds = train_df.reindex(columns=feat_set).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        def mx(df):
            X = df.reindex(columns=feat_set).copy().replace([np.inf, -np.inf], np.nan)
            return X.fillna(meds).fillna(0)

        # Direction filter
        tm = train_df[dir_target].abs() > T
        sm = self_test_df[dir_target].abs() > T
        train_filt = train_df[tm].copy()
        self_filt = self_test_df[sm].copy()

        # ---- Approach A: Single pooled model (시도 36 baseline) ----
        Xt_pool = mx(train_filt).values
        Xst_pool = mx(self_filt).values
        y_pool = (train_filt[dir_target] > 0).astype(int).values
        y_self = (self_filt[dir_target] > 0).astype(int).values
        clf_pool = xgb_lib.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.03,
                                           min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
                                           reg_alpha=1.0, reg_lambda=5.0,
                                           random_state=42, n_jobs=4, eval_metric="auc",
                                           early_stopping_rounds=30)
        Xv_pool = mx(val_df[val_df[dir_target].abs() > T]).values
        y_pool_v = (val_df.loc[val_df[dir_target].abs() > T, dir_target] > 0).astype(int).values
        clf_pool.fit(Xt_pool, y_pool, eval_set=[(Xv_pool, y_pool_v)], verbose=False)
        a_pool = roc_auc_score(y_self, clf_pool.predict_proba(Xst_pool)[:, 1]) if len(set(y_self)) > 1 else float("nan")

        # ---- Approach B: Per-regime sub-models ----
        per_regime_preds = np.zeros(len(self_filt))
        per_regime_auc = {}
        regime_groups = train_filt.groupby("_regime")
        sub_models = {}
        for regime, sub_train in regime_groups:
            if len(sub_train) < 500:
                continue
            X_sub = mx(sub_train).values
            y_sub = (sub_train[dir_target] > 0).astype(int).values
            try:
                clf_sub = xgb_lib.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03,
                                                  min_child_weight=50, subsample=0.8, colsample_bytree=0.7,
                                                  reg_alpha=1.0, reg_lambda=3.0,
                                                  random_state=42, n_jobs=4, eval_metric="auc")
                clf_sub.fit(X_sub, y_sub, verbose=False)
                sub_models[int(regime)] = clf_sub
            except Exception as e:
                log.warning(f"  regime {regime}: train fail: {e}")

        # Apply sub-models to test rows by their regime label
        # Fallback to pooled model if sub-model missing
        for i, (idx, test_row) in enumerate(self_filt.iterrows()):
            test_regime = int(test_row["_regime"])
            X_one = mx(self_filt.iloc[[i]]).values
            if test_regime in sub_models:
                per_regime_preds[i] = sub_models[test_regime].predict_proba(X_one)[0, 1]
            else:
                per_regime_preds[i] = clf_pool.predict_proba(X_one)[0, 1]

        a_per_regime = roc_auc_score(y_self, per_regime_preds) if len(set(y_self)) > 1 else float("nan")

        # ---- Approach C: Ensemble (pooled + per-regime, average) ----
        pooled_pred = clf_pool.predict_proba(Xst_pool)[:, 1]
        ensemble_pred = (pooled_pred + per_regime_preds) / 2.0
        a_ensemble = roc_auc_score(y_self, ensemble_pred) if len(set(y_self)) > 1 else float("nan")

        log.info(f"  AUC self: pooled {a_pool:.3f}  per-regime {a_per_regime:.3f}  ensemble {a_ensemble:.3f}")
        walk_results.append({
            "step": step_idx, "test_date": test_date,
            "n_train_self": len(train_self_dates),
            "auc_pooled": float(a_pool), "auc_per_regime": float(a_per_regime),
            "auc_ensemble": float(a_ensemble),
            "test_regime_dist": self_test_df["_regime"].value_counts().sort_index().to_dict(),
        })

    # Aggregate
    print()
    print("=" * 90)
    print("REGIME CLASSIFIER WALK-FORWARD (9 days)")
    print("=" * 90)
    print(f"\n{'Step':<6} {'Date':<14} {'pooled':<10} {'per-regime':<12} {'ensemble':<10} {'regimes':<14}")
    print("-" * 80)
    for r in walk_results:
        rd_str = ",".join(f"{k}:{v}" for k, v in r["test_regime_dist"].items())[:14]
        print(f"{r['step']:<6} {r['test_date']:<14} {r['auc_pooled']:<10.3f} {r['auc_per_regime']:<12.3f} {r['auc_ensemble']:<10.3f} {rd_str:<14}")

    p = np.array([r["auc_pooled"] for r in walk_results])
    pr = np.array([r["auc_per_regime"] for r in walk_results])
    e = np.array([r["auc_ensemble"] for r in walk_results])
    print(f"\n  pooled    : mean {p.mean():.3f}  std {p.std():.3f}  >0.55 {(p>0.55).sum()}/9")
    print(f"  per-regime: mean {pr.mean():.3f}  std {pr.std():.3f}  >0.55 {(pr>0.55).sum()}/9")
    print(f"  ensemble  : mean {e.mean():.3f}  std {e.std():.3f}  >0.55 {(e>0.55).sum()}/9")

    out = {"approach": "Regime classifier sub-models", "steps": walk_results,
           "summary": {
               "pooled_mean": float(p.mean()), "per_regime_mean": float(pr.mean()),
               "ensemble_mean": float(e.mean()),
           }}
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido37_regime_classifier.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    # Diagnosis
    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print(f"\n  pooled (시도 36): {p.mean():.3f}")
    print(f"  per-regime: {pr.mean():.3f}  (Δ {pr.mean() - p.mean():+.3f} vs pooled)")
    print(f"  ensemble : {e.mean():.3f}  (Δ {e.mean() - p.mean():+.3f} vs pooled)")
    if pr.mean() > p.mean() + 0.01 or e.mean() > p.mean() + 0.01:
        print(f"\n  ✅ Regime classifier 효과 있음")
    else:
        print(f"\n  ❌ Regime classifier 효과 없음. pooled = optimal.")
    log.info("\n시도 37 complete")


if __name__ == "__main__":
    main()
