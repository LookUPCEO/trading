"""시도 36 stochastic verification: 5 seeds × {raw, raw+norm, norm_only}."""
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("시도 36 seeds check: 5 seeds × 3 feature sets")
    log.info("=" * 70)

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
    feat_full = canonical + norm_features
    feat_norm_only = [f for f in canonical if f not in HIGH_SHIFT_FEATURES] + norm_features

    test_dates = SELF_ALL[1:]
    T = 0.20
    SEEDS = [42, 1, 2, 3, 4]

    from sklearn.metrics import roc_auc_score
    import xgboost as xgb_lib

    results = {"raw": [], "raw+norm": [], "norm_only": []}

    for seed in SEEDS:
        log.info(f"\n=== SEED {seed} ===")
        np.random.seed(seed)
        for feat_set, label in [(canonical, "raw"), (feat_full, "raw+norm"), (feat_norm_only, "norm_only")]:
            day_aucs = []
            for test_date in test_dates:
                test_dt_idx = SELF_ALL.index(test_date)
                train_self_dates = SELF_ALL[:test_dt_idx]
                self_train_df = pd.concat([self_dfs[d] for d in train_self_dates], ignore_index=True)
                self_test_df = self_dfs[test_date].copy()
                train_df = pd.concat([tardis_train, self_train_df], ignore_index=True)
                val_df = tardis_val

                meds = train_df.reindex(columns=feat_set).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
                def mx(df):
                    X = df.reindex(columns=feat_set).copy().replace([np.inf, -np.inf], np.nan)
                    return X.fillna(meds).fillna(0)
                Xt = mx(train_df); Xv = mx(val_df); Xst = mx(self_test_df)
                tm = train_df[dir_target].abs() > T
                vm = val_df[dir_target].abs() > T
                sm = self_test_df[dir_target].abs() > T
                Xt_f = Xt[tm].values; Xv_f = Xv[vm].values; Xst_f = Xst[sm].values
                y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
                y_dv = (val_df.loc[vm, dir_target] > 0).astype(int).values
                y_ds = (self_test_df.loc[sm, dir_target] > 0).astype(int).values

                clf = xgb_lib.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.03,
                                              min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
                                              reg_alpha=1.0, reg_lambda=5.0,
                                              random_state=seed, n_jobs=4, eval_metric="auc",
                                              early_stopping_rounds=30)
                clf.fit(Xt_f, y_dt, eval_set=[(Xv_f, y_dv)], verbose=False)
                a_s = roc_auc_score(y_ds, clf.predict_proba(Xst_f)[:, 1]) if len(set(y_ds)) > 1 else float("nan")
                day_aucs.append(a_s)
            mean_auc = float(np.nanmean(day_aucs))
            results[label].append({"seed": seed, "mean_auc": mean_auc, "day_aucs": day_aucs})
            log.info(f"  {label:<12}  seed {seed}  mean AUC {mean_auc:.3f}  std {np.nanstd(day_aucs):.3f}")

    # Aggregate
    print()
    print("=" * 90)
    print("STOCHASTIC VERIFICATION (5 seeds × 3 feature sets, 9 walk-forward days each)")
    print("=" * 90)
    print(f"\n{'Set':<14} {'Seed1':<8} {'Seed2':<8} {'Seed3':<8} {'Seed4':<8} {'Seed5':<8} {'Avg':<8} {'Std':<8}")
    print("-" * 70)
    for label, runs in results.items():
        means = [r["mean_auc"] for r in runs]
        avg = float(np.mean(means)); std = float(np.std(means))
        cells = "  ".join([f"{m:.3f}" for m in means])
        print(f"{label:<14} {cells}  {avg:.3f}   {std:.3f}")

    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido36_seeds_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    # Diagnosis
    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    raw_means = [r["mean_auc"] for r in results["raw"]]
    full_means = [r["mean_auc"] for r in results["raw+norm"]]
    norm_means = [r["mean_auc"] for r in results["norm_only"]]
    print(f"\n  raw      : {np.mean(raw_means):.3f} ± {np.std(raw_means):.3f}  (across 5 seeds)")
    print(f"  raw+norm : {np.mean(full_means):.3f} ± {np.std(full_means):.3f}")
    print(f"  norm_only: {np.mean(norm_means):.3f} ± {np.std(norm_means):.3f}")
    print(f"\n  vs sido32 baseline (n100 d5): 0.520")
    print(f"  vs sido32 best (n1000 d6): 0.554")
    print(f"\n  raw improvement over sido32 best: {np.mean(raw_means) - 0.554:+.3f}")
    print(f"  norm improvement over sido32 best: {np.mean(norm_means) - 0.554:+.3f}")
    log.info("\n시도 36 seeds check complete")


if __name__ == "__main__":
    main()
