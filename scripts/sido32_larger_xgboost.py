"""시도 32: Larger XGBoost (n_estimators 500-2000, depth 6-10), walk-forward."""
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("시도 32: Larger XGBoost sweep (walk-forward 9 days)")
    log.info("=" * 70)
    np.random.seed(42)

    log.info("\nBuilding Tardis...")
    tardis_train_df = build_split(DATES_TRAIN, log)
    tardis_val_df = build_split(DATES_VAL, log)
    feat_pre = get_feature_columns(tardis_train_df)
    tt_clean = tardis_train_df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
    medians = tt_clean.reindex(columns=feat_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    SELF_ALL = [f"2026-04-{d:02d}" for d in range(21, 31)]
    self_dfs = {}
    log.info("Building Self...")
    for d in SELF_ALL:
        df = build_self_date_dataset(d, log, train_medians=medians)
        df = df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
        self_dfs[d] = df

    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"
    tardis_train_df.dropna(subset=[vol_target, dir_target], inplace=True)
    tardis_val_df.dropna(subset=[vol_target, dir_target], inplace=True)
    base_features = get_feature_columns(tardis_train_df)
    log.info(f"\nFeatures: {len(base_features)}")

    # XGB configs
    configs = [
        {"name": "XGB n500 d6 mcw50",
         "p": dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                   min_child_weight=50, subsample=0.8, colsample_bytree=0.7,
                   reg_alpha=1.0, reg_lambda=1.0)},
        {"name": "XGB n1000 d6 mcw100",
         "p": dict(n_estimators=1000, max_depth=6, learning_rate=0.03,
                   min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
                   reg_alpha=1.0, reg_lambda=5.0)},
        {"name": "XGB n1000 d8 mcw100",
         "p": dict(n_estimators=1000, max_depth=8, learning_rate=0.03,
                   min_child_weight=100, subsample=0.7, colsample_bytree=0.6,
                   reg_alpha=1.0, reg_lambda=5.0)},
        {"name": "XGB n2000 d6 mcw200",
         "p": dict(n_estimators=2000, max_depth=6, learning_rate=0.02,
                   min_child_weight=200, subsample=0.8, colsample_bytree=0.6,
                   reg_alpha=5.0, reg_lambda=5.0)},
        {"name": "XGB n2000 d10 mcw200",
         "p": dict(n_estimators=2000, max_depth=10, learning_rate=0.02,
                   min_child_weight=200, subsample=0.7, colsample_bytree=0.5,
                   reg_alpha=5.0, reg_lambda=5.0)},
        # Baseline (sido29)
        {"name": "XGB n100 d5 (baseline)",
         "p": dict(n_estimators=100, max_depth=5, learning_rate=0.05,
                   min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.1, reg_lambda=1.0)},
    ]
    common = dict(random_state=42, n_jobs=4, eval_metric="auc", early_stopping_rounds=30)

    test_dates = SELF_ALL[1:]  # 4/22..4/30 (9 days)
    T = 0.20

    from sklearn.metrics import roc_auc_score
    import xgboost as xgb_lib

    walk_results = []  # list of {step, test_date, configs: {name: {auc_train, auc_val, auc_self, best_iter}}}

    for step_idx, test_date in enumerate(test_dates, 1):
        test_dt_idx = SELF_ALL.index(test_date)
        train_self_dates = SELF_ALL[:test_dt_idx]
        log.info(f"\n=== STEP {step_idx}/9: train Self {len(train_self_dates)}d, test {test_date} ===")

        self_train_df = pd.concat([self_dfs[d] for d in train_self_dates], ignore_index=True)
        self_test_df = self_dfs[test_date].copy()
        train_df = pd.concat([tardis_train_df, self_train_df], ignore_index=True)
        val_df = tardis_val_df.copy()

        meds = train_df.reindex(columns=base_features).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        def mx(df):
            X = df.reindex(columns=base_features).copy().replace([np.inf, -np.inf], np.nan)
            return X.fillna(meds).fillna(0)
        Xt = mx(train_df); Xv = mx(val_df); Xst = mx(self_test_df)

        tm = train_df[dir_target].abs() > T
        vm = val_df[dir_target].abs() > T
        sm = self_test_df[dir_target].abs() > T
        Xt_f = Xt[tm].values; Xv_f = Xv[vm].values; Xst_f = Xst[sm].values
        y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
        y_dv = (val_df.loc[vm, dir_target] > 0).astype(int).values
        y_ds = (self_test_df.loc[sm, dir_target] > 0).astype(int).values

        step_results = {}
        for cfg in configs:
            try:
                clf = xgb_lib.XGBClassifier(**cfg["p"], **common)
                clf.fit(Xt_f, y_dt, eval_set=[(Xv_f, y_dv)], verbose=False)
                n_used = clf.best_iteration if hasattr(clf, "best_iteration") and clf.best_iteration else cfg["p"]["n_estimators"]
                a_tr = roc_auc_score(y_dt, clf.predict_proba(Xt_f)[:, 1])
                a_v = roc_auc_score(y_dv, clf.predict_proba(Xv_f)[:, 1])
                a_s = roc_auc_score(y_ds, clf.predict_proba(Xst_f)[:, 1]) if len(set(y_ds)) > 1 else float("nan")
                step_results[cfg["name"]] = {"auc_train": float(a_tr), "auc_val": float(a_v),
                                              "auc_self": float(a_s), "best_iter": int(n_used)}
                log.info(f"  {cfg['name']:<26} tr {a_tr:.3f}  val {a_v:.3f}  self {a_s:.3f}  iter {n_used}")
            except Exception as e:
                log.warning(f"  {cfg['name']}: {e}")
                step_results[cfg["name"]] = None

        walk_results.append({"step": step_idx, "test_date": test_date,
                             "n_train_self": len(train_self_dates),
                             "configs": step_results})

    # ---- Aggregate ----
    print()
    print("=" * 110)
    print("LARGER XGBOOST WALK-FORWARD (9 days)")
    print("=" * 110)
    print(f"\n{'Config':<26}", end="")
    for r in walk_results:
        print(f"{r['test_date'][-2:]:<6}", end="")
    print(f"  {'mean':<7}{'std':<7}{'pos':<6}")
    print("-" * 130)

    config_names = [c["name"] for c in configs]
    aggregates = {}
    for name in config_names:
        aucs = []
        line = f"{name:<26}"
        for r in walk_results:
            res = r["configs"].get(name)
            if res and not np.isnan(res["auc_self"]):
                aucs.append(res["auc_self"])
                line += f"{res['auc_self']:<6.3f}"
            else:
                line += f"{'-':<6}"
        if aucs:
            v = np.array(aucs)
            mean = v.mean(); std = v.std(); pos = (v > 0.55).sum()
            line += f"  {mean:<7.3f}{std:<7.3f}{pos:<6}"
            aggregates[name] = {"mean_auc_self": float(mean), "std_auc_self": float(std),
                                 "n_above_055": int(pos), "all_aucs": [float(x) for x in aucs]}
        print(line)

    # ---- Save ----
    out = {"approach": "Larger XGBoost sweep walk-forward", "steps": walk_results, "aggregates": aggregates}
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido32_larger_xgboost.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    # ---- Diagnosis ----
    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    if aggregates:
        best = max(aggregates.items(), key=lambda kv: kv[1]["mean_auc_self"])
        bn, bs = best
        baseline = aggregates.get("XGB n100 d5 (baseline)")
        print(f"\nBest config: {bn}")
        print(f"  mean Self AUC {bs['mean_auc_self']:.3f}, std {bs['std_auc_self']:.3f}, {bs['n_above_055']}/9 days > 0.55")
        if baseline:
            delta = bs["mean_auc_self"] - baseline["mean_auc_self"]
            print(f"\nvs baseline (n100 d5): Δ {delta:+.3f}")
            if delta >= 0.02:
                print(f"  → 더 큰 모델이 의미 있게 향상. 모델 capacity 부족했음.")
            elif delta >= 0.005:
                print(f"  → 미세 향상. 모델 capacity 약간 부족.")
            else:
                print(f"  → 더 큰 모델 효과 없음. 시장 본질 한계.")
    log.info("\n시도 32 complete")


if __name__ == "__main__":
    main()
