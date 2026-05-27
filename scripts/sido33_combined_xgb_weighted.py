"""시도 33: XGB n1000 d6 mcw100 + time-weighted (sido31 + sido32 best)."""
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


def get_year_weight(date_str):
    y = int(str(date_str).split("-")[0])
    return {2022: 0.1, 2023: 0.3, 2024: 0.5, 2025: 0.7, 2026: 1.0}.get(y, 0.5)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("시도 33: XGB n1000 d6 mcw100 + time-weighted")
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

    if "_source_date" in tardis_train_df.columns:
        tardis_train_df["_year_weight"] = tardis_train_df["_source_date"].astype(str).map(get_year_weight)
    else:
        tardis_train_df["_year_weight"] = 0.5

    test_dates = SELF_ALL[1:]
    T = 0.20

    from sklearn.metrics import roc_auc_score
    import xgboost as xgb_lib

    walk_results = []
    cfg_p = dict(n_estimators=1000, max_depth=6, learning_rate=0.03,
                 min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
                 reg_alpha=1.0, reg_lambda=5.0,
                 random_state=42, n_jobs=4, eval_metric="auc",
                 early_stopping_rounds=30)

    for step_idx, test_date in enumerate(test_dates, 1):
        test_dt_idx = SELF_ALL.index(test_date)
        train_self_dates = SELF_ALL[:test_dt_idx]
        log.info(f"\n=== STEP {step_idx}/9: train Self {len(train_self_dates)}d, test {test_date} ===")

        self_train_df = pd.concat([self_dfs[d] for d in train_self_dates], ignore_index=True)
        self_train_df["_year_weight"] = 1.0
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
        weights_dt = train_df.loc[tm, "_year_weight"].values

        # Unweighted (sido32 baseline)
        clf_u = xgb_lib.XGBClassifier(**cfg_p)
        clf_u.fit(Xt_f, y_dt, eval_set=[(Xv_f, y_dv)], verbose=False)
        a_self_u = roc_auc_score(y_ds, clf_u.predict_proba(Xst_f)[:, 1]) if len(set(y_ds)) > 1 else float("nan")

        # Weighted
        clf_w = xgb_lib.XGBClassifier(**cfg_p)
        clf_w.fit(Xt_f, y_dt, sample_weight=weights_dt, eval_set=[(Xv_f, y_dv)], verbose=False)
        a_self_w = roc_auc_score(y_ds, clf_w.predict_proba(Xst_f)[:, 1]) if len(set(y_ds)) > 1 else float("nan")

        log.info(f"  AUC self: unweighted {a_self_u:.3f}  weighted {a_self_w:.3f}  (Δ {a_self_w - a_self_u:+.3f})")
        walk_results.append({
            "step": step_idx, "test_date": test_date,
            "auc_unweighted": float(a_self_u), "auc_weighted": float(a_self_w),
        })

    # Aggregate
    print()
    print("=" * 90)
    print("XGB n1000 d6 mcw100 — UNWEIGHTED vs WEIGHTED (9-day walk-forward)")
    print("=" * 90)
    print(f"\n{'Step':<6} {'Date':<14} {'unweighted':<12} {'weighted':<12} {'Δ':<10}")
    print("-" * 60)
    for r in walk_results:
        print(f"{r['step']:<6} {r['test_date']:<14} {r['auc_unweighted']:<12.3f} {r['auc_weighted']:<12.3f} {r['auc_weighted'] - r['auc_unweighted']:<+10.3f}")

    u = np.array([r["auc_unweighted"] for r in walk_results])
    w = np.array([r["auc_weighted"] for r in walk_results])
    print("\n  Unweighted: mean {:.3f}, std {:.3f}, >0.55 {}/9".format(u.mean(), u.std(), (u > 0.55).sum()))
    print("  Weighted  : mean {:.3f}, std {:.3f}, >0.55 {}/9".format(w.mean(), w.std(), (w > 0.55).sum()))

    out = {"approach": "XGB n1000 d6 mcw100 + time-weighted", "steps": walk_results,
           "summary": {"unweighted_mean": float(u.mean()), "weighted_mean": float(w.mean()),
                       "delta": float(w.mean() - u.mean())}}
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido33_combined.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    delta = w.mean() - u.mean()
    print(f"\n  XGB n100 baseline: 0.520")
    print(f"  XGB n1000 unweighted (시도 32): {u.mean():.3f} (Δ +{u.mean()-0.520:+.3f} vs baseline)")
    print(f"  XGB n1000 weighted (시도 33): {w.mean():.3f} (Δ {delta:+.3f} vs unweighted)")
    if w.mean() >= 0.58:
        print(f"\n  ✅ AUC ≥ 0.58 달성. 조합 효과 strong.")
    elif w.mean() >= 0.55 and delta > 0.005:
        print(f"\n  🟡 약간 향상, ceiling 부근")
    else:
        print(f"\n  ❌ 효과 미미")
    log.info("\n시도 33 complete")


if __name__ == "__main__":
    main()
