"""시도 31: Time-weighted training (LR + XGB), walk-forward 9 days."""
import sys, logging, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from mark19.ml.data_prep import DATES_TRAIN, DATES_VAL, build_split, get_feature_columns
from live_bot.parquet_retry import read_parquet_with_retry

import importlib.util
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_bsd", _HERE / "backtest_self_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_self_date_dataset = _mod.build_self_date_dataset


def get_year_weight(date_str):
    """2022→0.1, 2023→0.3, 2024→0.5, 2025→0.7, 2026→1.0"""
    y = int(date_str.split("-")[0])
    return {2022: 0.1, 2023: 0.3, 2024: 0.5, 2025: 0.7, 2026: 1.0}.get(y, 0.5)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("시도 31: Time-Weighted Training (walk-forward 9 days)")
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
        log.info(f"  Self {d}: {len(df)} rows")

    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"
    tardis_train_df.dropna(subset=[vol_target, dir_target], inplace=True)
    tardis_val_df.dropna(subset=[vol_target, dir_target], inplace=True)

    base_features = get_feature_columns(tardis_train_df)
    log.info(f"\nFeatures: {len(base_features)}")

    # Tardis weights by year
    if "_source_date" in tardis_train_df.columns:
        tardis_train_df["_year_weight"] = tardis_train_df["_source_date"].astype(str).map(get_year_weight)
    else:
        tardis_train_df["_year_weight"] = 0.5
    log.info(f"\nTardis weight distribution: {tardis_train_df.groupby(tardis_train_df['_year_weight'])._year_weight.count().to_dict()}")

    test_dates = SELF_ALL[1:]  # 4/22..4/30 (9 days)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    DIR_TH = 0.55
    VOL_TH = 0.6
    LOCKOUT = 60; SL = 1.5
    FEE_TAKER, FEE_MAKER = 0.055, -0.025; MAX_HOLD = 30
    T = 0.20

    walk_results = []

    for step_idx, test_date in enumerate(test_dates, 1):
        test_dt_idx = SELF_ALL.index(test_date)
        train_self_dates = SELF_ALL[:test_dt_idx]
        log.info(f"\n=== STEP {step_idx}/9: train Self {len(train_self_dates)}d, test {test_date} ===")

        self_train_df = pd.concat([self_dfs[d] for d in train_self_dates], ignore_index=True)
        self_train_df["_year_weight"] = 1.0  # 2026
        self_test_df = self_dfs[test_date].copy()

        train_df = pd.concat([tardis_train_df, self_train_df], ignore_index=True)
        val_df = tardis_val_df.copy()

        meds = train_df.reindex(columns=base_features).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        def mx(df):
            X = df.reindex(columns=base_features).copy().replace([np.inf, -np.inf], np.nan)
            return X.fillna(meds).fillna(0)
        Xt = mx(train_df); Xv = mx(val_df); Xst = mx(self_test_df)

        # Direction filter
        tm = train_df[dir_target].abs() > T
        vm = val_df[dir_target].abs() > T
        sm = self_test_df[dir_target].abs() > T
        Xt_f = Xt[tm].values; Xv_f = Xv[vm].values; Xst_f = Xst[sm].values
        y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
        y_dv = (val_df.loc[vm, dir_target] > 0).astype(int).values
        y_ds = (self_test_df.loc[sm, dir_target] > 0).astype(int).values
        weights_dt = train_df.loc[tm, "_year_weight"].values

        # ---- LR unweighted ----
        sd_u = StandardScaler(); X_td_u = sd_u.fit_transform(Xt_f); X_sd_u = sd_u.transform(Xst_f); X_vd_u = sd_u.transform(Xv_f)
        lr_u = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lr_u.fit(X_td_u, y_dt)
        lr_u_self = roc_auc_score(y_ds, lr_u.predict_proba(X_sd_u)[:, 1]) if len(set(y_ds)) > 1 else float("nan")

        # ---- LR weighted ----
        lr_w = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lr_w.fit(X_td_u, y_dt, sample_weight=weights_dt)
        lr_w_self = roc_auc_score(y_ds, lr_w.predict_proba(X_sd_u)[:, 1]) if len(set(y_ds)) > 1 else float("nan")

        # ---- XGB unweighted ----
        xgb_u = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                    reg_alpha=0.1, reg_lambda=1.0,
                                    random_state=42, n_jobs=4, eval_metric="auc",
                                    early_stopping_rounds=20)
        xgb_u.fit(Xt_f, y_dt, eval_set=[(Xv_f, y_dv)], verbose=False)
        xgb_u_self = roc_auc_score(y_ds, xgb_u.predict_proba(Xst_f)[:, 1]) if len(set(y_ds)) > 1 else float("nan")

        # ---- XGB weighted ----
        xgb_w = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                    reg_alpha=0.1, reg_lambda=1.0,
                                    random_state=42, n_jobs=4, eval_metric="auc",
                                    early_stopping_rounds=20)
        xgb_w.fit(Xt_f, y_dt, sample_weight=weights_dt, eval_set=[(Xv_f, y_dv)], verbose=False)
        xgb_w_self = roc_auc_score(y_ds, xgb_w.predict_proba(Xst_f)[:, 1]) if len(set(y_ds)) > 1 else float("nan")

        log.info(f"  AUC self: LR_u {lr_u_self:.3f}  LR_w {lr_w_self:.3f}  XGB_u {xgb_u_self:.3f}  XGB_w {xgb_w_self:.3f}")

        walk_results.append({
            "step": step_idx, "test_date": test_date,
            "n_train_self": len(train_self_dates),
            "lr_unweighted_auc": float(lr_u_self), "lr_weighted_auc": float(lr_w_self),
            "xgb_unweighted_auc": float(xgb_u_self), "xgb_weighted_auc": float(xgb_w_self),
        })

    # ---- Aggregate ----
    print()
    print("=" * 100)
    print("TIME-WEIGHTED RESULTS (Walk-forward 9 days)")
    print("=" * 100)
    print(f"\n{'Step':<5} {'Date':<12} {'TrainD':<8} {'LR_u':<10} {'LR_w':<10} {'XGB_u':<10} {'XGB_w':<10}")
    print("-" * 75)
    for r in walk_results:
        print(f"{r['step']:<5} {r['test_date']:<12} {r['n_train_self']:<8} {r['lr_unweighted_auc']:<10.3f} {r['lr_weighted_auc']:<10.3f} {r['xgb_unweighted_auc']:<10.3f} {r['xgb_weighted_auc']:<10.3f}")

    print()
    for col_name, key in [("LR unweighted", "lr_unweighted_auc"),
                           ("LR weighted",   "lr_weighted_auc"),
                           ("XGB unweighted","xgb_unweighted_auc"),
                           ("XGB weighted",  "xgb_weighted_auc")]:
        vals = [r[key] for r in walk_results if not np.isnan(r[key])]
        if not vals: continue
        v = np.array(vals)
        print(f"  {col_name:<18}  mean {v.mean():.3f}  std {v.std():.3f}  min {v.min():.3f}  max {v.max():.3f}")

    # ---- Save ----
    out = {"approach": "Time-weighted training", "steps": walk_results}
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido31_time_weighted.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    # ---- Diagnosis ----
    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    lr_u_mean = np.mean([r["lr_unweighted_auc"] for r in walk_results])
    lr_w_mean = np.mean([r["lr_weighted_auc"] for r in walk_results])
    xgb_u_mean = np.mean([r["xgb_unweighted_auc"] for r in walk_results])
    xgb_w_mean = np.mean([r["xgb_weighted_auc"] for r in walk_results])
    print(f"\nLR  : unweighted {lr_u_mean:.3f} → weighted {lr_w_mean:.3f}  (Δ {lr_w_mean - lr_u_mean:+.3f})")
    print(f"XGB : unweighted {xgb_u_mean:.3f} → weighted {xgb_w_mean:.3f}  (Δ {xgb_w_mean - xgb_u_mean:+.3f})")

    if max(lr_w_mean, xgb_w_mean) > 0.55:
        print(f"\n  ✅ Time-weighting 효과: AUC > 0.55 가능")
    elif max(lr_w_mean - lr_u_mean, xgb_w_mean - xgb_u_mean) > 0.01:
        print(f"\n  🟡 Time-weighting 미세 향상")
    else:
        print(f"\n  ❌ Time-weighting 효과 없음")

    log.info("\n시도 31 complete")


if __name__ == "__main__":
    main()
