"""mark36_v2: early_stop 끄기, n_estimators=100 강제. dir_proba 분포 검증 + walk-forward."""
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
    log.info("mark36_v2: early_stop OFF, n_estimators=100 fixed")
    log.info("=" * 70)
    np.random.seed(42)

    log.info("\nBuilding Tardis...")
    tardis_train = build_split(DATES_TRAIN, log)
    tardis_val = build_split(DATES_VAL, log)
    feat_pre = get_feature_columns(tardis_train)
    tt_clean = tardis_train.dropna(subset=["target_volatility_300s", "target_return_3600s"])
    medians = tt_clean.reindex(columns=feat_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    SELF_ALL = [f"2026-04-{d:02d}" for d in range(21, 31)] + [f"2026-05-{d:02d}" for d in range(1, 10)]
    self_dfs = {}
    log.info("Building Self...")
    for d in SELF_ALL:
        try:
            df = build_self_date_dataset(d, log, train_medians=medians)
            df = df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
            if len(df) > 0:
                self_dfs[d] = df
        except Exception as e:
            log.warning(f"  Self {d}: {e}")

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
    log.info(f"\nFeature set: {len(feat_set)} (norm_only)")

    # ---- 1. Train mark36_v2 with full Self train data (4/21 ~ 5/9 minus val) ----
    log.info("\n[1] Train mark36_v2: early_stop OFF, n_estimators=100")
    self_all = pd.concat([self_dfs[d] for d in SELF_ALL if d in self_dfs], ignore_index=True)
    train_df = pd.concat([tardis_train, self_all], ignore_index=True)
    val_df = tardis_val
    log.info(f"  Train: {len(train_df)}  Val: {len(val_df)}")

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

    # Direction filter
    T = 0.20
    tm = train_df[dir_target].abs() > T
    vm = val_df[dir_target].abs() > T
    Xt_f = Xt[tm].values; Xv_f = Xv[vm].values
    y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
    y_dv = (val_df.loc[vm, dir_target] > 0).astype(int).values
    log.info(f"  Direction samples: train {len(y_dt)}  val {len(y_dv)}")
    log.info(f"  Class balance train: long {(y_dt==1).sum()} short {(y_dt==0).sum()}")

    # XGB n_estimators=100 NO early stop
    clf_v2 = xgb_lib.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.03,
                                     min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
                                     reg_alpha=1.0, reg_lambda=5.0,
                                     random_state=42, n_jobs=4, eval_metric="auc")
    clf_v2.fit(Xt_f, y_dt, verbose=False)  # NO eval_set, NO early stopping
    auc_train = roc_auc_score(y_dt, clf_v2.predict_proba(Xt_f)[:, 1])
    auc_val = roc_auc_score(y_dv, clf_v2.predict_proba(Xv_f)[:, 1])
    log.info(f"\n  mark36_v2: best_iter (n/a, fixed 100), Train AUC {auc_train:.3f}, Val AUC {auc_val:.3f}")

    # ---- 2. Compare v1 vs v2 dir_proba on Self test ----
    print()
    print("=" * 80)
    print("[2] dir_proba distribution: v1 vs v2 (Self days)")
    print("=" * 80)

    bundle_v1 = joblib.load("/Users/dohun/Desktop/Mark/mark19/models/mark36_v1.joblib")
    XGB_V1 = bundle_v1["xgb_dir"]
    feat_v1 = bundle_v1["feature_cols"]
    meds_v1 = pd.Series(bundle_v1["train_medians"])

    print(f"\n  v1 best_iter: {XGB_V1.best_iteration}  (collapse confirmed)")
    print(f"  v2 n_estimators: 100 (no early_stop)\n")

    print(f"{'Date':<14} {'N':<6} {'v1 <0.45':<10} {'v1 >0.55':<10} {'v2 <0.45':<10} {'v2 >0.55':<10}")
    print("-" * 70)
    period_results = {}
    for d in sorted(self_dfs.keys()):
        df = self_dfs[d]
        # v1
        X_v1 = df.reindex(columns=feat_v1).copy().replace([np.inf, -np.inf], np.nan).fillna(meds_v1).fillna(0)
        p1 = XGB_V1.predict_proba(X_v1.values)[:, 1]
        # v2
        X_v2 = df.reindex(columns=feat_set).copy().replace([np.inf, -np.inf], np.nan).fillna(meds).fillna(0)
        p2 = clf_v2.predict_proba(X_v2.values)[:, 1]
        period_results[d] = {
            "n": len(df),
            "v1_below_045": float((p1 < 0.45).mean()), "v1_above_055": float((p1 > 0.55).mean()),
            "v2_below_045": float((p2 < 0.45).mean()), "v2_above_055": float((p2 > 0.55).mean()),
            "v2_min": float(p2.min()), "v2_max": float(p2.max()),
            "v2_q25": float(np.quantile(p2, 0.25)), "v2_q50": float(np.quantile(p2, 0.5)),
            "v2_q75": float(np.quantile(p2, 0.75)),
        }
        r = period_results[d]
        print(f"{d:<14} {r['n']:<6} {r['v1_below_045']*100:<10.2f} {r['v1_above_055']*100:<10.2f} {r['v2_below_045']*100:<10.2f} {r['v2_above_055']*100:<10.2f}")

    # Aggregate v2 SHORT signal generation
    total_n = sum(r["n"] for r in period_results.values())
    v2_short_total = sum(r["v2_below_045"] * r["n"] for r in period_results.values()) / total_n
    v2_long_total = sum(r["v2_above_055"] * r["n"] for r in period_results.values()) / total_n
    print(f"\n  Aggregate (19 days): v2 <0.45 {v2_short_total*100:.2f}%, >0.55 {v2_long_total*100:.2f}%")

    # ---- 3. v2 dir_proba range on Self overall ----
    print()
    print("=" * 80)
    print("[3] v2 dir_proba range (Self all)")
    print("=" * 80)
    self_all_p2 = []
    for d in self_dfs:
        df = self_dfs[d]
        X_v2 = df.reindex(columns=feat_set).copy().replace([np.inf, -np.inf], np.nan).fillna(meds).fillna(0)
        p2 = clf_v2.predict_proba(X_v2.values)[:, 1]
        self_all_p2.append(p2)
    self_all_p2 = np.concatenate(self_all_p2)
    print(f"  N: {len(self_all_p2)}")
    print(f"  min {self_all_p2.min():.4f}  q05 {np.quantile(self_all_p2, 0.05):.4f}  q25 {np.quantile(self_all_p2, 0.25):.4f}  q50 {np.quantile(self_all_p2, 0.5):.4f}  q75 {np.quantile(self_all_p2, 0.75):.4f}  q95 {np.quantile(self_all_p2, 0.95):.4f}  max {self_all_p2.max():.4f}")
    print(f"  v1 (for comparison):")
    self_all_p1 = []
    for d in self_dfs:
        df = self_dfs[d]
        X_v1 = df.reindex(columns=feat_v1).copy().replace([np.inf, -np.inf], np.nan).fillna(meds_v1).fillna(0)
        p1 = XGB_V1.predict_proba(X_v1.values)[:, 1]
        self_all_p1.append(p1)
    self_all_p1 = np.concatenate(self_all_p1)
    print(f"  v1 N: {len(self_all_p1)}")
    print(f"  v1 min {self_all_p1.min():.4f}  q05 {np.quantile(self_all_p1, 0.05):.4f}  q25 {np.quantile(self_all_p1, 0.25):.4f}  q50 {np.quantile(self_all_p1, 0.5):.4f}  q75 {np.quantile(self_all_p1, 0.75):.4f}  q95 {np.quantile(self_all_p1, 0.95):.4f}  max {self_all_p1.max():.4f}")

    # ---- 4. Walk-forward 9 days (same as sido36) for AUC measurement ----
    print()
    print("=" * 80)
    print("[4] Walk-forward 9 days with mark36_v2 config (no early_stop)")
    print("=" * 80)
    SELF_BASE_DATES = [f"2026-04-{d:02d}" for d in range(21, 31)]
    test_dates = SELF_BASE_DATES[1:]
    walk_results = []
    for step_idx, test_date in enumerate(test_dates, 1):
        idx = SELF_BASE_DATES.index(test_date)
        train_self_dates = SELF_BASE_DATES[:idx]
        log.info(f"  STEP {step_idx}/9 test {test_date}")
        self_train_df = pd.concat([self_dfs[d] for d in train_self_dates if d in self_dfs], ignore_index=True)
        self_test_df = self_dfs.get(test_date)
        if self_test_df is None: continue
        train_step = pd.concat([tardis_train, self_train_df], ignore_index=True)

        meds_step = train_step.reindex(columns=feat_set).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        def mxs(df):
            X = df.reindex(columns=feat_set).copy().replace([np.inf, -np.inf], np.nan)
            return X.fillna(meds_step).fillna(0)
        tm = train_step[dir_target].abs() > T
        sm = self_test_df[dir_target].abs() > T
        Xt = mxs(train_step)[tm].values
        Xst = mxs(self_test_df)[sm].values
        y_t = (train_step.loc[tm, dir_target] > 0).astype(int).values
        y_s = (self_test_df.loc[sm, dir_target] > 0).astype(int).values

        clf = xgb_lib.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.03,
                                      min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
                                      reg_alpha=1.0, reg_lambda=5.0,
                                      random_state=42, n_jobs=4, eval_metric="auc")
        clf.fit(Xt, y_t, verbose=False)
        auc = roc_auc_score(y_s, clf.predict_proba(Xst)[:, 1]) if len(set(y_s)) > 1 else float("nan")
        # SHORT signal rate
        all_p = clf.predict_proba(mxs(self_test_df).values)[:, 1]
        below_045 = float((all_p < 0.45).mean())
        above_055 = float((all_p > 0.55).mean())
        walk_results.append({"step": step_idx, "test_date": test_date, "auc": float(auc),
                              "below_045": below_045, "above_055": above_055})
        log.info(f"    AUC self {auc:.3f}  <0.45 {below_045*100:.1f}%  >0.55 {above_055*100:.1f}%")

    aucs = [r["auc"] for r in walk_results if not np.isnan(r["auc"])]
    a = np.array(aucs)
    print(f"\n  Walk-forward (no early_stop): mean AUC {a.mean():.3f}  std {a.std():.3f}  >0.55 {(a>0.55).sum()}/{len(a)}")
    print(f"  Avg <0.45 rate: {np.mean([r['below_045'] for r in walk_results])*100:.2f}%")
    print(f"  Avg >0.55 rate: {np.mean([r['above_055'] for r in walk_results])*100:.2f}%")

    # ---- 5. Save mark36_v2 ----
    out = {
        "lr_vol": lrv, "scaler_vol": sv,
        "xgb_dir": clf_v2,
        "feature_cols": feat_set,
        "high_shift_features": HIGH_SHIFT_FEATURES,
        "norm_features": norm_features,
        "train_medians": meds.to_dict(),
        "vol_med": vol_med, "T": T,
        "metadata": {
            "approach": "mark36_v2 — norm_only XGB n=100 fixed, NO early_stop",
            "n_train": int(len(y_dt)),
            "n_val": int(len(y_dv)),
            "auc_train": float(auc_train),
            "auc_val": float(auc_val),
            "v1_below_045_aggregate": float(np.mean([r["v1_below_045"] for r in period_results.values()])),
            "v2_below_045_aggregate": float(v2_short_total),
            "walk_forward_auc_mean": float(a.mean()) if len(aucs) else None,
            "walk_forward_auc_std": float(a.std()) if len(aucs) else None,
            "walk_forward_above_055_count": int((a > 0.55).sum()) if len(aucs) else 0,
        },
    }
    model_path = Path("/Users/dohun/Desktop/Mark/mark19/models/mark36_v2.joblib")
    joblib.dump(out, model_path, compress=3)
    log.info(f"\nSaved: {model_path}")

    json_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/mark36_v2.json")
    with open(json_path, "w") as f:
        json.dump({"metadata": out["metadata"],
                   "period_results": period_results,
                   "walk_forward": walk_results},
                  f, indent=2, default=str)
    log.info(f"JSON: {json_path}")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    if v2_short_total >= 0.05:
        print(f"\n  ✅ mark36_v2 generates SHORT signals: {v2_short_total*100:.2f}% (vs v1 ~0.0%)")
    elif v2_short_total >= 0.01:
        print(f"\n  🟡 mark36_v2 SHORT signals weak: {v2_short_total*100:.2f}%")
    else:
        print(f"\n  ❌ mark36_v2 still SHORT-zero: {v2_short_total*100:.2f}%")
        print(f"     Even with n_estimators=100, model bias persists (Case C: train imbalance)")

    if len(aucs) > 0 and a.mean() >= 0.55:
        print(f"\n  Walk-forward AUC {a.mean():.3f} (≥0.55, valid)")
    elif len(aucs) > 0:
        print(f"\n  Walk-forward AUC {a.mean():.3f} (≤0.55, weak)")

    log.info("\nmark36_v2 complete")


if __name__ == "__main__":
    main()
