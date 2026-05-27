"""시도 36: Regime-invariant features (day-mean normalized) + XGB n1000 d6."""
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


# Features that audit08 identified as BIG-shift between Tardis and Self
HIGH_SHIFT_FEATURES = [
    "ob_total_depth_1", "ob_total_depth_5", "ob_total_depth_10", "ob_total_depth_50",
    "ob_bid_depth_1", "ob_bid_depth_5", "ob_bid_depth_10", "ob_bid_depth_50",
    "ob_ask_depth_1", "ob_ask_depth_5", "ob_ask_depth_10", "ob_ask_depth_50",
    "ob_spread", "ob_spread_pct",
    "ob_bid_slope_10", "ob_ask_slope_10",
    "tr_total_count", "tr_buy_count", "tr_sell_count",
    "tr_total_volume", "tr_buy_volume", "tr_sell_volume",
    "tr_trades_per_sec_300s", "tr_total_volume_300s",
]


def add_normalized_features(df, log):
    """For each feature in HIGH_SHIFT_FEATURES, add `<feat>_norm` = feature / day_mean.
    Day-by-day normalization makes features comparable across regime/structural changes."""
    if "_source_date" not in df.columns:
        log.warning("  no _source_date, skip normalization")
        return df
    df = df.copy()
    added = []
    for feat in HIGH_SHIFT_FEATURES:
        if feat not in df.columns: continue
        # Day mean
        day_mean = df.groupby("_source_date")[feat].transform("mean")
        new_col = f"{feat}_norm"
        df[new_col] = (df[feat] / day_mean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        added.append(new_col)
    log.info(f"  Added {len(added)} normalized features")
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("시도 36: Regime-Invariant Features (day-mean normalized) + XGB n1000 d6")
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

    log.info("\nAdding normalized features...")
    log.info("  Tardis train")
    tardis_train_df = add_normalized_features(tardis_train_df, log)
    log.info("  Tardis val")
    tardis_val_df = add_normalized_features(tardis_val_df, log)
    log.info("  Self days")
    for d in self_dfs:
        self_dfs[d] = add_normalized_features(self_dfs[d], log)

    norm_features = [c for c in tardis_train_df.columns if c.endswith("_norm")]
    log.info(f"\nNormalized features added: {len(norm_features)}")

    # ---- KS test: shift before/after normalization ----
    print()
    print("=" * 80)
    print("DISTRIBUTION SHIFT (Tardis train vs Self all): RAW vs NORM")
    print("=" * 80)
    self_all = pd.concat(list(self_dfs.values()), ignore_index=True)
    from scipy import stats as scistats
    shift_compare = []
    print(f"\n{'Feature':<32} {'Raw KS':<10} {'Norm KS':<10} {'Δ':<10}")
    print("-" * 70)
    for raw in HIGH_SHIFT_FEATURES:
        norm = f"{raw}_norm"
        if raw not in tardis_train_df.columns or raw not in self_all.columns: continue
        if norm not in tardis_train_df.columns: continue
        # Raw KS
        a_raw = tardis_train_df[raw].dropna().sample(min(10000, len(tardis_train_df)), random_state=42)
        b_raw = self_all[raw].dropna().sample(min(5000, len(self_all)), random_state=42)
        ks_raw, _ = scistats.ks_2samp(a_raw.values, b_raw.values)
        # Norm KS
        a_n = tardis_train_df[norm].dropna()
        b_n = self_all[norm].dropna()
        if len(a_n) < 100 or len(b_n) < 100: continue
        a_ns = a_n.sample(min(10000, len(a_n)), random_state=42)
        b_ns = b_n.sample(min(5000, len(b_n)), random_state=42)
        ks_norm, _ = scistats.ks_2samp(a_ns.values, b_ns.values)
        shift_compare.append({"feature": raw, "ks_raw": float(ks_raw), "ks_norm": float(ks_norm)})
        print(f"{raw:<32} {ks_raw:<10.3f} {ks_norm:<10.3f} {ks_norm - ks_raw:<+10.3f}")

    avg_raw_ks = np.mean([s["ks_raw"] for s in shift_compare])
    avg_norm_ks = np.mean([s["ks_norm"] for s in shift_compare])
    log.info(f"\nAvg raw KS: {avg_raw_ks:.3f}, avg norm KS: {avg_norm_ks:.3f}, Δ {avg_norm_ks - avg_raw_ks:+.3f}")

    # ---- Walk-forward 9 days ----
    print()
    print("=" * 80)
    print("WALK-FORWARD 9 DAYS: XGB n1000 d6 mcw100 with regime-invariant features")
    print("=" * 80)

    # Build base features list (canonical + norm)
    canonical = get_feature_columns(tardis_train_df)
    feature_set_full = canonical + norm_features  # raw + norm
    feature_set_norm_only = [f for f in canonical if f not in HIGH_SHIFT_FEATURES] + norm_features
    log.info(f"\nFeature sets:")
    log.info(f"  full (canonical + norm): {len(feature_set_full)}")
    log.info(f"  norm_only (raw replaced): {len(feature_set_norm_only)}")
    log.info(f"  raw only (시도 32 baseline): {len(canonical)}")

    test_dates = SELF_ALL[1:]
    T = 0.20

    from sklearn.metrics import roc_auc_score
    import xgboost as xgb_lib

    cfg_p = dict(n_estimators=1000, max_depth=6, learning_rate=0.03,
                 min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
                 reg_alpha=1.0, reg_lambda=5.0,
                 random_state=42, n_jobs=4, eval_metric="auc",
                 early_stopping_rounds=30)

    walk_results = []
    for step_idx, test_date in enumerate(test_dates, 1):
        test_dt_idx = SELF_ALL.index(test_date)
        train_self_dates = SELF_ALL[:test_dt_idx]
        log.info(f"\n=== STEP {step_idx}/9: train Self {len(train_self_dates)}d, test {test_date} ===")

        self_train_df = pd.concat([self_dfs[d] for d in train_self_dates], ignore_index=True)
        self_test_df = self_dfs[test_date].copy()
        train_df = pd.concat([tardis_train_df, self_train_df], ignore_index=True)
        val_df = tardis_val_df.copy()

        step_results = {}
        for feat_set, feat_name in [(canonical, "raw"), (feature_set_full, "raw+norm"), (feature_set_norm_only, "norm_only")]:
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

            clf = xgb_lib.XGBClassifier(**cfg_p)
            clf.fit(Xt_f, y_dt, eval_set=[(Xv_f, y_dv)], verbose=False)
            n_used = clf.best_iteration if hasattr(clf, "best_iteration") and clf.best_iteration else 1000
            a_tr = roc_auc_score(y_dt, clf.predict_proba(Xt_f)[:, 1])
            a_v = roc_auc_score(y_dv, clf.predict_proba(Xv_f)[:, 1])
            a_s = roc_auc_score(y_ds, clf.predict_proba(Xst_f)[:, 1]) if len(set(y_ds)) > 1 else float("nan")
            step_results[feat_name] = {"auc_train": float(a_tr), "auc_val": float(a_v),
                                        "auc_self": float(a_s), "best_iter": int(n_used),
                                        "n_features": len(feat_set)}
            log.info(f"  {feat_name:<10} (n_feat {len(feat_set)})  tr {a_tr:.3f}  val {a_v:.3f}  self {a_s:.3f}  iter {n_used}")

        walk_results.append({"step": step_idx, "test_date": test_date,
                             "n_train_self": len(train_self_dates),
                             "configs": step_results})

    # Aggregate
    print()
    print("=" * 100)
    print("WALK-FORWARD COMPARISON: raw vs raw+norm vs norm_only")
    print("=" * 100)
    print(f"\n{'Step':<6} {'Date':<14}", end="")
    for n in ("raw", "raw+norm", "norm_only"):
        print(f"{n:<12}", end="")
    print()
    print("-" * 60)
    for r in walk_results:
        print(f"{r['step']:<6} {r['test_date']:<14}", end="")
        for n in ("raw", "raw+norm", "norm_only"):
            v = r["configs"].get(n, {}).get("auc_self", float("nan"))
            print(f"{v:<12.3f}", end="")
        print()

    print("\n  Aggregate:")
    aggregates = {}
    for n in ("raw", "raw+norm", "norm_only"):
        vals = [r["configs"].get(n, {}).get("auc_self", float("nan")) for r in walk_results]
        v = np.array([x for x in vals if not np.isnan(x)])
        if len(v) > 0:
            aggregates[n] = {"mean": float(v.mean()), "std": float(v.std()),
                             "above_055": int((v > 0.55).sum()), "n": len(v)}
            print(f"    {n:<12}  mean {v.mean():.3f}  std {v.std():.3f}  >0.55 {(v > 0.55).sum()}/{len(v)}")

    # Save
    out = {
        "approach": "Regime-invariant (day-mean normalized) features + XGB n1000 d6",
        "shift_compare": shift_compare,
        "avg_raw_ks": float(avg_raw_ks), "avg_norm_ks": float(avg_norm_ks),
        "ks_reduction": float(avg_raw_ks - avg_norm_ks),
        "steps": walk_results,
        "aggregates": aggregates,
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido36_regime_invariant.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    # Diagnosis
    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print(f"\n  Avg KS shift: raw {avg_raw_ks:.3f} → norm {avg_norm_ks:.3f}  (Δ {avg_norm_ks - avg_raw_ks:+.3f})")
    if "raw" in aggregates and "raw+norm" in aggregates and "norm_only" in aggregates:
        d_full = aggregates["raw+norm"]["mean"] - aggregates["raw"]["mean"]
        d_only = aggregates["norm_only"]["mean"] - aggregates["raw"]["mean"]
        best = max(aggregates.items(), key=lambda kv: kv[1]["mean"])
        print(f"  Self AUC: raw {aggregates['raw']['mean']:.3f} → raw+norm {aggregates['raw+norm']['mean']:.3f} (Δ {d_full:+.3f})")
        print(f"            raw {aggregates['raw']['mean']:.3f} → norm_only {aggregates['norm_only']['mean']:.3f} (Δ {d_only:+.3f})")
        print(f"\n  Best: {best[0]} mean {best[1]['mean']:.3f}, std {best[1]['std']:.3f}, {best[1]['above_055']}/{best[1]['n']} >0.55")
        if best[1]["mean"] >= 0.58:
            print(f"\n  ✅ Regime-invariant features 효과적 (≥ 0.58)")
        elif best[1]["mean"] >= 0.555:
            print(f"\n  🟡 약간 향상, ceiling 부근")
        else:
            print(f"\n  ❌ Regime-invariant 효과 미미. 본질 한계.")

    log.info("\n시도 36 complete")


if __name__ == "__main__":
    main()
