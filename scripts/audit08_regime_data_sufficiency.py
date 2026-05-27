"""Audit 08: Regime / data sufficiency 진단."""
import sys, logging, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.ml.data_prep import DATES_TRAIN, DATES_VAL, DATES_TEST, build_split, get_feature_columns

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
    log.info("Audit 08: Regime / Data Sufficiency")
    log.info("=" * 70)

    log.info("\nBuilding Tardis...")
    tardis_train = build_split(DATES_TRAIN, log)
    tardis_val = build_split(DATES_VAL, log)
    tardis_test = build_split(DATES_TEST, log)
    feat_pre = get_feature_columns(tardis_train)
    tt_clean = tardis_train.dropna(subset=["target_volatility_300s", "target_return_3600s"])
    medians = tt_clean.reindex(columns=feat_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    log.info("Building Self...")
    SELF_DATES = [f"2026-04-{d:02d}" for d in range(21, 31)] + ["2026-05-01"]
    self_dfs = []
    for d in SELF_DATES:
        try:
            df = build_self_date_dataset(d, log, train_medians=medians)
            self_dfs.append(df)
        except Exception as e:
            log.warning(f"  Self {d}: {e}")
    self_all = pd.concat(self_dfs, ignore_index=True) if self_dfs else pd.DataFrame()
    log.info(f"Tardis train {len(tardis_train)}, val {len(tardis_val)}, test {len(tardis_test)}, Self {len(self_all)}")

    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"
    for d in [tardis_train, tardis_val, tardis_test, self_all]:
        d.dropna(subset=[vol_target, dir_target], inplace=True)

    # ---- 1. Regime distributions ----
    print()
    print("=" * 80)
    print("[1] VOL DISTRIBUTION (target_volatility_300s)")
    print("=" * 80)
    print(f"\n{'Source':<14} {'N':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'q05':<10} {'q50':<10} {'q95':<10} {'Max':<10}")
    print("-" * 100)
    for label, df in [("Tardis train", tardis_train), ("Tardis val", tardis_val),
                       ("Tardis test", tardis_test), ("Self", self_all)]:
        v = df[vol_target]
        print(f"{label:<14} {len(v):<10} {v.mean():<10.4f} {v.std():<10.4f} {v.min():<10.4f} {v.quantile(0.05):<10.4f} {v.median():<10.4f} {v.quantile(0.95):<10.4f} {v.max():<10.4f}")

    print()
    print("=" * 80)
    print("[1] DIRECTION DISTRIBUTION (target_return_3600s)")
    print("=" * 80)
    print(f"\n{'Source':<14} {'N':<10} {'Mean':<10} {'Std':<10} {'q05':<10} {'q50':<10} {'q95':<10} {'%>0.2%':<10} {'%<-0.2%':<10}")
    print("-" * 100)
    for label, df in [("Tardis train", tardis_train), ("Tardis val", tardis_val),
                       ("Tardis test", tardis_test), ("Self", self_all)]:
        r = df[dir_target]
        pos_strong = (r > 0.20).mean() * 100
        neg_strong = (r < -0.20).mean() * 100
        print(f"{label:<14} {len(r):<10} {r.mean():<+10.4f} {r.std():<10.4f} {r.quantile(0.05):<+10.4f} {r.median():<+10.4f} {r.quantile(0.95):<+10.4f} {pos_strong:<10.2f} {neg_strong:<10.2f}")

    # ---- 2. Per-day regime characterization ----
    print()
    print("=" * 80)
    print("[2] PER-DAY REGIME (Self days)")
    print("=" * 80)
    print(f"\n{'Date':<14} {'Vol mean':<12} {'Dir mean':<12} {'%up':<10} {'%down':<10} {'Trend':<14}")
    print("-" * 80)
    if "_source_date" in self_all.columns:
        for date_str, group in self_all.groupby("_source_date"):
            v = group[vol_target].mean()
            r = group[dir_target]
            pct_up = (r > 0.20).mean() * 100
            pct_dn = (r < -0.20).mean() * 100
            # Trend = sum direction / sum |direction| (closer to ±1 = trend, near 0 = chop)
            trend = r.sum() / r.abs().sum() if r.abs().sum() > 0 else 0
            label = "uptrend" if trend > 0.3 else ("downtrend" if trend < -0.3 else "chop")
            print(f"{date_str:<14} {v:<12.4f} {r.mean():<+12.4f} {pct_up:<10.2f} {pct_dn:<10.2f} {trend:<+8.3f} {label:<6}")

    # ---- 3. Train-Test distribution shift (vol + selected features) ----
    print()
    print("=" * 80)
    print("[3] DISTRIBUTION SHIFT (Train vs Self test, KS test)")
    print("=" * 80)
    from scipy import stats
    shifts = {}
    feats_check = ["target_volatility_300s", "target_return_3600s"] + [c for c in feat_pre[:30]]
    print(f"\n{'Feature':<35} {'Train mean':<14} {'Self mean':<14} {'KS stat':<10} {'p-value':<10} {'shift?':<8}")
    print("-" * 100)
    for f in feats_check:
        if f not in tardis_train.columns or f not in self_all.columns: continue
        a = tardis_train[f].dropna()
        b = self_all[f].dropna()
        if len(a) < 100 or len(b) < 100: continue
        # Subsample for speed
        a_s = a.sample(min(10000, len(a)), random_state=42)
        b_s = b.sample(min(5000, len(b)), random_state=42)
        try:
            ks, p = stats.ks_2samp(a_s.values, b_s.values)
            shift_label = "BIG" if ks > 0.3 else ("MED" if ks > 0.1 else "small")
            shifts[f] = {"train_mean": float(a.mean()), "self_mean": float(b.mean()), "ks": float(ks), "p": float(p)}
            print(f"{f:<35} {a.mean():<+14.4f} {b.mean():<+14.4f} {ks:<10.3f} {p:<10.2e} {shift_label:<8}")
        except Exception as e:
            log.warning(f"  KS {f}: {e}")

    # Top shifted features (highest KS)
    print()
    big_shifts = sorted(shifts.items(), key=lambda kv: -kv[1]["ks"])[:10]
    print("Top 10 most-shifted features:")
    for f, s in big_shifts:
        print(f"  {f:<35} KS {s['ks']:.3f}  train {s['train_mean']:+.4f}  self {s['self_mean']:+.4f}")

    # ---- 4. Cumulative train size effect on Direction AUC ----
    print()
    print("=" * 80)
    print("[4] CUMULATIVE TRAIN SIZE → DIRECTION AUC")
    print("=" * 80)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    base_features = get_feature_columns(tardis_train)
    T = 0.20

    def train_eval_subset(subset_df, name):
        # filter direction
        sub_clean = subset_df[subset_df[dir_target].abs() > T]
        if len(sub_clean) < 100:
            return None
        meds = sub_clean.reindex(columns=base_features).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        def mx(df):
            X = df.reindex(columns=base_features).copy().replace([np.inf, -np.inf], np.nan)
            return X.fillna(meds).fillna(0)
        Xt = mx(sub_clean)
        y_t = (sub_clean[dir_target] > 0).astype(int).values
        # Self test eval
        self_test_df = self_all[self_all["_source_date"].isin(["2026-04-28", "2026-04-29", "2026-04-30"])]
        self_clean = self_test_df[self_test_df[dir_target].abs() > T]
        if len(self_clean) < 30:
            return None
        Xs = mx(self_clean)
        y_s = (self_clean[dir_target] > 0).astype(int).values

        sd = StandardScaler()
        try:
            X_t = sd.fit_transform(Xt); X_s = sd.transform(Xs)
        except Exception as e:
            log.warning(f"  {name}: scaler fail: {e}")
            return None
        clf = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        clf.fit(X_t, y_t)
        a_t = roc_auc_score(y_t, clf.predict_proba(X_t)[:, 1])
        a_s = roc_auc_score(y_s, clf.predict_proba(X_s)[:, 1]) if len(set(y_s)) > 1 else float("nan")
        return {"name": name, "n_train": len(sub_clean), "auc_train": a_t, "auc_self": a_s,
                "n_self": len(self_clean)}

    # Subsample DATES_TRAIN
    print(f"\n{'Subset':<24} {'N train':<10} {'AUC train':<12} {'AUC self test':<14}")
    print("-" * 70)
    cumulative_results = []
    for n_dates in [6, 12, 18, 24, 30, 36]:
        if n_dates > len(DATES_TRAIN):
            continue
        sub_dates = DATES_TRAIN[:n_dates]
        sub_df = tardis_train[tardis_train["_source_date"].isin(sub_dates)]
        r = train_eval_subset(sub_df, f"Tardis {n_dates}d")
        if r:
            cumulative_results.append(r)
            print(f"  {r['name']:<24} {r['n_train']:<10} {r['auc_train']:<12.3f} {r['auc_self']:<14.3f}")

    # Hybrid: Tardis full + Self train
    self_train_dates = ["2026-04-21", "2026-04-22", "2026-04-23", "2026-04-24", "2026-04-25", "2026-04-26", "2026-04-27"]
    self_train_df = self_all[self_all["_source_date"].isin(self_train_dates)]
    hybrid_df = pd.concat([tardis_train, self_train_df], ignore_index=True)
    r = train_eval_subset(hybrid_df, f"Hybrid Tardis36+Self7")
    if r:
        cumulative_results.append(r)
        print(f"  {r['name']:<24} {r['n_train']:<10} {r['auc_train']:<12.3f} {r['auc_self']:<14.3f}")

    # ---- 5. OOD detection (Self test rows in train distribution?) ----
    print()
    print("=" * 80)
    print("[5] OOD DETECTION (Self test vs Tardis train Mahalanobis-lite)")
    print("=" * 80)

    # Use a subset of features for OOD distance
    ood_features = [f for f in ["target_volatility_300s", "ob_obi_top1", "ob_obi_top5", "ob_obi_top50",
                                 "tr_total_volume_300s"] if f in tardis_train.columns and f in self_all.columns]
    if ood_features:
        train_sub = tardis_train[ood_features].dropna()
        self_sub = self_all[ood_features].dropna()
        # Standardize using train stats
        mu = train_sub.mean(); sigma = train_sub.std() + 1e-9
        train_z = ((train_sub - mu) / sigma).values
        self_z = ((self_sub - mu) / sigma).values
        # L2 distance to nearest train neighbor (subsample for speed)
        train_z_sub = train_z[np.random.RandomState(42).choice(len(train_z), min(5000, len(train_z)), replace=False)]
        # Per self row, distance to nearest train
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1).fit(train_z_sub)
        dists, _ = nn.kneighbors(self_z)
        # OOD threshold = 99th percentile of within-train distance
        nn_train = NearestNeighbors(n_neighbors=2).fit(train_z_sub)
        dists_train, _ = nn_train.kneighbors(train_z_sub)
        ood_thr = np.quantile(dists_train[:, 1], 0.99)
        ood_ratio = (dists.flatten() > ood_thr).mean()
        print(f"\n  Features used: {ood_features}")
        print(f"  Train internal q99 distance: {ood_thr:.3f}")
        print(f"  Self mean distance to nearest train: {dists.mean():.3f}")
        print(f"  Self max distance: {dists.max():.3f}")
        print(f"  Self OOD ratio (> q99 train internal): {ood_ratio*100:.1f}%")

    # ---- Save ----
    out = {
        "regime_summary": {
            "vol": {label: {"n": int(len(df)), "mean": float(df[vol_target].mean()),
                            "std": float(df[vol_target].std())}
                    for label, df in [("tardis_train", tardis_train),
                                       ("tardis_val", tardis_val),
                                       ("tardis_test", tardis_test),
                                       ("self", self_all)]},
        },
        "shifts_top10": [{"feature": f, **s} for f, s in big_shifts],
        "cumulative_results": cumulative_results,
        "ood_ratio_self": float(ood_ratio) if ood_features else None,
        "ood_features": ood_features,
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/audit08_regime_data_sufficiency.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    # ---- Diagnosis ----
    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    if cumulative_results:
        aucs = [r["auc_self"] for r in cumulative_results]
        last_aucs = aucs[-3:] if len(aucs) >= 3 else aucs
        # Saturation: change in last 3 < 0.01
        saturating = max(last_aucs) - min(last_aucs) < 0.015 if len(last_aucs) >= 2 else False
        rising = aucs[-1] > aucs[0] + 0.02 if len(aucs) >= 2 else False
        print(f"\nAUC vs train size curve:")
        for r in cumulative_results:
            print(f"  {r['name']:<24}  Self AUC {r['auc_self']:.3f}")
        print(f"\n  Saturating? {'YES' if saturating else 'NO'}")
        print(f"  Still rising? {'YES' if rising and not saturating else 'NO'}")

        if saturating:
            print(f"\n  → 더 많은 데이터로 ceiling 깰 가능성 LOW.")
            print(f"  → 다른 차원 권장 (timeframe / asset / strategy / MM)")
        elif rising:
            print(f"\n  → 더 많은 데이터로 ceiling 향상 가능. 1-2주 더 누적 후 재검증.")
        else:
            print(f"\n  → AUC 변동 비단조 — feature 의 noise 가능성")

    if ood_features:
        if ood_ratio < 0.05:
            print(f"\nOOD: Self test의 {ood_ratio*100:.1f}% only OOD → 분포 align, 모델 generalize 가능")
        elif ood_ratio < 0.20:
            print(f"\nOOD: Self test의 {ood_ratio*100:.1f}% OOD → 일부 새로운 regime, 추가 데이터 도움 가능")
        else:
            print(f"\nOOD: Self test의 {ood_ratio*100:.1f}% OOD → 큰 distribution shift, regime 다름")

    log.info("\nAudit 08 complete")


if __name__ == "__main__":
    main()
