"""Audit: Lookahead bias in add_normalized_features (sido36).

Replicates sido36v2_backtest.py walk-forward EXACTLY but with 3 norm modes
to isolate the impact of the lookahead leak in `groupby.transform("mean")`.

Modes:
  LEAKY:    full-day mean per day (current sido36 behavior — peeks future)
            → reference: sido36v2_backtest.json says +1.074%/day, 7/9
  CAUSAL:   yesterday-norm for Self days, leaky for Tardis (Tardis is train-only,
            leak doesn't matter for offline-known train data)
            → measures real OOS PnL when test inference is causal AND train uses causal
  MISMATCH: train with LEAKY (mark36_v2 style) + test with CAUSAL yesterday-norm.
            Simulates SHADOW v2 deployment: model trained on day-mean features,
            inference uses causal substitute → measures distribution-mismatch cost.

Decision:
  CAUSAL ≈ LEAKY → leak harmless, mark36_v3 is fine
  CAUSAL << LEAKY → leak inflated mark36_v2 results
  MISMATCH << LEAKY → SHADOW v2 underperforms its training
"""
import sys, logging, json, importlib.util
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.ml.data_prep import DATES_TRAIN, DATES_VAL, build_split, get_feature_columns

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_bsd", _HERE / "backtest_self_data.py")
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
build_self_date_dataset = _mod.build_self_date_dataset

_spec36 = importlib.util.spec_from_file_location("_sido36", _HERE / "sido36_regime_invariant.py")
_mod36 = importlib.util.module_from_spec(_spec36); _spec36.loader.exec_module(_mod36)
HIGH_SHIFT_FEATURES = _mod36.HIGH_SHIFT_FEATURES
add_normalized_features_LEAKY = _mod36.add_normalized_features


# ---- Causal norm functions ----

def add_normalized_features_yesterday(df, prev_day_means, log):
    """Yesterday-norm: each row in day D divided by D-1's full-day mean.

    prev_day_means: dict[date_str → dict[feat → mean]]. Caller computes once.
    For days without prev (e.g., earliest day), falls back to that day's mean (leaky).
    """
    if "_source_date" not in df.columns:
        log.warning("  no _source_date; skip norm")
        return df
    df = df.copy()
    added = 0
    for feat in HIGH_SHIFT_FEATURES:
        if feat not in df.columns: continue
        norm_col = f"{feat}_norm"
        df[norm_col] = np.nan
        for d in df["_source_date"].unique():
            mask = df["_source_date"] == d
            ymean = prev_day_means.get(d, {}).get(feat)
            if ymean is None or pd.isna(ymean) or ymean == 0:
                # Fallback: use this day's mean (leaky — only happens for earliest day)
                ymean = df.loc[mask, feat].mean()
            if pd.isna(ymean) or ymean == 0:
                df.loc[mask, norm_col] = 0.0
            else:
                df.loc[mask, norm_col] = df.loc[mask, feat] / ymean
        df[norm_col] = df[norm_col].replace([np.inf, -np.inf], np.nan)
        added += 1
    log.info(f"  Added {added} yesterday-norm features (fallback-leaky days: {sum(1 for d in df['_source_date'].unique() if d not in prev_day_means)})")
    return df


def compute_day_means(self_dfs, log):
    """Compute per-day means for HIGH_SHIFT_FEATURES across self_dfs.
    Returns dict[date_str → dict[feat → mean]]."""
    means = {}
    for d, df in self_dfs.items():
        means[d] = {}
        for feat in HIGH_SHIFT_FEATURES:
            if feat in df.columns:
                means[d][feat] = float(df[feat].mean())
    return means


def yesterday_means_for_self(self_dfs, log):
    """For each self day, find its 'yesterday' (D-1 if available in self_dfs).
    Returns dict[D → dict[feat → D-1's mean]]."""
    sorted_days = sorted(self_dfs.keys())
    day_means = compute_day_means(self_dfs, log)
    result = {}
    for d in sorted_days:
        d_dt = pd.to_datetime(d).date()
        prev_dt = d_dt - pd.Timedelta(days=1)
        prev_str = prev_dt.strftime("%Y-%m-%d")
        if prev_str in day_means:
            result[d] = day_means[prev_str]
        # else: caller falls back to current-day mean (leaky for that one day)
    return result


# ---- Backtest core (copied from sido36v2 but parameterized by norm mode) ----

def run_walk_forward(mode, tardis_train, tardis_val, self_dfs, sorted_self,
                     test_dates, prev_day_means_self, log):
    """mode ∈ {LEAKY, CAUSAL, MISMATCH}."""
    log.info(f"\n{'='*70}\n=== WALK-FORWARD mode={mode} ===\n{'='*70}")

    # Apply norm to TRAIN/VAL/Self according to mode
    # LEAKY: leaky everywhere (sido36 baseline)
    # CAUSAL: leaky for Tardis (offline train data — irrelevant), yesterday-norm for Self
    # MISMATCH: leaky everywhere for TRAIN, but TEST self_day uses yesterday-norm
    tardis_train_n = add_normalized_features_LEAKY(tardis_train.copy(), log)
    tardis_val_n = add_normalized_features_LEAKY(tardis_val.copy(), log)

    self_train_n = {}    # how Self days are normalized when used as TRAIN
    self_test_n = {}     # how Self days are normalized when used as TEST
    for d, df in self_dfs.items():
        if mode == "LEAKY":
            self_train_n[d] = add_normalized_features_LEAKY(df.copy(), log)
            self_test_n[d] = self_train_n[d]
        elif mode == "CAUSAL":
            n = add_normalized_features_yesterday(df.copy(), prev_day_means_self, log)
            self_train_n[d] = n
            self_test_n[d] = n
        elif mode == "MISMATCH":
            # TRAIN: leaky; TEST: causal
            self_train_n[d] = add_normalized_features_LEAKY(df.copy(), log)
            self_test_n[d] = add_normalized_features_yesterday(df.copy(), prev_day_means_self, log)
        else:
            raise ValueError(mode)

    norm_features = [c for c in tardis_train_n.columns if c.endswith("_norm")]
    canonical = get_feature_columns(tardis_train_n)
    feat_set = [f for f in canonical if f not in HIGH_SHIFT_FEATURES] + norm_features
    log.info(f"  feat_set: {len(feat_set)}")

    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
    T = 0.20

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb_lib

    DIR_THS = [0.50, 0.55, 0.58]
    VOL_TH = 0.6
    LOCKOUT = 60; SL = 1.5
    FEE_TAKER, FEE_MAKER = 0.055, -0.025; MAX_HOLD = 30

    walk_results = []
    for step_idx, test_date in enumerate(test_dates, 1):
        idx = sorted_self.index(test_date)
        train_self_dates = sorted_self[:idx]
        log.info(f"  STEP {step_idx}/{len(test_dates)} test={test_date} train_self_n={len(train_self_dates)}")

        # Train uses self_train_n versions
        self_train_df = pd.concat([self_train_n[d] for d in train_self_dates], ignore_index=True)
        # Test uses self_test_n version
        self_test_df = self_test_n[test_date].copy()
        train_df = pd.concat([tardis_train_n, self_train_df], ignore_index=True)

        meds = train_df.reindex(columns=feat_set).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        def mx(df_):
            X = df_.reindex(columns=feat_set).copy().replace([np.inf, -np.inf], np.nan)
            return X.fillna(meds).fillna(0)
        Xt = mx(train_df); Xst = mx(self_test_df)

        # Vol LR (uses leaky Tardis val medians for vol_med target — same as sido36v2)
        vol_med = float(train_df[vol_target].median())
        y_vt = (train_df[vol_target] > vol_med).astype(int).values
        sv = StandardScaler(); X_tv = sv.fit_transform(Xt)
        lrv = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lrv.fit(X_tv, y_vt)

        # Direction
        tm = train_df[dir_target].abs() > T
        sm = self_test_df[dir_target].abs() > T
        Xt_f = Xt[tm].values; Xst_f = Xst[sm].values
        y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
        y_ds = (self_test_df.loc[sm, dir_target] > 0).astype(int).values
        if len(set(y_dt)) < 2 or len(set(y_ds)) < 2:
            log.warning(f"    skip: single-class targets"); continue

        clf = xgb_lib.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.03,
                                      min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
                                      reg_alpha=1.0, reg_lambda=5.0,
                                      random_state=42, n_jobs=4, eval_metric="auc")
        clf.fit(Xt_f, y_dt, verbose=False)
        auc_self = roc_auc_score(y_ds, clf.predict_proba(Xst_f)[:, 1])

        # Backtest
        bt = self_test_df.copy().reset_index(drop=True)
        Xbt = mx(bt)
        bt["vol_proba"] = lrv.predict_proba(sv.transform(Xbt))[:, 1]
        bt["dir_proba"] = clf.predict_proba(Xbt.values)[:, 1]
        bt["actual_return"] = bt[dir_target].values
        ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in bt.columns), None)
        price_col = next((c for c in ["ob_mid_price", "mid"] if c in bt.columns), None)
        bt = bt.sort_values(ts_col).reset_index(drop=True)

        def drift_fill(d_df, idx, direction):
            if idx >= len(d_df): return False
            e = d_df.iloc[idx][price_col]
            if pd.isna(e): return False
            lim = e * (0.99995 if direction == 1 else 1.00005)
            for t in range(1, MAX_HOLD + 1):
                if idx + t >= len(d_df): return False
                x = d_df.iloc[idx + t][price_col]
                if pd.isna(x): continue
                if direction == 1 and x <= lim: return True
                if direction == -1 and x >= lim: return True
                lim = x * (0.99995 if direction == 1 else 1.00005)
            return False

        per_th = {}
        for dir_th in DIR_THS:
            d_df = bt
            trades = []; i, n = 0, len(d_df)
            n_long = 0; n_short = 0
            while i < n:
                r = d_df.iloc[i]
                if pd.isna(r["actual_return"]) or pd.isna(r[price_col]):
                    i += 1; continue
                direction = 0; trade = False
                if r["vol_proba"] > VOL_TH:
                    if r["dir_proba"] > dir_th: direction = 1; trade = True; n_long += 1
                    elif r["dir_proba"] < (1 - dir_th): direction = -1; trade = True; n_short += 1
                if trade:
                    e = r[price_col]; ar = direction * r["actual_return"]; sl = False
                    for t in range(1, LOCKOUT + 1):
                        if i + t >= n: break
                        x = d_df.iloc[i + t][price_col]
                        if pd.isna(x): continue
                        p = direction * (x - e) / e * 100
                        if p <= -SL: ar = -SL; sl = True; break
                    if sl: fee_e = FEE_TAKER
                    else:
                        filled = drift_fill(d_df, i + LOCKOUT, -direction)
                        fee_e = FEE_MAKER if filled else FEE_TAKER
                    trades.append({"net_pnl": ar - (FEE_TAKER + fee_e)})
                    i += LOCKOUT
                else:
                    i += 1
            n_total = len(trades)
            ps = sum(t["net_pnl"] for t in trades) if trades else 0
            wr = (sum(1 for t in trades if t["net_pnl"] > 0) / n_total) if n_total else 0
            per_th[str(dir_th)] = {"pnl": ps, "n_trades": n_total, "win_rate": wr,
                                    "n_long": n_long, "n_short": n_short}

        log.info(f"    AUC {auc_self:.3f}  TH0.58 {per_th['0.58']['pnl']:+.3f}% ({per_th['0.58']['n_trades']}t L{per_th['0.58']['n_long']}/S{per_th['0.58']['n_short']})")
        walk_results.append({"step": step_idx, "test_date": test_date,
                             "auc_self": float(auc_self), "per_th": per_th})

    return walk_results, DIR_THS


def summarize(mode, walk_results, dir_ths, log):
    log.info(f"\n--- {mode} summary ---")
    out = {"mode": mode, "per_th": {}, "steps": walk_results}
    for th in dir_ths:
        pnls = [r["per_th"][str(th)]["pnl"] for r in walk_results]
        v = np.array(pnls)
        out["per_th"][str(th)] = {
            "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "total": float(v.sum()), "positive": int((v > 0).sum()),
            "n": len(v),
        }
        log.info(f"  TH {th}: mean {v.mean():+.3f}%/day  std {v.std():.3f}  pos {(v>0).sum()}/{len(v)}  total {v.sum():+.3f}%")
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("AUDIT: norm lookahead bias")
    log.info("=" * 70)
    np.random.seed(42)

    log.info("\nBuilding Tardis...")
    tardis_train = build_split(DATES_TRAIN, log)
    tardis_val = build_split(DATES_VAL, log)
    feat_pre = get_feature_columns(tardis_train)
    tt_clean = tardis_train.dropna(subset=["target_volatility_300s", "target_return_3600s"])
    medians = tt_clean.reindex(columns=feat_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    SELF_ALL = [f"2026-04-{d:02d}" for d in range(21, 31)]  # 4/21-4/30 (10d, 9 walk-forward)
    self_dfs = {}
    log.info("Building Self...")
    for d in SELF_ALL:
        df = build_self_date_dataset(d, log, train_medians=medians)
        df = df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
        self_dfs[d] = df
    sorted_self = sorted(self_dfs.keys())
    test_dates = sorted_self[1:]  # 4/22-4/30 (9 days)

    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
    tardis_train.dropna(subset=[vol_target, dir_target], inplace=True)
    tardis_val.dropna(subset=[vol_target, dir_target], inplace=True)
    for d in self_dfs:
        self_dfs[d] = self_dfs[d].dropna(subset=[vol_target, dir_target])

    # Pre-compute yesterday means for Self days
    log.info("\nComputing per-day means + yesterday lookups...")
    prev_day_means_self = yesterday_means_for_self(self_dfs, log)
    log.info(f"  yesterday available for {len(prev_day_means_self)}/{len(self_dfs)} self days "
             f"(missing: {set(sorted_self) - set(prev_day_means_self.keys())})")

    # Run all 3 modes
    summaries = {}
    for mode in ["LEAKY", "CAUSAL", "MISMATCH"]:
        results, ths = run_walk_forward(
            mode, tardis_train, tardis_val, self_dfs, sorted_self,
            test_dates, prev_day_means_self, log,
        )
        summaries[mode] = summarize(mode, results, ths, log)

    # Compare
    print()
    print("=" * 90)
    print("COMPARISON: LEAKY vs CAUSAL vs MISMATCH (per TH)")
    print("=" * 90)
    print(f"{'TH':<6} {'Mode':<10} {'Mean PnL':<14} {'Std':<8} {'Positive':<10} {'Total':<10}")
    print("-" * 70)
    for th in ["0.50", "0.55", "0.58"]:
        for mode in ["LEAKY", "CAUSAL", "MISMATCH"]:
            d = summaries[mode]["per_th"].get(th, {})
            mean = d.get("mean", float("nan"))
            std = d.get("std", float("nan"))
            pos = d.get("positive", 0)
            n = d.get("n", 0)
            total = d.get("total", 0)
            print(f"{th:<6} {mode:<10} {mean:+.3f}%/day      {std:.3f}    {pos}/{n:<8} {total:+.3f}%")
        print()

    # Diagnosis
    print("=" * 90)
    print("DIAGNOSIS")
    print("=" * 90)
    th_best = "0.58"
    leaky_m = summaries["LEAKY"]["per_th"][th_best]["mean"]
    causal_m = summaries["CAUSAL"]["per_th"][th_best]["mean"]
    mismatch_m = summaries["MISMATCH"]["per_th"][th_best]["mean"]
    leak_inflation = leaky_m - causal_m
    mismatch_cost = causal_m - mismatch_m
    print(f"\nAt TH {th_best}:")
    print(f"  LEAKY (current sido36/mark36_v2):  {leaky_m:+.3f}%/day")
    print(f"  CAUSAL (yesterday-norm, no leak):  {causal_m:+.3f}%/day")
    print(f"  MISMATCH (leaky train + causal test, ≈ SHADOW v2):  {mismatch_m:+.3f}%/day")
    print(f"\n  Leak inflation: LEAKY - CAUSAL = {leak_inflation:+.3f}%/day")
    print(f"  Mismatch cost:  CAUSAL - MISMATCH = {mismatch_cost:+.3f}%/day")

    if causal_m >= 0.3:
        verdict = "audit-pass"
        rec = "  ✅ CAUSAL ≥ +0.3%/day — mark36_v3 (causal-trained) viable. Proceed to PAPER."
    elif causal_m >= 0:
        verdict = "audit-marginal"
        rec = "  🟡 CAUSAL marginal positive — leak provided most of mark36_v2 PnL; mark36_v3 weak alpha."
    else:
        verdict = "audit-fail"
        rec = "  ❌ CAUSAL ≤ 0 — mark36_v2 +1.074%/day was largely an artifact of the leak."
    print(rec)
    print(f"\n  SHADOW v2 expected vs actual: train was leaky → live causal — expect approx MISMATCH PnL ({mismatch_m:+.3f}%/day).")

    # Save
    out = {
        "approach": "audit lookahead bias in add_normalized_features",
        "summaries": summaries,
        "comparison_at_th_0.58": {
            "leaky_mean": leaky_m, "causal_mean": causal_m, "mismatch_mean": mismatch_m,
            "leak_inflation": leak_inflation, "mismatch_cost": mismatch_cost,
            "verdict": verdict,
        },
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/audit_norm_lookahead.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")
    log.info("Audit complete")


if __name__ == "__main__":
    main()
