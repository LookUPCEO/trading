"""Path A: Raw-features baseline PnL re-verify.

Goal: Establish a leak-free PnL baseline. sido32 best config gives mean AUC 0.554
on RAW (canonical) features only — but no PnL was measured. This script runs the
same 9-day walk-forward + sido36v2-style trade simulation on:
  - mark36_v2 config (n=100, d=6, mcw=100)  ← same as audit, no norm features
  - sido32 best   config (n=1000, d=6, mcw=100)  ← higher capacity

vs audit_norm_lookahead results:
  LEAKY (with norm): +1.074%/day
  CAUSAL (causal norm): -0.378%/day
  Path A (no norm at all): ?

If Path A ≥ CAUSAL → norm features add nothing useful (their value was the leak)
If Path A << CAUSAL → causal-norm has SOME real signal even if leaky-norm was inflated
"""
import sys, logging, json, importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.ml.data_prep import DATES_TRAIN, DATES_VAL, build_split, get_feature_columns

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_bsd", _HERE / "backtest_self_data.py")
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
build_self_date_dataset = _mod.build_self_date_dataset


def run_walk_forward(cfg_name, cfg_p, tardis_train, tardis_val, self_dfs, sorted_self,
                      test_dates, log):
    log.info(f"\n=== Config: {cfg_name} ===")
    feat_set = get_feature_columns(tardis_train)  # RAW canonical features only
    log.info(f"  feat_set: {len(feat_set)}")

    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
    T = 0.20

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb_lib

    DIR_THS = [0.55, 0.58]
    VOL_TH = 0.6
    LOCKOUT = 60; SL = 1.5
    FEE_TAKER, FEE_MAKER = 0.055, -0.025; MAX_HOLD = 30

    walk_results = []
    for step_idx, test_date in enumerate(test_dates, 1):
        idx = sorted_self.index(test_date)
        train_self_dates = sorted_self[:idx]
        log.info(f"  STEP {step_idx}/{len(test_dates)} test={test_date} train_self_n={len(train_self_dates)}")

        self_train_df = pd.concat([self_dfs[d] for d in train_self_dates], ignore_index=True)
        self_test_df = self_dfs[test_date].copy()
        train_df = pd.concat([tardis_train, self_train_df], ignore_index=True)

        meds = train_df.reindex(columns=feat_set).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        def mx(df_):
            X = df_.reindex(columns=feat_set).copy().replace([np.inf, -np.inf], np.nan)
            return X.fillna(meds).fillna(0)
        Xt = mx(train_df); Xst = mx(self_test_df)

        vol_med = float(train_df[vol_target].median())
        y_vt = (train_df[vol_target] > vol_med).astype(int).values
        sv = StandardScaler(); X_tv = sv.fit_transform(Xt)
        lrv = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lrv.fit(X_tv, y_vt)

        tm = train_df[dir_target].abs() > T
        sm = self_test_df[dir_target].abs() > T
        Xt_f = Xt[tm].values; Xst_f = Xst[sm].values
        y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
        y_ds = (self_test_df.loc[sm, dir_target] > 0).astype(int).values
        if len(set(y_dt)) < 2 or len(set(y_ds)) < 2:
            log.warning(f"    skip: single-class targets"); continue

        clf = xgb_lib.XGBClassifier(**cfg_p)
        clf.fit(Xt_f, y_dt, verbose=False)
        auc_self = roc_auc_score(y_ds, clf.predict_proba(Xst_f)[:, 1])

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


def summarize(cfg_name, walk_results, dir_ths, log):
    out = {"config": cfg_name, "per_th": {}, "steps": walk_results}
    log.info(f"\n--- {cfg_name} summary ---")
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
    log.info("Path A: Raw-features baseline PnL re-verify (NO norm features)")
    log.info("=" * 70)
    np.random.seed(42)

    log.info("\nBuilding Tardis...")
    tardis_train = build_split(DATES_TRAIN, log)
    tardis_val = build_split(DATES_VAL, log)
    feat_pre = get_feature_columns(tardis_train)
    tt_clean = tardis_train.dropna(subset=["target_volatility_300s", "target_return_3600s"])
    medians = tt_clean.reindex(columns=feat_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    SELF_ALL = [f"2026-04-{d:02d}" for d in range(21, 31)]
    self_dfs = {}
    log.info("Building Self...")
    for d in SELF_ALL:
        df = build_self_date_dataset(d, log, train_medians=medians)
        df = df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
        self_dfs[d] = df
    sorted_self = sorted(self_dfs.keys())
    test_dates = sorted_self[1:]

    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
    tardis_train.dropna(subset=[vol_target, dir_target], inplace=True)
    tardis_val.dropna(subset=[vol_target, dir_target], inplace=True)
    for d in self_dfs:
        self_dfs[d] = self_dfs[d].dropna(subset=[vol_target, dir_target])

    configs = {
        "mark36_v2_cfg_no_norm": dict(n_estimators=100, max_depth=6, learning_rate=0.03,
                                       min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
                                       reg_alpha=1.0, reg_lambda=5.0,
                                       random_state=42, n_jobs=4, eval_metric="auc"),
        "sido32_best_n1000_d6_mcw100": dict(n_estimators=1000, max_depth=6, learning_rate=0.03,
                                              min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
                                              reg_alpha=1.0, reg_lambda=5.0,
                                              random_state=42, n_jobs=4, eval_metric="auc"),
    }

    summaries = {}
    for cfg_name, cfg_p in configs.items():
        results, ths = run_walk_forward(cfg_name, cfg_p, tardis_train, tardis_val, self_dfs,
                                          sorted_self, test_dates, log)
        summaries[cfg_name] = summarize(cfg_name, results, ths, log)

    # Comparison
    print()
    print("=" * 90)
    print("Path A — RAW BASELINE (no norm) vs audit reference")
    print("=" * 90)
    print(f"{'Config':<35} {'TH':<6} {'Mean PnL':<14} {'Pos':<8} {'Total':<10}")
    print("-" * 80)
    for cfg in configs:
        for th in ["0.55", "0.58"]:
            d = summaries[cfg]["per_th"].get(th, {})
            print(f"{cfg:<35} {th:<6} {d['mean']:+.3f}%/day      {d['positive']}/{d['n']:<6} {d['total']:+.3f}%")
    print()
    print("Reference (audit_norm_lookahead, TH 0.58):")
    print(f"  LEAKY (mark36_v2 with leak):       +1.074%/day  (7/9 positive)")
    print(f"  CAUSAL (causal yesterday-norm):    -0.378%/day  (4/9 positive)")
    print(f"  MISMATCH (leaky train + causal test): -0.481%/day")

    # Diagnosis
    print()
    print("=" * 90)
    print("DIAGNOSIS")
    print("=" * 90)
    best_cfg = max(configs.keys(), key=lambda c: summaries[c]["per_th"]["0.58"]["mean"])
    best_pnl = summaries[best_cfg]["per_th"]["0.58"]["mean"]
    print(f"\nBest Path A config @ TH 0.58: {best_cfg} → {best_pnl:+.3f}%/day")
    print(f"  vs CAUSAL (-0.378%/day)         : Δ {best_pnl - (-0.378):+.3f}")
    print(f"  vs LEAKY (+1.074%/day, leak)    : Δ {best_pnl - 1.074:+.3f}")
    if best_pnl >= 0.3:
        print(f"\n  ✅ Raw features alone viable (≥ +0.3%/day). norm features unnecessary.")
    elif best_pnl >= 0:
        print(f"\n  🟡 Raw marginal. Direction model 한계 가능성.")
    else:
        print(f"\n  ❌ Raw also negative. Direction modeling 자체 한계 (not just norm).")

    out = {"approach": "Path A: raw-features baseline PnL", "summaries": summaries}
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/pathA_raw_baseline_pnl.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")
    log.info("Path A complete")


if __name__ == "__main__":
    main()
