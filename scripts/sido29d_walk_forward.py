"""시도 29d: Walk-forward validation of XGB n100 d5 (mark29 framework)."""
import sys, logging, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from mark19.ml.data_prep import DATES_TRAIN, DATES_VAL, DATES_TEST, build_split, get_feature_columns

import importlib.util
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_bsd", _HERE / "backtest_self_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_self_date_dataset = _mod.build_self_date_dataset


def build_self_split(dates, log, train_medians=None):
    dfs = []
    for d in dates:
        try:
            df = build_self_date_dataset(d, log, train_medians=train_medians)
            if len(df) > 0: dfs.append(df)
        except Exception as e:
            log.error(f"  build_self {d}: {e}")
    if not dfs: return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("시도 29d: Walk-forward validation (4 steps)")
    log.info("=" * 70)
    np.random.seed(42)

    log.info("\nBuilding Tardis (full)...")
    tardis_train_df = build_split(DATES_TRAIN, log)
    tardis_val_df = build_split(DATES_VAL, log)
    feat_pre = get_feature_columns(tardis_train_df)
    tt_clean = tardis_train_df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
    tardis_medians = tt_clean.reindex(columns=feat_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    log.info("Building Self all dates 4/21-30...")
    SELF_ALL = [f"2026-04-{d:02d}" for d in range(21, 31)]
    self_dfs = {}
    for d in SELF_ALL:
        df = build_self_date_dataset(d, log, train_medians=tardis_medians)
        df = df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
        self_dfs[d] = df
        log.info(f"  Self {d}: {len(df)} rows")

    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"
    tardis_train_df.dropna(subset=[vol_target, dir_target], inplace=True)
    tardis_val_df.dropna(subset=[vol_target, dir_target], inplace=True)

    base_features = get_feature_columns(tardis_train_df)
    log.info(f"\nFeatures: {len(base_features)}")

    # Walk-forward: train Tardis + Self[21..N], test Self[N+1]
    steps = [
        {"step": 1, "train_self_end": "2026-04-26", "test": "2026-04-27"},
        {"step": 2, "train_self_end": "2026-04-27", "test": "2026-04-28"},
        {"step": 3, "train_self_end": "2026-04-28", "test": "2026-04-29"},
        {"step": 4, "train_self_end": "2026-04-29", "test": "2026-04-30"},
    ]

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    DIR_THS = [0.50, 0.52, 0.55]
    VOL_TH = 0.6
    LOCKOUT = 60; SL = 1.5
    FEE_TAKER, FEE_MAKER = 0.055, -0.025; MAX_HOLD = 30
    T = 0.20

    walk_results = []

    for step_cfg in steps:
        step = step_cfg["step"]; test_date = step_cfg["test"]
        train_end_date = step_cfg["train_self_end"]
        log.info(f"\n=== STEP {step}: train Self 4/21~{train_end_date.split('-')[2]}, test {test_date} ===")

        train_self_dates = [d for d in SELF_ALL if d <= train_end_date]
        self_train_df = pd.concat([self_dfs[d] for d in train_self_dates], ignore_index=True)
        self_test_df = self_dfs[test_date].copy()

        train_df = pd.concat([tardis_train_df, self_train_df], ignore_index=True)
        val_df = tardis_val_df.copy()
        log.info(f"  train {len(train_df)}  val {len(val_df)}  test {len(self_test_df)}")

        meds = train_df.reindex(columns=base_features).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        def mx(df):
            X = df.reindex(columns=base_features).copy()
            X = X.replace([np.inf, -np.inf], np.nan)
            return X.fillna(meds).fillna(0)
        Xt = mx(train_df); Xv = mx(val_df); Xst = mx(self_test_df)

        # Vol LR
        vol_med = float(train_df[vol_target].median())
        y_vt = (train_df[vol_target] > vol_med).astype(int).values
        sv = StandardScaler(); X_tv = sv.fit_transform(Xt); X_sv = sv.transform(Xst)
        lrv = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lrv.fit(X_tv, y_vt)

        # Direction XGB n100 d5
        tm = train_df[dir_target].abs() > T
        vm = val_df[dir_target].abs() > T
        sm = self_test_df[dir_target].abs() > T
        Xt_f = Xt[tm].values; Xv_f = Xv[vm].values; Xst_f = Xst[sm].values
        y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
        y_dv = (val_df.loc[vm, dir_target] > 0).astype(int).values
        y_ds = (self_test_df.loc[sm, dir_target] > 0).astype(int).values

        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=4, eval_metric="auc",
            early_stopping_rounds=20,
        )
        clf.fit(Xt_f, y_dt, eval_set=[(Xv_f, y_dv)], verbose=False)
        n_used = clf.best_iteration if hasattr(clf, "best_iteration") and clf.best_iteration else 100
        a_tr = roc_auc_score(y_dt, clf.predict_proba(Xt_f)[:, 1])
        a_v = roc_auc_score(y_dv, clf.predict_proba(Xv_f)[:, 1])
        a_s = roc_auc_score(y_ds, clf.predict_proba(Xst_f)[:, 1]) if len(set(y_ds)) > 1 else float("nan")
        log.info(f"  AUC: Train {a_tr:.3f}  Val {a_v:.3f}  Test self {a_s:.3f}  (best_iter {n_used})")

        # Backtest test_date with multiple THs
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
            while i < n:
                r = d_df.iloc[i]
                if pd.isna(r["actual_return"]) or pd.isna(r[price_col]):
                    i += 1; continue
                direction = 0; trade = False
                if r["vol_proba"] > VOL_TH:
                    if r["dir_proba"] > dir_th: direction = 1; trade = True
                    elif r["dir_proba"] < (1 - dir_th): direction = -1; trade = True
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
            ps = sum(t["net_pnl"] for t in trades) if trades else 0
            wr = (sum(1 for t in trades if t["net_pnl"] > 0) / len(trades)) if trades else 0
            per_th[str(dir_th)] = {"n_trades": len(trades), "pnl": ps, "win_rate": wr}
            log.info(f"    TH {dir_th}: {ps:+.3f}% ({len(trades)} trades, win {wr*100:.1f}%)")

        walk_results.append({
            "step": step, "test_date": test_date,
            "train_self_dates": train_self_dates,
            "auc_train": float(a_tr), "auc_val": float(a_v), "auc_self_test": float(a_s),
            "best_iter": int(n_used),
            "per_th": per_th,
        })

    # ---- Aggregate ----
    print()
    print("=" * 110)
    print("WALK-FORWARD RESULTS (XGB n100 d5)")
    print("=" * 110)
    for th in DIR_THS:
        print(f"\nDIR_TH = {th}:")
        print(f"{'Step':<6} {'Test':<14} {'AUC self':<10} {'PnL':<10} {'Trades':<8} {'Win':<8}")
        print("-" * 60)
        pnls = []; trades_n = []
        for r in walk_results:
            t = r["per_th"][str(th)]
            print(f"{r['step']:<6} {r['test_date']:<14} {r['auc_self_test']:<10.3f} {t['pnl']:<+10.3f}% {t['n_trades']:<8} {t['win_rate']*100:<8.1f}%")
            pnls.append(t["pnl"]); trades_n.append(t["n_trades"])
        avg = float(np.mean(pnls)); std = float(np.std(pnls))
        positive_days = sum(1 for p in pnls if p > 0)
        print(f"  Mean {avg:+.3f}%/day  Std {std:.3f}  Positive {positive_days}/4 days  Total trades {sum(trades_n)}")

    # ---- Aggregate AUC consistency ----
    print()
    print("=" * 80)
    print("AUC CONSISTENCY ACROSS WALK-FORWARD STEPS")
    print("=" * 80)
    aucs_self = [r["auc_self_test"] for r in walk_results]
    aucs_train = [r["auc_train"] for r in walk_results]
    aucs_val = [r["auc_val"] for r in walk_results]
    print(f"  AUC train: {[f'{x:.3f}' for x in aucs_train]}  mean {np.mean(aucs_train):.3f}")
    print(f"  AUC val  : {[f'{x:.3f}' for x in aucs_val]}  mean {np.mean(aucs_val):.3f}")
    print(f"  AUC self : {[f'{x:.3f}' for x in aucs_self]}  mean {np.mean(aucs_self):.3f}  std {np.std(aucs_self):.3f}")

    # ---- Save ----
    out = {
        "approach": "Walk-forward XGB n100 d5",
        "steps": walk_results,
        "summary": {
            "auc_self_mean": float(np.mean(aucs_self)),
            "auc_self_std": float(np.std(aucs_self)),
            "per_th_summary": {
                str(th): {
                    "mean_pnl": float(np.mean([r["per_th"][str(th)]["pnl"] for r in walk_results])),
                    "std_pnl": float(np.std([r["per_th"][str(th)]["pnl"] for r in walk_results])),
                    "positive_days": int(sum(1 for r in walk_results if r["per_th"][str(th)]["pnl"] > 0)),
                    "total_days": len(walk_results),
                } for th in DIR_THS
            }
        }
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido29d_walk_forward.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    best_th = max(DIR_THS, key=lambda th: out["summary"]["per_th_summary"][str(th)]["mean_pnl"])
    s = out["summary"]["per_th_summary"][str(best_th)]
    print(f"\nBest TH: {best_th}, mean {s['mean_pnl']:+.3f}%/day, std {s['std_pnl']:.3f}, positive {s['positive_days']}/{s['total_days']}")
    if s["mean_pnl"] >= 0.7 and s["positive_days"] >= 3:
        print(f"  STRONG: 일 {s['mean_pnl']:.2f}% + {s['positive_days']}/4 양수일 → robust")
    elif s["mean_pnl"] >= 0.3:
        print(f"  POSITIVE but variance 큼 (std {s['std_pnl']:.2f})")
    else:
        print(f"  NEGATIVE/MARGINAL: walk-forward에서 강도 약화됨")

    log.info("\n시도 29d complete")


if __name__ == "__main__":
    main()
