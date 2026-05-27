"""시도 29f: Ensemble (LR + XGB avg) walk-forward validation."""
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("시도 29f: Ensemble (LR + XGB) walk-forward")
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

    DIR_THS = [0.50, 0.52, 0.55, 0.58]
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

        # Direction filter
        tm = train_df[dir_target].abs() > T
        vm = val_df[dir_target].abs() > T
        sm = self_test_df[dir_target].abs() > T
        Xt_f = Xt[tm].values; Xv_f = Xv[vm].values; Xst_f = Xst[sm].values
        y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
        y_dv = (val_df.loc[vm, dir_target] > 0).astype(int).values
        y_ds = (self_test_df.loc[sm, dir_target] > 0).astype(int).values

        # LR Direction
        sd = StandardScaler(); X_td = sd.fit_transform(Xt_f); X_sd = sd.transform(Xst_f); X_vd = sd.transform(Xv_f)
        lrd = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lrd.fit(X_td, y_dt)
        lr_auc_train = roc_auc_score(y_dt, lrd.predict_proba(X_td)[:, 1])
        lr_auc_val = roc_auc_score(y_dv, lrd.predict_proba(X_vd)[:, 1])
        lr_auc_self = roc_auc_score(y_ds, lrd.predict_proba(X_sd)[:, 1]) if len(set(y_ds)) > 1 else float("nan")
        lr_proba_full = lrd.predict_proba(sd.transform(mx(self_test_df).values))[:, 1]

        # XGB Direction n100 d5
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=4, eval_metric="auc",
            early_stopping_rounds=20,
        )
        clf.fit(Xt_f, y_dt, eval_set=[(Xv_f, y_dv)], verbose=False)
        n_used = clf.best_iteration if hasattr(clf, "best_iteration") and clf.best_iteration else 100
        xgb_auc_train = roc_auc_score(y_dt, clf.predict_proba(Xt_f)[:, 1])
        xgb_auc_val = roc_auc_score(y_dv, clf.predict_proba(Xv_f)[:, 1])
        xgb_auc_self = roc_auc_score(y_ds, clf.predict_proba(Xst_f)[:, 1]) if len(set(y_ds)) > 1 else float("nan")
        xgb_proba_full = clf.predict_proba(mx(self_test_df).values)[:, 1]

        # Ensemble avg
        ens_proba = (lr_proba_full + xgb_proba_full) / 2.0
        ens_auc_self = roc_auc_score(y_ds, ens_proba[sm.values]) if len(set(y_ds)) > 1 else float("nan")

        log.info(f"  LR  | tr {lr_auc_train:.3f}  val {lr_auc_val:.3f}  self {lr_auc_self:.3f}")
        log.info(f"  XGB | tr {xgb_auc_train:.3f}  val {xgb_auc_val:.3f}  self {xgb_auc_self:.3f}  best_iter {n_used}")
        log.info(f"  ENS | self {ens_auc_self:.3f}")

        # Backtest
        bt = self_test_df.copy().reset_index(drop=True)
        Xbt = mx(bt)
        bt["vol_proba"] = lrv.predict_proba(sv.transform(Xbt))[:, 1]
        bt["dir_proba"] = ens_proba
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
            n_sl = 0; n_maker = 0; n_taker_exit = 0
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
                    if sl:
                        fee_e = FEE_TAKER; n_sl += 1
                    else:
                        filled = drift_fill(d_df, i + LOCKOUT, -direction)
                        fee_e = FEE_MAKER if filled else FEE_TAKER
                        if filled: n_maker += 1
                        else: n_taker_exit += 1
                    trades.append({"net_pnl": ar - (FEE_TAKER + fee_e), "sl": sl})
                    i += LOCKOUT
                else:
                    i += 1
            n_total = len(trades)
            ps = sum(t["net_pnl"] for t in trades) if trades else 0
            wr = (sum(1 for t in trades if t["net_pnl"] > 0) / n_total) if n_total else 0
            maker_rate = (n_maker / max(n_total - n_sl, 1)) if n_total else 0  # of non-SL trades
            per_th[str(dir_th)] = {"n_trades": n_total, "pnl": ps, "win_rate": wr,
                                    "n_sl": n_sl, "n_maker": n_maker, "n_taker_exit": n_taker_exit,
                                    "maker_rate": maker_rate}
            log.info(f"    TH {dir_th}: {ps:+.3f}% ({n_total}t, win {wr*100:.1f}%, SL {n_sl}, maker {n_maker}/{n_total-n_sl}={maker_rate*100:.0f}%)")

        walk_results.append({
            "step": step, "test_date": test_date,
            "train_self_dates": train_self_dates,
            "lr_auc_train": float(lr_auc_train), "lr_auc_val": float(lr_auc_val), "lr_auc_self": float(lr_auc_self),
            "xgb_auc_train": float(xgb_auc_train), "xgb_auc_val": float(xgb_auc_val), "xgb_auc_self": float(xgb_auc_self),
            "ens_auc_self": float(ens_auc_self),
            "best_iter": int(n_used),
            "per_th": per_th,
        })

    # ---- Aggregate ----
    print()
    print("=" * 110)
    print("ENSEMBLE WALK-FORWARD (LR + XGB avg)")
    print("=" * 110)
    print(f"\n{'Step':<6} {'Test':<14} {'LR self':<9} {'XGB self':<10} {'ENS self':<10}")
    print("-" * 55)
    for r in walk_results:
        print(f"{r['step']:<6} {r['test_date']:<14} {r['lr_auc_self']:<9.3f} {r['xgb_auc_self']:<10.3f} {r['ens_auc_self']:<10.3f}")

    for th in DIR_THS:
        print(f"\nDIR_TH = {th}")
        print(f"{'Step':<6} {'Test':<14} {'PnL':<10} {'Trades':<8} {'Win':<8} {'SL':<5} {'MakerRate':<10}")
        print("-" * 65)
        pnls = []
        for r in walk_results:
            t = r["per_th"][str(th)]
            print(f"{r['step']:<6} {r['test_date']:<14} {t['pnl']:<+10.3f}% {t['n_trades']:<8} {t['win_rate']*100:<8.1f}% {t['n_sl']:<5} {t['maker_rate']*100:<10.0f}%")
            pnls.append(t["pnl"])
        avg = float(np.mean(pnls)); std = float(np.std(pnls))
        mn = float(np.min(pnls)); mx_ = float(np.max(pnls))
        positive = sum(1 for p in pnls if p > 0)
        print(f"  Mean {avg:+.3f}%  Std {std:.3f}  Min {mn:+.3f}  Max {mx_:+.3f}  Positive {positive}/4 days")

    # ---- Save ----
    out = {
        "approach": "Ensemble (LR + XGB avg) walk-forward",
        "steps": walk_results,
        "summary": {
            "per_th_summary": {
                str(th): {
                    "mean_pnl": float(np.mean([r["per_th"][str(th)]["pnl"] for r in walk_results])),
                    "std_pnl": float(np.std([r["per_th"][str(th)]["pnl"] for r in walk_results])),
                    "min_pnl": float(np.min([r["per_th"][str(th)]["pnl"] for r in walk_results])),
                    "max_pnl": float(np.max([r["per_th"][str(th)]["pnl"] for r in walk_results])),
                    "positive_days": int(sum(1 for r in walk_results if r["per_th"][str(th)]["pnl"] > 0)),
                    "total_days": len(walk_results),
                    "total_trades": int(sum(r["per_th"][str(th)]["n_trades"] for r in walk_results)),
                    "total_makers": int(sum(r["per_th"][str(th)]["n_maker"] for r in walk_results)),
                } for th in DIR_THS
            },
            "ens_auc_mean": float(np.mean([r["ens_auc_self"] for r in walk_results])),
            "ens_auc_std": float(np.std([r["ens_auc_self"] for r in walk_results])),
        }
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido29f_ensemble_walk_forward.json")
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
    print(f"\nBest TH: {best_th}")
    print(f"  Mean: {s['mean_pnl']:+.3f}%/day, Std {s['std_pnl']:.3f}")
    print(f"  Min/Max: {s['min_pnl']:+.3f} / {s['max_pnl']:+.3f}")
    print(f"  Positive: {s['positive_days']}/{s['total_days']} days")
    print(f"  Total trades: {s['total_trades']}, Makers: {s['total_makers']}")

    if s["positive_days"] >= 3 and s["mean_pnl"] >= 0.5:
        print(f"\n  STRONG: 3+/4 양수일 + mean ≥ 0.5%/day → robust ensemble 후보")
    elif s["positive_days"] >= 3 and s["mean_pnl"] >= 0.0:
        print(f"\n  POSITIVE: 3+/4 양수일이지만 average 약함")
    elif s["mean_pnl"] >= 0.5:
        print(f"\n  HIGH MEAN BUT VARIABLE: 평균 양수, std 큼 (positive {s['positive_days']}/4)")
    elif s["mean_pnl"] >= 0.0:
        print(f"\n  WEAK POSITIVE: 평균 약하게 양수, robust 아님")
    else:
        print(f"\n  NEGATIVE: ensemble도 walk-forward로는 음수 → strategy 자체 한계")

    log.info("\n시도 29f complete")


if __name__ == "__main__":
    main()
