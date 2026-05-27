"""시도 29g: Extended walk-forward (9 days) — Ensemble @ TH 0.55."""
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
    log.info("시도 29g: Extended Walk-Forward (9 days)")
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

    # Walk-forward steps: test 4/22..4/30, train Self up to (test-1)
    test_dates = SELF_ALL[1:]  # 4/22..4/30 = 9 days
    log.info(f"Walk-forward steps: {len(test_dates)} days ({test_dates[0]} → {test_dates[-1]})")

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
        train_self_dates = SELF_ALL[:test_dt_idx]  # all dates before test
        log.info(f"\n=== STEP {step_idx}/{len(test_dates)}: train Self {train_self_dates[0]}~{train_self_dates[-1]} ({len(train_self_dates)} days), test {test_date} ===")

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
        xgb_auc_self = roc_auc_score(y_ds, clf.predict_proba(Xst_f)[:, 1]) if len(set(y_ds)) > 1 else float("nan")
        xgb_proba_full = clf.predict_proba(mx(self_test_df).values)[:, 1]

        ens_proba = (lr_proba_full + xgb_proba_full) / 2.0
        ens_auc_self = roc_auc_score(y_ds, ens_proba[sm.values]) if len(set(y_ds)) > 1 else float("nan")

        log.info(f"  AUC: LR {lr_auc_self:.3f}  XGB {xgb_auc_self:.3f}  ENS {ens_auc_self:.3f}  (xgb best_iter {n_used})")

        # Backtest @ TH 0.55
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

        d_df = bt
        trades = []; i, n = 0, len(d_df)
        n_sl = 0; n_maker = 0; n_taker_exit = 0
        while i < n:
            r = d_df.iloc[i]
            if pd.isna(r["actual_return"]) or pd.isna(r[price_col]):
                i += 1; continue
            direction = 0; trade = False
            if r["vol_proba"] > VOL_TH:
                if r["dir_proba"] > DIR_TH: direction = 1; trade = True
                elif r["dir_proba"] < (1 - DIR_TH): direction = -1; trade = True
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
        maker_rate = (n_maker / max(n_total - n_sl, 1)) if n_total else 0
        sl_rate = (n_sl / n_total) if n_total else 0

        log.info(f"  PnL {ps:+.3f}% ({n_total}t, win {wr*100:.1f}%, SL {n_sl} {sl_rate*100:.0f}%, maker {n_maker}/{n_total - n_sl}={maker_rate*100:.0f}%)")

        walk_results.append({
            "step": step_idx, "test_date": test_date,
            "n_train_self_days": len(train_self_dates),
            "lr_auc_self": float(lr_auc_self),
            "xgb_auc_self": float(xgb_auc_self),
            "ens_auc_self": float(ens_auc_self),
            "best_iter": int(n_used),
            "pnl": float(ps), "n_trades": n_total, "win_rate": float(wr),
            "n_sl": n_sl, "n_maker": n_maker, "sl_rate": float(sl_rate), "maker_rate": float(maker_rate),
        })

    # ---- Aggregate ----
    print()
    print("=" * 110)
    print("EXTENDED WALK-FORWARD (Ensemble LR + XGB avg @ TH 0.55) — 9 days")
    print("=" * 110)
    print(f"\n{'Step':<6} {'Date':<14} {'Self days':<10} {'AUC ENS':<10} {'PnL':<10} {'Trades':<8} {'Win':<8} {'SL':<6} {'Maker':<8}")
    print("-" * 90)
    pnls = []
    for r in walk_results:
        print(f"{r['step']:<6} {r['test_date']:<14} {r['n_train_self_days']:<10} {r['ens_auc_self']:<10.3f} {r['pnl']:<+10.3f}% {r['n_trades']:<8} {r['win_rate']*100:<8.1f}% {r['n_sl']:<6} {r['maker_rate']*100:<8.0f}%")
        pnls.append(r["pnl"])

    pnls_arr = np.array(pnls)
    print("\n" + "-" * 90)
    print(f"  Mean: {pnls_arr.mean():+.3f}%/day")
    print(f"  Std : {pnls_arr.std():.3f}")
    print(f"  Min : {pnls_arr.min():+.3f}")
    print(f"  Max : {pnls_arr.max():+.3f}")
    print(f"  Median: {np.median(pnls_arr):+.3f}")
    pos = (pnls_arr > 0).sum()
    print(f"  Positive: {pos}/{len(pnls)} ({pos/len(pnls)*100:.0f}%)")

    # Subset analysis: 4/27 onward (sufficient train data)
    later = [r for r in walk_results if r["test_date"] >= "2026-04-27"]
    if later:
        later_pnls = np.array([r["pnl"] for r in later])
        print(f"\n  [Subset: 4/27 onward, {len(later)} days]")
        print(f"  Mean {later_pnls.mean():+.3f}%/day, Std {later_pnls.std():.3f}, Positive {(later_pnls > 0).sum()}/{len(later_pnls)}")

    # Cumulative
    cum = np.cumsum(pnls_arr)
    print(f"\n  Cumulative PnL over 9 days: {cum[-1]:+.3f}%")

    # ---- Save ----
    out = {
        "approach": "Ensemble (LR + XGB avg) walk-forward extended @ TH 0.55",
        "steps": walk_results,
        "summary": {
            "mean_pnl": float(pnls_arr.mean()),
            "std_pnl": float(pnls_arr.std()),
            "min_pnl": float(pnls_arr.min()),
            "max_pnl": float(pnls_arr.max()),
            "median_pnl": float(np.median(pnls_arr)),
            "positive_days": int(pos),
            "total_days": int(len(pnls)),
            "cumulative_pnl": float(cum[-1]),
            "subset_4_27_onward": {
                "mean": float(later_pnls.mean()) if later else None,
                "std": float(later_pnls.std()) if later else None,
                "positive_days": int((later_pnls > 0).sum()) if later else None,
                "total_days": int(len(later_pnls)) if later else None,
            } if later else None,
        }
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido29g_extended_walk_forward.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    if pos >= 7 and pnls_arr.mean() >= 0.5:
        print(f"\n  STRONG: {pos}/9 positive + mean ≥ 0.5%/day → robust strategy 입증")
    elif pos >= 6 and pnls_arr.mean() >= 0.3:
        print(f"\n  GOOD: {pos}/9 positive + mean ≥ 0.3%/day → 양호하지만 일 1% 미달")
    elif pos >= 5:
        print(f"\n  MODERATE: {pos}/9 positive — 50%+ but variance 큼")
    elif pnls_arr.mean() >= 0.3:
        print(f"\n  HIGH MEAN BUT VARIABLE: mean +{pnls_arr.mean():.3f}, positive only {pos}/9")
    else:
        print(f"\n  WEAK: mean {pnls_arr.mean():+.3f}%/day, positive {pos}/9 — 시도 29f 4/4가 운")

    log.info("\n시도 29g complete")


if __name__ == "__main__":
    main()
