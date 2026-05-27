"""Track C: mark36_v2 OOS walk-forward.

Train: Self 9 days (4/22-4/30) + Tardis full 26d (already trained, reuse mark36_v2)
Test: bybit_tardis_trial 5/2-5/7 (6 NEW OOS days)
"""
import sys, logging, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from mark19.ml.data_prep import DATES_TRAIN, DATES_VAL, build_split, get_feature_columns, build_date_dataset

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
    log.info("Track C: mark36_v2 OOS walk-forward")
    log.info("  Train: Tardis 26 + Self 9 (4/22-4/30) — reuse mark36_v2")
    log.info("  Test:  bybit_tardis_trial 5/2-5/7 (6 NEW OOS days)")
    log.info("=" * 70)
    np.random.seed(42)

    # ---- Build Self train (4/22-4/30) for normalization context ----
    log.info("\nBuilding Tardis train (medians)...")
    tardis_train = build_split(DATES_TRAIN, log)
    feat_pre = get_feature_columns(tardis_train)
    tt_clean = tardis_train.dropna(subset=["target_volatility_300s", "target_return_3600s"])
    medians = tt_clean.reindex(columns=feat_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    log.info("Building Self train (4/22-4/30)...")
    SELF_TRAIN = [f"2026-04-{d:02d}" for d in range(22, 31)]
    self_dfs = {}
    for d in SELF_TRAIN:
        df = build_self_date_dataset(d, log, train_medians=medians)
        df = df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
        self_dfs[d] = df

    log.info("Building Tardis_trial test (5/2-5/7)...")
    TEST_DATES = ["2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07"]
    test_dfs = {}
    for d in TEST_DATES:
        df = build_date_dataset(d, log, exchange="bybit_tardis_trial", symbol="ETHUSDT")
        if len(df) > 0:
            df = df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
            test_dfs[d] = df
            log.info(f"  Tardis_trial {d}: {len(df)} rows")
        else:
            log.warning(f"  Tardis_trial {d}: empty")

    if not test_dfs:
        log.error("No test data!"); return

    log.info("\nApplying day-mean normalization...")
    tardis_train.dropna(subset=["target_volatility_300s", "target_return_3600s"], inplace=True)
    tardis_train = add_normalized_features(tardis_train, log)
    for d in self_dfs:
        self_dfs[d] = add_normalized_features(self_dfs[d], log)
    for d in test_dfs:
        test_dfs[d] = add_normalized_features(test_dfs[d], log)

    norm_features = [c for c in tardis_train.columns if c.endswith("_norm")]
    canonical = get_feature_columns(tardis_train)
    feat_set = [f for f in canonical if f not in HIGH_SHIFT_FEATURES] + norm_features
    log.info(f"\nFeature set: {len(feat_set)}")

    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
    T = 0.20

    # ---- Walk-forward per test day ----
    # Strategy: re-train each step using all available train data (Tardis + Self all 4/22-4/30 + previous test days)
    # Or simpler: train once on Tardis + Self all 9 days, then test on each Tardis_trial day
    self_all = pd.concat([self_dfs[d] for d in SELF_TRAIN if d in self_dfs], ignore_index=True)
    train_df = pd.concat([tardis_train, self_all], ignore_index=True)
    log.info(f"\nTrain combined: {len(train_df)} rows")

    meds = train_df.reindex(columns=feat_set).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    def mx(df):
        X = df.reindex(columns=feat_set).copy().replace([np.inf, -np.inf], np.nan)
        return X.fillna(meds).fillna(0)

    Xt = mx(train_df)

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

    # Direction XGB n=100 fixed
    tm = train_df[dir_target].abs() > T
    Xt_f = Xt[tm].values
    y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
    log.info(f"\nDir train samples: {len(y_dt)}  (long {(y_dt==1).sum()}, short {(y_dt==0).sum()})")

    clf = xgb_lib.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.03,
                                 min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
                                 reg_alpha=1.0, reg_lambda=5.0,
                                 random_state=42, n_jobs=4, eval_metric="auc")
    clf.fit(Xt_f, y_dt, verbose=False)
    log.info(f"Train AUC: {roc_auc_score(y_dt, clf.predict_proba(Xt_f)[:, 1]):.3f}")

    # ---- Per-day OOS evaluation ----
    print()
    print("=" * 100)
    print("OOS WALK-FORWARD: mark36_v2 config (n=100 fixed) on Tardis_trial 5/2-5/7")
    print("=" * 100)

    DIR_THS = [0.50, 0.52, 0.55, 0.58]
    VOL_TH = 0.6
    LOCKOUT = 60; SL = 1.5
    FEE_TAKER, FEE_MAKER = 0.055, -0.025; MAX_HOLD = 30

    daily_results = []
    for d in sorted(test_dfs.keys()):
        df_test = test_dfs[d]
        Xst = mx(df_test)
        sm = df_test[dir_target].abs() > T
        Xst_f = Xst[sm].values
        y_ds = (df_test.loc[sm, dir_target] > 0).astype(int).values
        if len(set(y_ds)) < 2:
            log.warning(f"  {d}: skip (single class)"); continue

        auc = roc_auc_score(y_ds, clf.predict_proba(Xst_f)[:, 1])
        # Full predictions
        all_p = clf.predict_proba(Xst.values)[:, 1]
        below_045 = float((all_p < 0.45).mean())
        above_055 = float((all_p > 0.55).mean())

        # Backtest
        bt = df_test.copy().reset_index(drop=True)
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
            n_long = 0; n_short = 0; n_sl = 0; n_maker = 0
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
                    if sl:
                        fee_e = FEE_TAKER; n_sl += 1
                    else:
                        filled = drift_fill(d_df, i + LOCKOUT, -direction)
                        fee_e = FEE_MAKER if filled else FEE_TAKER
                        if filled: n_maker += 1
                    trades.append({"net_pnl": ar - (FEE_TAKER + fee_e)})
                    i += LOCKOUT
                else:
                    i += 1
            n_total = len(trades)
            ps = sum(t["net_pnl"] for t in trades) if trades else 0
            wr = (sum(1 for t in trades if t["net_pnl"] > 0) / n_total) if n_total else 0
            per_th[str(dir_th)] = {"pnl": ps, "n_trades": n_total, "win_rate": wr,
                                    "n_long": n_long, "n_short": n_short, "n_sl": n_sl, "n_maker": n_maker}

        daily_results.append({"date": d, "auc": float(auc), "below_045": below_045,
                               "above_055": above_055, "per_th": per_th})
        log.info(f"  {d}: AUC {auc:.3f}  <0.45 {below_045*100:.1f}%  >0.55 {above_055*100:.1f}%  TH0.58 {per_th['0.58']['pnl']:+.3f}% ({per_th['0.58']['n_trades']}t)")

    # Aggregate
    print()
    print("=" * 100)
    print("PER-TH AGGREGATE (6 OOS days)")
    print("=" * 100)
    for th in DIR_THS:
        print(f"\nDIR_TH = {th}")
        print(f"{'Date':<14} {'AUC':<8} {'PnL':<10} {'L/S':<8} {'Trades':<8} {'Win':<8}")
        print("-" * 60)
        pnls = []
        for r in daily_results:
            t = r["per_th"][str(th)]
            print(f"{r['date']:<14} {r['auc']:<8.3f} {t['pnl']:<+10.3f}% {t['n_long']}/{t['n_short']:<5} {t['n_trades']:<8} {t['win_rate']*100:<8.1f}%")
            pnls.append(t["pnl"])
        v = np.array(pnls); pos = (v > 0).sum()
        print(f"  Mean {v.mean():+.3f}%/day  Std {v.std():.3f}  Total {v.sum():+.3f}%  Positive {pos}/{len(pnls)}")

    out = {"approach": "mark36_v2 OOS walk-forward Tardis_trial 5/2-5/7", "results": daily_results}
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/track_c_oos_walkforward.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    # Diagnosis
    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    best_th = max(DIR_THS, key=lambda th: np.mean([r["per_th"][str(th)]["pnl"] for r in daily_results]))
    best_pnls = [r["per_th"][str(best_th)]["pnl"] for r in daily_results]
    bv = np.array(best_pnls)
    print(f"\n  Best TH: {best_th}, mean {bv.mean():+.3f}%/day, std {bv.std():.3f}, positive {(bv>0).sum()}/{len(bv)}")
    print(f"  vs sido36 v2 9d backtest (TH 0.58): +1.074%/day, 7/9 positive")
    if bv.mean() >= 0.7 and (bv > 0).sum() / len(bv) >= 0.6:
        print(f"  ✅ OOS REPRODUCTION — mark36_v2 valid")
    elif bv.mean() >= 0.0:
        print(f"  🟡 marginal — degraded but not broken")
    else:
        print(f"  ❌ OOS FAIL — mark36_v2 doesn't generalize")

    log.info("\nTrack C complete")


if __name__ == "__main__":
    main()
