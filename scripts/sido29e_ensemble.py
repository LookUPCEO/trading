"""시도 29e: Ensemble (XGB n100 d5 + LR baseline) average proba."""
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
    log.info("시도 29e: Ensemble (XGB + LR)")
    log.info("=" * 70)
    np.random.seed(42)

    SELF_TRAIN = [f"2026-04-{d:02d}" for d in range(21, 27)]
    SELF_VAL = ["2026-04-27"]
    SELF_TEST = ["2026-04-28", "2026-04-29", "2026-04-30"]

    log.info("\nBuilding Tardis...")
    tardis_train_df = build_split(DATES_TRAIN, log)
    tardis_val_df = build_split(DATES_VAL, log)
    feat_pre = get_feature_columns(tardis_train_df)
    tt_clean = tardis_train_df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
    tardis_medians = tt_clean.reindex(columns=feat_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    log.info("Building Self...")
    self_train_df = build_self_split(SELF_TRAIN, log, train_medians=tardis_medians)
    self_val_df = build_self_split(SELF_VAL, log, train_medians=tardis_medians)
    self_test_df = build_self_split(SELF_TEST, log, train_medians=tardis_medians)

    train_df = pd.concat([tardis_train_df, self_train_df], ignore_index=True)
    val_df = pd.concat([tardis_val_df, self_val_df], ignore_index=True)
    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
    for d in [train_df, val_df, self_test_df]:
        d.dropna(subset=[vol_target, dir_target], inplace=True)
    log.info(f"\nSizes: train {len(train_df)} / val {len(val_df)} / Self test {len(self_test_df)}")

    feat_cols = get_feature_columns(train_df)
    log.info(f"Features: {len(feat_cols)}")

    meds = train_df.reindex(columns=feat_cols).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    def mx(df):
        X = df.reindex(columns=feat_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.fillna(meds).fillna(0)
    Xt = mx(train_df); Xv = mx(val_df); Xst = mx(self_test_df)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    # Vol LR (combined)
    vol_med = float(train_df[vol_target].median())
    y_vt = (train_df[vol_target] > vol_med).astype(int).values
    y_vs = (self_test_df[vol_target] > vol_med).astype(int).values
    sv = StandardScaler(); X_tv = sv.fit_transform(Xt); X_sv = sv.transform(Xst)
    lrv = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lrv.fit(X_tv, y_vt)
    log.info(f"\nVol AUC self: {roc_auc_score(y_vs, lrv.predict_proba(X_sv)[:, 1]):.3f}")

    # Direction filter
    T = 0.20
    tm = train_df[dir_target].abs() > T
    vm = val_df[dir_target].abs() > T
    sm = self_test_df[dir_target].abs() > T
    Xt_f = Xt[tm].values; Xv_f = Xv[vm].values; Xst_f = Xst[sm].values
    y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
    y_dv = (val_df.loc[vm, dir_target] > 0).astype(int).values
    y_ds = (self_test_df.loc[sm, dir_target] > 0).astype(int).values
    log.info(f"Direction samples: train {len(y_dt)}  val {len(y_dv)}  test {len(y_ds)}")

    # LR direction (171 features baseline)
    sd = StandardScaler(); X_td = sd.fit_transform(Xt_f); X_sd = sd.transform(Xst_f)
    lrd = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lrd.fit(X_td, y_dt)
    lr_proba_full = lrd.predict_proba(sd.transform(mx(self_test_df).values))[:, 1]
    lr_auc_self = roc_auc_score(y_ds, lrd.predict_proba(X_sd)[:, 1])
    log.info(f"LR Self AUC: {lr_auc_self:.3f}")

    # XGB direction n100 d5
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=4, eval_metric="auc",
        early_stopping_rounds=20,
    )
    clf.fit(Xt_f, y_dt, eval_set=[(Xv_f, y_dv)], verbose=False)
    xgb_proba_full = clf.predict_proba(mx(self_test_df).values)[:, 1]
    xgb_auc_self = roc_auc_score(y_ds, clf.predict_proba(Xst_f)[:, 1])
    log.info(f"XGB Self AUC: {xgb_auc_self:.3f}  (best_iter {clf.best_iteration})")

    # Ensemble: simple average + rank-based
    def rank_normalize(p):
        # Convert to percentile rank
        s = pd.Series(p)
        return s.rank(method="average") / len(s)

    ens_avg = (lr_proba_full + xgb_proba_full) / 2.0
    lr_rank = rank_normalize(lr_proba_full)
    xgb_rank = rank_normalize(xgb_proba_full)
    ens_rank = (lr_rank + xgb_rank) / 2.0

    # AUC for ensemble (only filtered rows)
    sm_idx = self_test_df.reset_index(drop=True).index[sm.reset_index(drop=True)]
    ens_avg_auc = roc_auc_score(y_ds, ens_avg[sm.values])
    ens_rank_auc = roc_auc_score(y_ds, ens_rank[sm.values])
    log.info(f"Ensemble (avg)  Self AUC: {ens_avg_auc:.3f}")
    log.info(f"Ensemble (rank) Self AUC: {ens_rank_auc:.3f}")

    # Distributions
    print()
    print("=" * 80)
    print("PROBA DISTRIBUTION (Self test)")
    print("=" * 80)
    for name, p in [("LR", lr_proba_full), ("XGB", xgb_proba_full),
                    ("Ensemble avg", ens_avg), ("Ensemble rank", ens_rank)]:
        print(f"  {name:<18} min {p.min():.3f}  q25 {np.quantile(p,0.25):.3f}  med {np.median(p):.3f}  q75 {np.quantile(p,0.75):.3f}  max {p.max():.3f}")

    # Backtest each ensemble variant + DIR_TH sweep
    self_bt = self_test_df.copy().reset_index(drop=True)
    self_bt["vol_proba"] = lrv.predict_proba(sv.transform(mx(self_bt)))[:, 1]
    self_bt["actual_return"] = self_bt[dir_target].values

    VOL_TH = 0.6
    LOCKOUT = 60; SL = 1.5
    FEE_TAKER, FEE_MAKER = 0.055, -0.025; MAX_HOLD = 30
    ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in self_bt.columns), None)
    price_col = next((c for c in ["ob_mid_price", "mid"] if c in self_bt.columns), None)

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

    def backtest_with_proba(dir_proba, dir_th):
        bt = self_bt.copy()
        bt["dir_proba"] = dir_proba
        bt = bt.sort_values(["_source_date", ts_col]).reset_index(drop=True)
        daily = []
        for date_str in SELF_TEST:
            d_df = bt[bt["_source_date"] == date_str].sort_values(ts_col).reset_index(drop=True)
            if len(d_df) < 100:
                daily.append({"date": date_str, "n_trades": 0, "pnl_sum": 0, "win_rate": 0}); continue
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
            if trades:
                ps = sum(t["net_pnl"] for t in trades)
                wr = sum(1 for t in trades if t["net_pnl"] > 0) / len(trades)
                daily.append({"date": date_str, "n_trades": len(trades), "pnl_sum": ps, "win_rate": wr})
            else:
                daily.append({"date": date_str, "n_trades": 0, "pnl_sum": 0, "win_rate": 0})
        return daily

    THS = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]
    variants = {
        "LR only": lr_proba_full,
        "XGB only": xgb_proba_full,
        "Ensemble avg": ens_avg,
        "Ensemble rank": ens_rank,
    }

    print()
    print("=" * 100)
    print("ENSEMBLE BACKTEST (Self test 4/28-4/30, Drift)")
    print("=" * 100)
    print(f"{'Variant':<18} {'TH':<6} {'Total':<10} {'Avg':<10} {'Trades':<8} {'4/28':<14} {'4/29':<14} {'4/30':<14}")
    print("-" * 100)
    all_results = {}
    for vname, p in variants.items():
        all_results[vname] = {}
        for th in THS:
            daily = backtest_with_proba(p, th)
            total = sum(d["pnl_sum"] for d in daily)
            avg = float(np.mean([d["pnl_sum"] for d in daily]))
            n_total = sum(d["n_trades"] for d in daily)
            cells = [f"{d['pnl_sum']:+.2f}%({d['n_trades']}t)" for d in daily]
            print(f"{vname:<18} {th:<6.2f} {total:<+10.3f}% {avg:<+10.3f}% {n_total:<8} {cells[0]:<14} {cells[1]:<14} {cells[2]:<14}")
            all_results[vname][str(th)] = {
                "daily": daily, "total": total, "avg": avg, "n_total": n_total,
            }
        print()

    # Find best variant + TH
    best_combo = None; best_avg = -999
    for vname, ths in all_results.items():
        for th, r in ths.items():
            if r["avg"] > best_avg and r["n_total"] >= 5:
                best_avg = r["avg"]; best_combo = (vname, th)
    log.info(f"\nBest combo: {best_combo[0]} @ TH {best_combo[1]} → daily {best_avg:+.3f}%")

    out = {
        "lr_vol": lrv, "scaler_vol": sv,
        "lr_dir": lrd, "scaler_dir": sd,
        "xgb_dir": clf,
        "feature_cols": feat_cols,
        "train_medians": meds.to_dict(),
        "train_vol_median": vol_med, "T": T,
        "metadata": {
            "approach": "Ensemble (LR + XGB n100 d5)",
            "lr_auc_self": float(lr_auc_self),
            "xgb_auc_self": float(xgb_auc_self),
            "ens_avg_auc_self": float(ens_avg_auc),
            "ens_rank_auc_self": float(ens_rank_auc),
            "best_combo": {"variant": best_combo[0], "dir_th": best_combo[1], "daily_avg": best_avg},
            "all_results": all_results,
        },
    }
    model_path = Path("/Users/dohun/Desktop/Mark/mark19/models/mark29e_v1.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, model_path, compress=3)
    log.info(f"\nModel: {model_path}")

    json_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido29e_ensemble.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"metadata": {k: v for k, v in out["metadata"].items()}}, f, indent=2, default=str)
    log.info(f"JSON: {json_path}")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print(f"\nLR AUC:      {lr_auc_self:.3f}")
    print(f"XGB AUC:     {xgb_auc_self:.3f}")
    print(f"Ens avg AUC: {ens_avg_auc:.3f}")
    print(f"Ens rank AUC:{ens_rank_auc:.3f}")
    if max(ens_avg_auc, ens_rank_auc) > xgb_auc_self:
        print(f"\n  Ensemble > XGB single — diversity 효과 있음")
    elif max(ens_avg_auc, ens_rank_auc) > xgb_auc_self - 0.01:
        print(f"\n  Ensemble ≈ XGB — LR dilution 없음")
    else:
        print(f"\n  Ensemble < XGB — LR이 XGB 신호를 약화")
    log.info("\n시도 29e complete")


if __name__ == "__main__":
    main()
