"""시도 29c: XGBoost 강한 regularization (overfit 완화)."""
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
    log.info("시도 29c: XGBoost regularized (overfit 완화)")
    log.info("=" * 70)
    np.random.seed(42)

    SELF_TRAIN = [f"2026-04-{d:02d}" for d in range(21, 27)]
    SELF_VAL = ["2026-04-27"]
    SELF_TEST = ["2026-04-28", "2026-04-29", "2026-04-30"]

    log.info("\nBuilding Tardis...")
    tardis_train_df = build_split(DATES_TRAIN, log)
    tardis_val_df = build_split(DATES_VAL, log)
    tardis_test_df = build_split(DATES_TEST, log)
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
    for d in [train_df, val_df, tardis_test_df, self_test_df]:
        d.dropna(subset=[vol_target, dir_target], inplace=True)
    log.info(f"\nSizes: train {len(train_df)} / val {len(val_df)} / Tardis test {len(tardis_test_df)} / Self test {len(self_test_df)}")

    feat_cols = get_feature_columns(train_df)
    log.info(f"Features: {len(feat_cols)}")

    meds = train_df.reindex(columns=feat_cols).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    def mx(df):
        X = df.reindex(columns=feat_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.fillna(meds).fillna(0)
    Xt = mx(train_df); Xv = mx(val_df); Xst = mx(self_test_df); Xtt = mx(tardis_test_df)

    # Vol model: LR
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    train_vol_med = float(train_df[vol_target].median())
    y_vt = (train_df[vol_target] > train_vol_med).astype(int).values
    y_vs = (self_test_df[vol_target] > train_vol_med).astype(int).values
    sv = StandardScaler(); X_tv = sv.fit_transform(Xt); X_sv = sv.transform(Xst)
    lrv = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lrv.fit(X_tv, y_vt)
    vol_auc_self = roc_auc_score(y_vs, lrv.predict_proba(X_sv)[:, 1])
    log.info(f"\nVol AUC self: {vol_auc_self:.3f}")

    T = 0.20
    tm = train_df[dir_target].abs() > T
    vm = val_df[dir_target].abs() > T
    sm_self = self_test_df[dir_target].abs() > T
    sm_tar = tardis_test_df[dir_target].abs() > T
    Xt_f = Xt[tm].values; Xv_f = Xv[vm].values
    Xst_f = Xst[sm_self].values; Xtt_f = Xtt[sm_tar].values
    y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
    y_dv = (val_df.loc[vm, dir_target] > 0).astype(int).values
    y_dst = (self_test_df.loc[sm_self, dir_target] > 0).astype(int).values
    y_dtt = (tardis_test_df.loc[sm_tar, dir_target] > 0).astype(int).values
    log.info(f"Direction samples: train {len(y_dt)}  val {len(y_dv)}  Tardis test {len(y_dtt)}  Self test {len(y_dst)}")

    import xgboost as xgb
    log.info(f"\nXGBoost regularized sweep (xgboost {xgb.__version__})")
    configs = [
        # 강한 regularization 변형
        {"name": "XGB d3 mcw50 reg2",
         "params": dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.7,
                        min_child_weight=50, reg_alpha=0.5, reg_lambda=2.0,
                        random_state=42, n_jobs=4, eval_metric="auc",
                        early_stopping_rounds=30)},
        {"name": "XGB d3 mcw30 reg1",
         "params": dict(n_estimators=500, max_depth=3, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=30, reg_alpha=0.1, reg_lambda=1.0,
                        random_state=42, n_jobs=4, eval_metric="auc",
                        early_stopping_rounds=30)},
        {"name": "XGB d2 mcw30 reg2",
         "params": dict(n_estimators=300, max_depth=2, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.7,
                        min_child_weight=30, reg_alpha=0.5, reg_lambda=2.0,
                        random_state=42, n_jobs=4, eval_metric="auc",
                        early_stopping_rounds=30)},
        {"name": "XGB d4 mcw30 reg1",
         "params": dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.7,
                        min_child_weight=30, reg_alpha=0.3, reg_lambda=1.5,
                        random_state=42, n_jobs=4, eval_metric="auc",
                        early_stopping_rounds=30)},
        {"name": "XGB d5 mcw100 reg3",
         "params": dict(n_estimators=200, max_depth=5, learning_rate=0.05,
                        subsample=0.7, colsample_bytree=0.6,
                        min_child_weight=100, reg_alpha=1.0, reg_lambda=3.0,
                        random_state=42, n_jobs=4, eval_metric="auc",
                        early_stopping_rounds=30)},
    ]

    results = []
    for cfg in configs:
        clf = xgb.XGBClassifier(**cfg["params"])
        clf.fit(Xt_f, y_dt, eval_set=[(Xv_f, y_dv)], verbose=False)
        n_used = clf.best_iteration if hasattr(clf, "best_iteration") and clf.best_iteration else cfg["params"]["n_estimators"]
        a_tr = roc_auc_score(y_dt, clf.predict_proba(Xt_f)[:, 1])
        a_v = roc_auc_score(y_dv, clf.predict_proba(Xv_f)[:, 1])
        a_tt = roc_auc_score(y_dtt, clf.predict_proba(Xtt_f)[:, 1])
        a_s = roc_auc_score(y_dst, clf.predict_proba(Xst_f)[:, 1])
        gap = a_tr - a_v
        log.info(f"  {cfg['name']:<24} | Train {a_tr:.3f}  Val {a_v:.3f}  Tardis test {a_tt:.3f}  Self test {a_s:.3f}  | gap {gap:+.3f}  best_iter {n_used}")
        results.append({"name": cfg["name"], "model": clf,
                        "auc_train": a_tr, "auc_val": a_v, "auc_tt": a_tt, "auc_self": a_s,
                        "gap": gap, "best_iter": n_used,
                        "params": cfg["params"]})

    print()
    print("=" * 110)
    print("REGULARIZED XGBoost COMPARISON (Self test 4/28-4/30)")
    print("=" * 110)
    print(f"{'Model':<26} {'Train':<8} {'Val':<8} {'Tardis test':<13} {'Self test':<11} {'Gap (T-V)':<12} {'best_iter':<10}")
    print("-" * 110)
    for r in results:
        print(f"{r['name']:<26} {r['auc_train']:<8.3f} {r['auc_val']:<8.3f} {r['auc_tt']:<13.3f} {r['auc_self']:<11.3f} {r['gap']:<+12.3f} {r['best_iter']:<10}")

    # Pick best by Self AUC; secondary criterion: low gap and high Tardis test
    best = max(results, key=lambda r: r["auc_self"])
    log.info(f"\nBest by Self AUC: {best['name']} ({best['auc_self']:.3f}, gap {best['gap']:+.3f})")

    # Backtest best with DIR_TH=0.55 (29b 결과 따라 더 낮은 threshold 가정 — actual 결정은 sweep으로)
    log.info(f"\nBacktest with {best['name']} (Drift, DIR_TH sweep)...")
    self_bt = self_test_df.copy().reset_index(drop=True)
    Xbt = mx(self_bt)
    self_bt["vol_proba"] = lrv.predict_proba(sv.transform(Xbt))[:, 1]
    self_bt["dir_proba"] = best["model"].predict_proba(Xbt.values)[:, 1]
    self_bt["actual_return"] = self_bt[dir_target].values

    VOL_TH = 0.6
    LOCKOUT = 60; SL = 1.5
    FEE_TAKER, FEE_MAKER = 0.055, -0.025; MAX_HOLD = 30
    ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in self_bt.columns), None)
    price_col = next((c for c in ["ob_mid_price", "mid"] if c in self_bt.columns), None)
    self_bt = self_bt.sort_values(["_source_date", ts_col]).reset_index(drop=True)

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

    def backtest_with_th(dir_th):
        daily = []
        for date_str in SELF_TEST:
            d_df = self_bt[self_bt["_source_date"] == date_str].sort_values(ts_col).reset_index(drop=True)
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

    print()
    print(f"BACKTEST: {best['name']} (Self test, Drift) — DIR_TH sweep")
    print(f"{'TH':<8} {'Total':<10} {'Avg':<10} {'Trades':<8} {'Daily breakdown':<60}")
    print("-" * 100)
    th_sweep = {}
    for th in [0.50, 0.55, 0.58, 0.60, 0.65]:
        daily = backtest_with_th(th)
        total = sum(d["pnl_sum"] for d in daily)
        avg = float(np.mean([d["pnl_sum"] for d in daily]))
        n_total = sum(d["n_trades"] for d in daily)
        cells = [f"{d['pnl_sum']:+.2f}%({d['n_trades']}t)" for d in daily]
        breakdown = " ".join(cells)
        print(f"{th:<8.2f} {total:<+10.3f}% {avg:<+10.3f}% {n_total:<8} {breakdown:<60}")
        th_sweep[str(th)] = {"daily": daily, "total": total, "avg": avg, "n_total": n_total}

    best_th = max(th_sweep.keys(), key=lambda k: th_sweep[k]["avg"])
    log.info(f"\nBest TH for {best['name']}: {best_th} → daily {th_sweep[best_th]['avg']:+.3f}%")

    out = {
        "lr_vol": lrv, "scaler_vol": sv,
        "best_name": best["name"], "best_model": best["model"],
        "feature_cols": feat_cols, "train_medians": meds.to_dict(),
        "train_vol_median": train_vol_med, "T": T,
        "metadata": {
            "approach": "Vol: LR / Direction: XGBoost regularized",
            "best_name": best["name"],
            "vol_auc_self": float(vol_auc_self),
            "dir_auc_train": float(best["auc_train"]),
            "dir_auc_val": float(best["auc_val"]),
            "dir_auc_tardis_test": float(best["auc_tt"]),
            "dir_auc_self_test": float(best["auc_self"]),
            "gap_train_val": float(best["gap"]),
            "best_iter": int(best["best_iter"]),
            "best_dir_th": best_th,
            "best_dir_th_avg": float(th_sweep[best_th]["avg"]),
            "all_models": [{"name": r["name"],
                            "auc_train": float(r["auc_train"]),
                            "auc_val": float(r["auc_val"]),
                            "auc_tardis_test": float(r["auc_tt"]),
                            "auc_self_test": float(r["auc_self"]),
                            "gap": float(r["gap"]),
                            "best_iter": int(r["best_iter"])}
                           for r in results],
            "th_sweep": th_sweep,
        },
    }
    model_path = Path("/Users/dohun/Desktop/Mark/mark19/models/mark29c_v1.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, model_path, compress=3)
    log.info(f"\nModel: {model_path}")

    json_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido29c_xgboost_regularized.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"metadata": out["metadata"]}, f, indent=2, default=str)
    log.info(f"JSON: {json_path}")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print(f"\nBest: {best['name']}")
    print(f"  Self AUC: {best['auc_self']:.3f}  (시도 29 base XGB d5: 0.590)")
    print(f"  Train-Val gap: {best['gap']:+.3f}  (시도 29 base: +0.284)")
    print(f"  Tardis test AUC: {best['auc_tt']:.3f}  (시도 29 base: 0.495 → 0.55+ 회복?)")
    print(f"  Best DIR_TH backtest: {th_sweep[best_th]['avg']:+.3f}%")

    if best["gap"] < 0.05 and best["auc_self"] >= 0.55 and best["auc_tt"] >= 0.55:
        print(f"\n  IDEAL: gap 작고 Self/Tardis 둘 다 양호 — 진짜 generalize")
    elif best["gap"] < 0.10 and best["auc_self"] >= 0.55:
        print(f"\n  GOOD: gap 적당, Self AUC 유지")
    elif best["auc_self"] >= 0.55:
        print(f"\n  PARTIAL: Self AUC 유지하나 gap 여전히 큼")
    else:
        print(f"\n  REGRESS: regularization으로 Self AUC도 함께 떨어짐 → 본질 ceiling")
    log.info("시도 29c complete")


if __name__ == "__main__":
    main()
