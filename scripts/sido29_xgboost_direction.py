"""시도 29: XGBoost Direction model (LR ceiling 검증)."""
import sys, logging, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

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
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            log.error(f"  build_self {d}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("시도 29: XGBoost Direction (LR ceiling 검증)")
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

    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"
    for d in [train_df, val_df, tardis_test_df, self_test_df]:
        d.dropna(subset=[vol_target, dir_target], inplace=True)
    log.info(f"\nSizes: train {len(train_df)} / val {len(val_df)} / Tardis test {len(tardis_test_df)} / Self test {len(self_test_df)}")

    feat_cols = get_feature_columns(train_df)
    log.info(f"Features: {len(feat_cols)}")

    # ---- prep matrices ----
    meds = train_df.reindex(columns=feat_cols).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    def mx(df):
        X = df.reindex(columns=feat_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.fillna(meds).fillna(0)
    Xt = mx(train_df); Xv = mx(val_df); Xst = mx(self_test_df); Xtt = mx(tardis_test_df)

    # ---- Vol model: LR (keep) ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    log.info("\n[Vol model] LR (combined train)")
    train_vol_med = float(train_df[vol_target].median())
    y_vt = (train_df[vol_target] > train_vol_med).astype(int).values
    y_vs = (self_test_df[vol_target] > train_vol_med).astype(int).values
    sv = StandardScaler(); X_tv = sv.fit_transform(Xt); X_sv = sv.transform(Xst)
    lrv = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lrv.fit(X_tv, y_vt)
    vol_auc_self = roc_auc_score(y_vs, lrv.predict_proba(X_sv)[:, 1])
    log.info(f"  Vol AUC self: {vol_auc_self:.3f}")

    # ---- Direction filter ----
    T = 0.20
    tm = train_df[dir_target].abs() > T
    vm = val_df[dir_target].abs() > T
    sm_self = self_test_df[dir_target].abs() > T
    sm_tar = tardis_test_df[dir_target].abs() > T
    Xt_f = Xt[tm].values
    Xv_f = Xv[vm].values
    Xst_f = Xst[sm_self].values
    Xtt_f = Xtt[sm_tar].values
    y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
    y_dv = (val_df.loc[vm, dir_target] > 0).astype(int).values
    y_dst = (self_test_df.loc[sm_self, dir_target] > 0).astype(int).values
    y_dtt = (tardis_test_df.loc[sm_tar, dir_target] > 0).astype(int).values
    log.info(f"  Direction samples: train {len(y_dt)}  val {len(y_dv)}  Self test {len(y_dst)}  Tardis test {len(y_dtt)}")

    # ---- LR baseline ----
    log.info("\n[Direction model] LR baseline (시도 23 Base와 동일)")
    sd = StandardScaler(); X_td = sd.fit_transform(Xt_f); X_vd = sd.transform(Xv_f)
    X_sd = sd.transform(Xst_f); X_ttd = sd.transform(Xtt_f)
    lrd = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
    lrd.fit(X_td, y_dt)
    lr_auc_train = roc_auc_score(y_dt, lrd.predict_proba(X_td)[:, 1])
    lr_auc_val = roc_auc_score(y_dv, lrd.predict_proba(X_vd)[:, 1])
    lr_auc_self = roc_auc_score(y_dst, lrd.predict_proba(X_sd)[:, 1])
    lr_auc_tt = roc_auc_score(y_dtt, lrd.predict_proba(X_ttd)[:, 1])
    log.info(f"  LR  | Train {lr_auc_train:.3f}  Val {lr_auc_val:.3f}  Tardis test {lr_auc_tt:.3f}  Self test {lr_auc_self:.3f}")

    # ---- XGBoost sweep ----
    import xgboost as xgb
    log.info(f"\n[Direction model] XGBoost sweep (xgboost {xgb.__version__})")
    configs = [
        {"n_estimators": 50,  "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.03},
    ]
    common = dict(
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=4, eval_metric="auc",
        early_stopping_rounds=20,
    )

    results = []
    results.append({
        "name": "LR (baseline)",
        "model": lrd, "scaler": sd, "is_xgb": False,
        "auc_train": lr_auc_train, "auc_val": lr_auc_val,
        "auc_tt": lr_auc_tt, "auc_self": lr_auc_self,
        "n_estimators_used": None,
    })

    # XGBoost works on raw values (no scaling needed)
    for i, cfg in enumerate(configs):
        name = f"XGB n{cfg['n_estimators']} d{cfg['max_depth']} lr{cfg['learning_rate']}"
        clf = xgb.XGBClassifier(**cfg, **common)
        clf.fit(Xt_f, y_dt, eval_set=[(Xv_f, y_dv)], verbose=False)
        n_used = clf.best_iteration if hasattr(clf, "best_iteration") and clf.best_iteration else cfg["n_estimators"]
        a_tr = roc_auc_score(y_dt, clf.predict_proba(Xt_f)[:, 1])
        a_v = roc_auc_score(y_dv, clf.predict_proba(Xv_f)[:, 1])
        a_tt = roc_auc_score(y_dtt, clf.predict_proba(Xtt_f)[:, 1])
        a_s = roc_auc_score(y_dst, clf.predict_proba(Xst_f)[:, 1])
        log.info(f"  {name:<32} | Train {a_tr:.3f}  Val {a_v:.3f}  Tardis test {a_tt:.3f}  Self test {a_s:.3f}  (best_iter {n_used})")
        results.append({
            "name": name, "model": clf, "scaler": None, "is_xgb": True,
            "auc_train": a_tr, "auc_val": a_v, "auc_tt": a_tt, "auc_self": a_s,
            "n_estimators_used": n_used,
        })

    print()
    print("=" * 100)
    print("DIRECTION MODEL COMPARISON")
    print("=" * 100)
    print(f"{'Model':<34} {'Train':<8} {'Val':<8} {'Tardis test':<13} {'Self test':<11} {'Overfit':<10}")
    print("-" * 100)
    for r in results:
        of = r["auc_train"] - r["auc_val"]
        print(f"{r['name']:<34} {r['auc_train']:<8.3f} {r['auc_val']:<8.3f} {r['auc_tt']:<13.3f} {r['auc_self']:<11.3f} {of:+.3f}")

    best = max(results, key=lambda r: r["auc_self"])
    log.info(f"\nBest by Self test AUC: {best['name']} ({best['auc_self']:.3f})")

    # ---- Top 10 features ----
    print()
    print("=" * 80)
    print(f"TOP 10 FEATURES (best: {best['name']})")
    print("=" * 80)
    if best["is_xgb"]:
        importances = best["model"].feature_importances_
    else:
        importances = np.abs(best["model"].coef_[0])
    top_idx = np.argsort(importances)[-10:][::-1]
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:>2}. {feat_cols[i]:<40} {importances[i]:.4f}")

    # ---- Backtest best ----
    log.info(f"\nBacktest with {best['name']} (Drift)...")
    self_bt = self_test_df.copy().reset_index(drop=True)
    self_bt["vol_proba"] = lrv.predict_proba(sv.transform(mx(self_bt)))[:, 1]
    Xbt = mx(self_bt).values
    if best["is_xgb"]:
        self_bt["dir_proba"] = best["model"].predict_proba(Xbt)[:, 1]
    else:
        self_bt["dir_proba"] = best["model"].predict_proba(best["scaler"].transform(Xbt))[:, 1]
    self_bt["actual_return"] = self_bt[dir_target].values

    DIR_TH, VOL_TH = 0.65, 0.6
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

    daily_results = []
    for date_str in SELF_TEST:
        d_df = self_bt[self_bt["_source_date"] == date_str].sort_values(ts_col).reset_index(drop=True)
        if len(d_df) < 100:
            daily_results.append({"date": date_str, "n_trades": 0, "pnl_sum": 0, "win_rate": 0}); continue
        trades = []; i, n = 0, len(d_df)
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
            daily_results.append({"date": date_str, "n_trades": len(trades), "pnl_sum": ps, "win_rate": wr})
        else:
            daily_results.append({"date": date_str, "n_trades": 0, "pnl_sum": 0, "win_rate": 0})

    daily_avg = float(np.mean([d["pnl_sum"] for d in daily_results]))
    print()
    print(f"BACKTEST: {best['name']} (Self test, Drift)")
    for d in daily_results:
        print(f"  {d['date']}: {d['pnl_sum']:+.3f}% ({d['n_trades']} trades, win {d['win_rate']*100:.1f}%)")
    print(f"\nDaily avg: {daily_avg:+.3f}%")

    print()
    print("=" * 80)
    print("COMPARISON 시도 17 ~ 시도 29")
    print("=" * 80)
    print(f"{'Trial':<28} {'Vol AUC':<10} {'Dir AUC self':<14} {'Daily':<10}")
    print("-" * 70)
    print(f"{'시도 17 (LR Base)':<28} {'-':<10} {'~0.55':<14} {'+1.23%(P2)':<10}")
    print(f"{'시도 22 (Hybrid LR)':<28} {'-':<10} {'0.514':<14} {'-0.696%':<10}")
    print(f"{'시도 23 (Combined LR)':<28} {'0.691':<10} {'0.561':<14} {'+0.082%':<10}")
    print(f"{'시도 23b (Self-only LR)':<28} {'0.691':<10} {'0.504':<14} {'-1.372%':<10}")
    print(f"{'시도 23c (Deep OB LR)':<28} {'0.663':<10} {'0.523':<14} {'+0.039%':<10}")
    print(f"{'시도 29 (' + best['name'] + ')':<28} {vol_auc_self:<10.3f} {best['auc_self']:<14.3f} {daily_avg:<+10.3f}%")

    out = {
        "lr_vol": lrv, "scaler_vol": sv,
        "best_name": best["name"], "best_is_xgb": best["is_xgb"],
        "best_model": best["model"], "best_scaler": best["scaler"],
        "feature_cols": feat_cols,
        "train_medians": meds.to_dict(),
        "train_vol_median": train_vol_med, "T": T,
        "metadata": {
            "approach": "Vol: LR / Direction: XGBoost sweep + LR baseline",
            "best_name": best["name"],
            "best_n_estimators_used": best["n_estimators_used"],
            "vol_auc_self": float(vol_auc_self),
            "dir_auc_train": float(best["auc_train"]),
            "dir_auc_val": float(best["auc_val"]),
            "dir_auc_tardis_test": float(best["auc_tt"]),
            "dir_auc_self_test": float(best["auc_self"]),
            "self_daily_avg": daily_avg,
            "all_models": [{"name": r["name"],
                            "auc_train": float(r["auc_train"]),
                            "auc_val": float(r["auc_val"]),
                            "auc_tardis_test": float(r["auc_tt"]),
                            "auc_self_test": float(r["auc_self"]),
                            "n_estimators_used": r["n_estimators_used"]}
                           for r in results],
        },
    }
    model_path = Path("/Users/dohun/Desktop/Mark/mark19/models/mark29_v1.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, model_path, compress=3)
    log.info(f"\nModel: {model_path}")

    json_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido29_xgboost.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"metadata": out["metadata"], "daily_results": daily_results}, f, indent=2, default=str)
    log.info(f"JSON:  {json_path}")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    delta_lr = best["auc_self"] - lr_auc_self
    print(f"\nXGBoost vs LR (Self test): Δ = {delta_lr:+.3f}")
    if delta_lr >= 0.03:
        print(f"  STRONG nonlinear lift — XGBoost가 진짜 효과")
        print(f"  → mark29_v1 LIVE 적용 검토")
    elif delta_lr >= 0.01:
        print(f"  PARTIAL lift — 약간의 nonlinear pattern 존재")
        print(f"  → 검증 sample 작아서 추가 일자 수집 후 재평가")
    elif delta_lr >= -0.01:
        print(f"  NO lift — LR ceiling이 본질적 한계 (signal-to-noise)")
        print(f"  → 시도 27 (timeframe) 또는 시도 28 (다른 시장) 권장")
    else:
        print(f"  WORSE — XGBoost overfit, LR이 더 robust")
        print(f"  → 시도 30 (funding harvesting) 등 strategy shift")

    overfit_best = best["auc_train"] - best["auc_val"]
    if overfit_best > 0.15:
        print(f"\n  주의: best model overfit gap {overfit_best:+.3f} (train >> val)")
        print(f"  → early_stopping 작동 확인, 더 강한 regularization 필요할 수도")
    log.info("\n시도 29 complete")


if __name__ == "__main__":
    main()
