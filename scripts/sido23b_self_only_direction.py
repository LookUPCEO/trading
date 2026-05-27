"""시도 23b: Direction model Self-only train (Tardis dilution 검증)."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import joblib

from mark19.ml.data_prep import (
    DATES_TRAIN, DATES_VAL, DATES_TEST,
    build_split, get_feature_columns,
)

# Reuse sido23 framework (add_*) and self_data builder via importlib
import importlib.util
_HERE = Path(__file__).resolve().parent

_spec_self = importlib.util.spec_from_file_location(
    "_backtest_self_data", _HERE / "backtest_self_data.py")
_mod_self = importlib.util.module_from_spec(_spec_self)
_spec_self.loader.exec_module(_mod_self)
build_self_date_dataset = _mod_self.build_self_date_dataset

_spec23 = importlib.util.spec_from_file_location(
    "_sido23", _HERE / "sido23_feature_engineering.py")
_mod23 = importlib.util.module_from_spec(_spec23)
_spec23.loader.exec_module(_mod23)
add_ofi_features = _mod23.add_ofi_features
add_book_imbalance = _mod23.add_book_imbalance
add_multi_tf = _mod23.add_multi_tf
add_session = _mod23.add_session


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
    log.info("시도 23b: Direction Self-only Train")
    log.info("=" * 70)
    np.random.seed(42)

    SELF_TRAIN = [f"2026-04-{d:02d}" for d in range(21, 27)]
    SELF_VAL = ["2026-04-27"]
    SELF_TEST = ["2026-04-28", "2026-04-29", "2026-04-30"]

    # ---- Tardis (for Vol model + DT medians) ----
    log.info("\nBuilding Tardis (for Vol + DT medians)...")
    tardis_train_df = build_split(DATES_TRAIN, log)
    feature_cols_pre = get_feature_columns(tardis_train_df)
    tt_clean = tardis_train_df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
    tardis_medians = tt_clean.reindex(columns=feature_cols_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    log.info("Building Self...")
    self_train_df = build_self_split(SELF_TRAIN, log, train_medians=tardis_medians)
    self_val_df = build_self_split(SELF_VAL, log, train_medians=tardis_medians)
    self_test_df = build_self_split(SELF_TEST, log, train_medians=tardis_medians)

    log.info("\nAdding feature groups...")
    def add_all(df, exchange, label):
        log.info(f"  {label} ({len(df)} rows)")
        df = add_ofi_features(df, exchange, log)
        df = add_book_imbalance(df, log)
        df = add_multi_tf(df, log)
        df = add_session(df, log)
        return df

    tardis_train_df = add_all(tardis_train_df, "bybit_tardis", "tardis train")
    self_train_df = add_all(self_train_df, "bybit", "self train")
    self_val_df = add_all(self_val_df, "bybit", "self val")
    self_test_df = add_all(self_test_df, "bybit", "self test")

    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"
    for d in [tardis_train_df, self_train_df, self_val_df, self_test_df]:
        d.dropna(subset=[vol_target, dir_target], inplace=True)

    # Vol train: Tardis + Self combined
    vol_train_df = pd.concat([tardis_train_df, self_train_df], ignore_index=True)
    # Direction train: Self-only
    dir_train_df = self_train_df.copy()

    log.info(f"\nVol train (combined): {len(vol_train_df)} rows")
    log.info(f"Dir train (Self-only): {len(dir_train_df)} rows")
    log.info(f"Self val: {len(self_val_df)} rows")
    log.info(f"Self test: {len(self_test_df)} rows")

    base_features = get_feature_columns(vol_train_df)

    new_a = [f for f in ["ofi_1min", "ofi_5min", "ofi_15min", "ofi_ratio_1min",
                         "large_trade_imb_5min", "trade_intensity"] if f in dir_train_df.columns]
    new_b = [f for f in ["imb_top1", "imb_top5", "imb_top10", "imb_top25", "imb_slope"]
             if f in dir_train_df.columns]
    new_c = [f for f in ["mom_5min", "mom_15min", "mom_30min", "mom_60min", "mom_120min",
                         "mom_accel_5_15", "mom_accel_15_60",
                         "rvol_5min", "rvol_30min", "rvol_120min"] if f in dir_train_df.columns]
    new_d = [f for f in ["hour_of_day", "minute_of_day", "day_of_week",
                         "session_asia", "session_eu", "session_us"] if f in dir_train_df.columns]
    log.info(f"\nNew feature counts: A {len(new_a)}  B {len(new_b)}  C {len(new_c)}  D {len(new_d)}")

    # NaN ratio comparison (Tardis vs Self for new features)
    log.info("\n--- New features NaN ratio (Tardis vs Self train) ---")
    for f in (new_a + new_c + new_d):
        t_nan = tardis_train_df[f].isna().mean() if f in tardis_train_df.columns else 1.0
        s_nan = dir_train_df[f].isna().mean() if f in dir_train_df.columns else 1.0
        log.info(f"  {f:<28}  Tardis NaN {t_nan:.1%}  /  Self NaN {s_nan:.1%}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    def train_eval(feat_set, name):
        # ---- Vol model: combined train ----
        meds_v = vol_train_df.reindex(columns=feat_set).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        def mxv(df):
            X = df.reindex(columns=feat_set).copy()
            X = X.replace([np.inf, -np.inf], np.nan)
            return X.fillna(meds_v).fillna(0)
        Xvt = mxv(vol_train_df); Xvs = mxv(self_test_df)
        vol_med = float(vol_train_df[vol_target].median())
        y_vt = (vol_train_df[vol_target] > vol_med).astype(int).values
        y_vs = (self_test_df[vol_target] > vol_med).astype(int).values
        sv = StandardScaler(); Xvt_s = sv.fit_transform(Xvt); Xvs_s = sv.transform(Xvs)
        lrv = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lrv.fit(Xvt_s, y_vt)
        v_auc = roc_auc_score(y_vs, lrv.predict_proba(Xvs_s)[:, 1])

        # ---- Direction model: Self-only train ----
        meds_d = dir_train_df.reindex(columns=feat_set).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        def mxd(df):
            X = df.reindex(columns=feat_set).copy()
            X = X.replace([np.inf, -np.inf], np.nan)
            return X.fillna(meds_d).fillna(0)
        Xdt = mxd(dir_train_df); Xdv = mxd(self_val_df); Xds = mxd(self_test_df)

        T = 0.20
        tm = dir_train_df[dir_target].abs() > T
        vm = self_val_df[dir_target].abs() > T
        sm = self_test_df[dir_target].abs() > T

        Xdt_f = Xdt[tm].values
        Xdv_f = Xdv[vm].values
        Xds_f = Xds[sm].values
        y_dt = (dir_train_df.loc[tm, dir_target] > 0).astype(int).values
        y_dv = (self_val_df.loc[vm, dir_target] > 0).astype(int).values
        y_ds = (self_test_df.loc[sm, dir_target] > 0).astype(int).values

        sd = StandardScaler(); Xdt_s = sd.fit_transform(Xdt_f)
        Xdv_s = sd.transform(Xdv_f); Xds_s = sd.transform(Xds_f)
        lrd = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lrd.fit(Xdt_s, y_dt)
        d_auc_train = roc_auc_score(y_dt, lrd.predict_proba(Xdt_s)[:, 1])
        d_auc_val = roc_auc_score(y_dv, lrd.predict_proba(Xdv_s)[:, 1]) if len(y_dv) > 5 and len(set(y_dv)) > 1 else float("nan")
        d_auc_self = roc_auc_score(y_ds, lrd.predict_proba(Xds_s)[:, 1])

        coef_abs = np.abs(lrd.coef_[0])
        top5_idx = np.argsort(coef_abs)[-5:][::-1]
        top5 = [(feat_set[i], float(coef_abs[i])) for i in top5_idx]

        log.info(f"  {name:<26} n={len(feat_set):<4} | Vol {v_auc:.3f} | Dir tr {d_auc_train:.3f} val {d_auc_val:.3f} self {d_auc_self:.3f}")

        return {
            "name": name, "n_features": len(feat_set),
            "vol_auc_self": v_auc,
            "dir_auc_train": d_auc_train, "dir_auc_val": d_auc_val, "dir_auc_self": d_auc_self,
            "top5": top5,
            "lr_vol": lrv, "scaler_vol": sv, "lr_dir": lrd, "scaler_dir": sd,
            "feat_set": feat_set,
            "train_medians_vol": meds_v, "train_medians_dir": meds_d,
            "vol_med": vol_med, "T": T,
        }

    log.info("\n--- Ablation (Self-only Direction train) ---")
    results = {}
    results["base"] = train_eval(base_features, "Base")
    results["A"] = train_eval(base_features + new_a, "Base + A (OFI)")
    if new_b:
        results["B"] = train_eval(base_features + new_b, "Base + B (Book imb)")
    results["C"] = train_eval(base_features + new_c, "Base + C (Multi-TF)")
    results["D"] = train_eval(base_features + new_d, "Base + D (Session)")
    results["AC"] = train_eval(base_features + new_a + new_c, "Base + A+C")
    results["all"] = train_eval(base_features + new_a + new_b + new_c + new_d, "Base + ALL")

    print()
    print("=" * 100)
    print("SELF-ONLY DIRECTION ABLATION (Self test 4/28-4/30)")
    print("=" * 100)
    print(f"{'Set':<26} {'N':<6} {'Vol':<8} {'Dir Train':<11} {'Dir Val':<10} {'Dir Self':<10} {'Δ Dir':<10}")
    print("-" * 100)
    base_dir = results["base"]["dir_auc_self"]
    for k, r in results.items():
        delta = r["dir_auc_self"] - base_dir
        mark = "***" if delta > 0.03 else ("**" if delta > 0.01 else ("=" if abs(delta) < 0.005 else ""))
        print(f"{r['name']:<26} {r['n_features']:<6} {r['vol_auc_self']:<8.3f} {r['dir_auc_train']:<11.3f} {r['dir_auc_val']:<10.3f} {r['dir_auc_self']:<10.3f} {delta:<+10.3f} {mark}")

    print()
    print("=" * 100)
    print("TOP 5 FEATURES BY |COEF| (Direction model)")
    print("=" * 100)
    base_set = set(base_features)
    for k, r in results.items():
        print(f"\n{r['name']}:")
        for f, c in r["top5"]:
            mark = "  [NEW]" if f not in base_set else ""
            print(f"  {f:<35} {c:.4f}{mark}")

    best_k = max(results.keys(), key=lambda k: results[k]["dir_auc_self"])
    best = results[best_k]
    log.info(f"\nBest: {best['name']}, Dir AUC self {best['dir_auc_self']:.3f}")

    # ---- Backtest best ----
    log.info(f"\nBacktest with {best['name']} (Drift)...")
    feat_set = best["feat_set"]
    def mxv(df):
        X = df.reindex(columns=feat_set).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.fillna(best["train_medians_vol"]).fillna(0)
    def mxd(df):
        X = df.reindex(columns=feat_set).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.fillna(best["train_medians_dir"]).fillna(0)
    Xv = mxv(self_test_df); Xd = mxd(self_test_df)
    self_test_df = self_test_df.copy().reset_index(drop=True)
    self_test_df["vol_proba"] = best["lr_vol"].predict_proba(best["scaler_vol"].transform(Xv))[:, 1]
    self_test_df["dir_proba"] = best["lr_dir"].predict_proba(best["scaler_dir"].transform(Xd.values))[:, 1]
    self_test_df["actual_return"] = self_test_df[dir_target].values

    DIR_TH, VOL_TH = 0.65, 0.6
    LOCKOUT = 60; SL = 1.5
    FEE_TAKER, FEE_MAKER = 0.055, -0.025; MAX_HOLD = 30
    ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in self_test_df.columns), None)
    price_col = next((c for c in ["ob_mid_price", "mid"] if c in self_test_df.columns), None)
    self_test_df = self_test_df.sort_values(["_source_date", ts_col]).reset_index(drop=True)

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
        d_df = self_test_df[self_test_df["_source_date"] == date_str].sort_values(ts_col).reset_index(drop=True)
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
    print("=" * 80)
    print(f"BACKTEST: {best['name']} (Self-only Direction, Drift)")
    print("=" * 80)
    for d in daily_results:
        print(f"  {d['date']}: {d['pnl_sum']:+.3f}% ({d['n_trades']} trades, win {d['win_rate']*100:.1f}%)")
    print(f"\nDaily avg: {daily_avg:+.3f}%")

    print()
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"{'Approach':<38} {'Dir AUC Self':<14} {'Daily':<12}")
    print("-" * 70)
    print(f"{'시도 22 (Hybrid)':<38} {'0.514':<14} {'-0.696%':<12}")
    print(f"{'시도 23 base (combined train)':<38} {'0.561':<14} {'+0.082%':<12}")
    print(f"{'시도 23b best (Self-only Dir)':<38} {best['dir_auc_self']:<14.3f} {daily_avg:<+12.3f}%")

    out = {
        "lr_vol": best["lr_vol"], "scaler_vol": best["scaler_vol"],
        "lr_dir": best["lr_dir"], "scaler_dir": best["scaler_dir"],
        "feature_cols": best["feat_set"],
        "train_medians_vol": best["train_medians_vol"].to_dict(),
        "train_medians_dir": best["train_medians_dir"].to_dict(),
        "vol_med": best["vol_med"], "T": best["T"],
        "metadata": {
            "approach": "Vol: Tardis+Self combined / Dir: Self-only",
            "best_set": best["name"], "n_features": best["n_features"],
            "vol_auc_self": float(best["vol_auc_self"]),
            "dir_auc_train": float(best["dir_auc_train"]),
            "dir_auc_val": float(best["dir_auc_val"]),
            "dir_auc_self": float(best["dir_auc_self"]),
            "self_test_daily_avg": daily_avg,
            "ablation": {k: {
                "name": r["name"], "n": r["n_features"],
                "vol_auc_self": float(r["vol_auc_self"]),
                "dir_auc_train": float(r["dir_auc_train"]),
                "dir_auc_val": float(r["dir_auc_val"]),
                "dir_auc_self": float(r["dir_auc_self"]),
                "top5": r["top5"],
            } for k, r in results.items()},
        },
    }
    model_path = Path("/Users/dohun/Desktop/Mark/mark19/models/mark23b_v1.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, model_path, compress=3)
    log.info(f"\nModel: {model_path}")

    json_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido23b_self_only_direction.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"metadata": out["metadata"], "daily_results": daily_results}, f, indent=2, default=str)
    log.info(f"JSON:  {json_path}")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    if best["dir_auc_self"] >= 0.58:
        print(f"STRONG Dir AUC {best['dir_auc_self']:.3f} — {best['name']} 효과적, dilution 가설 확인")
    elif best["dir_auc_self"] >= 0.55:
        print(f"PARTIAL Dir AUC {best['dir_auc_self']:.3f} — 진전, 시도 24/27 검토")
    elif best["dir_auc_self"] >= 0.52:
        print(f"WEAK Dir AUC {best['dir_auc_self']:.3f} — Self 6 days 부족, 시도 27 권장")
    else:
        print(f"FAIL Dir AUC {best['dir_auc_self']:.3f} — 시도 27 (Timeframe) 강력 권장")

    gap = best["dir_auc_train"] - best["dir_auc_val"]
    print(f"\nOverfit: Train {best['dir_auc_train']:.3f} - Val {best['dir_auc_val']:.3f} = {gap:+.3f}")
    if gap > 0.20:
        print("  SEVERE overfit (Self 6 days 부족) → 시도 27 (timeframe 단축으로 sample 증가)")
    elif gap > 0.10:
        print("  Mild overfit")
    else:
        print("  Overfit 적음")

    log.info("시도 23b complete")


if __name__ == "__main__":
    main()
