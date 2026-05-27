"""시도 29b: DIR_TH sweep on mark29_v1 (XGB n100 d5)."""
import sys, logging, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from mark19.ml.data_prep import DATES_TRAIN, build_split, get_feature_columns

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
    log.info("시도 29b: DIR_TH sweep on mark29_v1")
    log.info("=" * 70)

    SELF_TEST = ["2026-04-28", "2026-04-29", "2026-04-30"]

    # Load model
    bundle = joblib.load("/Users/dohun/Desktop/Mark/mark19/models/mark29_v1.joblib")
    lrv = bundle["lr_vol"]; sv = bundle["scaler_vol"]
    best_model = bundle["best_model"]; best_scaler = bundle["best_scaler"]
    feat_cols = bundle["feature_cols"]
    medians = pd.Series(bundle["train_medians"])
    log.info(f"Loaded model: {bundle['best_name']} (xgb={bundle['best_is_xgb']})")
    log.info(f"  Features: {len(feat_cols)}")

    # Need Tardis medians for self DT synth
    log.info("\nBuilding Tardis (small subset for medians)...")
    tardis_train_df = build_split(DATES_TRAIN[:5], log)
    tardis_train_df.dropna(subset=["target_volatility_300s", "target_return_3600s"], inplace=True)
    feat_pre = get_feature_columns(tardis_train_df)
    tt_meds = tardis_train_df.reindex(columns=feat_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    log.info("Building Self test...")
    dfs = []
    for d in SELF_TEST:
        try:
            df = build_self_date_dataset(d, log, train_medians=tt_meds)
            if len(df) > 0: dfs.append(df)
        except Exception as e:
            log.error(f"  build_self {d}: {e}")
    self_test_df = pd.concat(dfs, ignore_index=True)
    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
    self_test_df.dropna(subset=[vol_target, dir_target], inplace=True)
    log.info(f"Self test: {len(self_test_df)} rows")

    # Predict
    def mx(df):
        X = df.reindex(columns=feat_cols).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.fillna(medians).fillna(0)

    self_bt = self_test_df.copy().reset_index(drop=True)
    Xbt = mx(self_bt)
    self_bt["vol_proba"] = lrv.predict_proba(sv.transform(Xbt))[:, 1]
    if bundle["best_is_xgb"]:
        self_bt["dir_proba"] = best_model.predict_proba(Xbt.values)[:, 1]
    else:
        self_bt["dir_proba"] = best_model.predict_proba(best_scaler.transform(Xbt.values))[:, 1]
    self_bt["actual_return"] = self_bt[dir_target].values

    log.info(f"\nDir proba distribution: min {self_bt['dir_proba'].min():.3f}  q25 {self_bt['dir_proba'].quantile(0.25):.3f}  med {self_bt['dir_proba'].median():.3f}  q75 {self_bt['dir_proba'].quantile(0.75):.3f}  max {self_bt['dir_proba'].max():.3f}")

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
    print("=" * 100)
    print("DIR_TH SWEEP (XGB n100 d5, Self test 4/28-4/30, Drift)")
    print("=" * 100)
    print(f"{'TH':<8} {'4/28':<22} {'4/29':<22} {'4/30':<22} {'Total':<10} {'Avg':<10} {'Trades':<8}")
    print("-" * 100)

    sweep_results = {}
    for th in [0.50, 0.55, 0.58, 0.60, 0.65]:
        daily = backtest_with_th(th)
        total = sum(d["pnl_sum"] for d in daily)
        avg = float(np.mean([d["pnl_sum"] for d in daily]))
        n_total = sum(d["n_trades"] for d in daily)
        cells = [f"{d['pnl_sum']:+.3f}% ({d['n_trades']}t,{d['win_rate']*100:.0f}%w)" for d in daily]
        print(f"{th:<8.2f} {cells[0]:<22} {cells[1]:<22} {cells[2]:<22} {total:<+10.3f}% {avg:<+10.3f}% {n_total:<8}")
        sweep_results[str(th)] = {"daily": daily, "total": total, "avg": avg, "n_total": n_total}

    best_th = max(sweep_results.keys(), key=lambda k: sweep_results[k]["avg"])
    log.info(f"\nBest TH: {best_th}, daily avg {sweep_results[best_th]['avg']:+.3f}%, total {sweep_results[best_th]['n_total']} trades")

    json_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido29b_dir_th_sweep.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(sweep_results, f, indent=2, default=str)
    log.info(f"JSON: {json_path}")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    base_avg = sweep_results[best_th]["avg"]
    if base_avg >= 0.5:
        print(f"GOOD: best TH {best_th} → daily {base_avg:+.3f}%, 시도 23 +0.082% 대비 우월")
    elif base_avg >= 0.0:
        print(f"WEAK POSITIVE: best TH {best_th} → daily {base_avg:+.3f}%, 시도 23 비슷")
    else:
        print(f"NEGATIVE: best TH {best_th} → daily {base_avg:+.3f}%, threshold 조정만으론 부족")
    log.info("시도 29b complete")


if __name__ == "__main__":
    main()
