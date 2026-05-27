"""시도 23c: Deep OB features (mid-imb, price/size slope, OB pressure) 검증."""
import sys, logging, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from mark19.ml.data_prep import DATES_TRAIN, DATES_VAL, DATES_TEST, build_split, get_feature_columns
from mark19.storage import read_range

import importlib.util
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_bsd", _HERE / "backtest_self_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_self_date_dataset = _mod.build_self_date_dataset

DEEP_COLS = [
    "deep_imb_L5to10", "deep_imb_L10to15", "deep_imb_L15to20", "deep_imb_L20to25",
    "deep_bid_slope", "deep_ask_slope", "deep_mid_slope",
    "deep_bid_size_slope", "deep_ask_size_slope",
    "deep_bid_size_delta_1m", "deep_bid_size_delta_5m",
    "deep_ask_size_delta_1m", "deep_ask_size_delta_5m",
    "deep_imb_top10_delta_1m", "deep_imb_top10_delta_5m",
]
GROUP_E = ["deep_imb_L5to10", "deep_imb_L10to15", "deep_imb_L15to20", "deep_imb_L20to25"]
GROUP_F = ["deep_bid_slope", "deep_ask_slope", "deep_mid_slope"]
GROUP_G = ["deep_bid_size_slope", "deep_ask_size_slope"]
GROUP_H = ["deep_bid_size_delta_1m", "deep_bid_size_delta_5m",
           "deep_ask_size_delta_1m", "deep_ask_size_delta_5m",
           "deep_imb_top10_delta_1m", "deep_imb_top10_delta_5m"]


def compute_deep_ob(date_str, exchange, log):
    y, m, d = map(int, date_str.split("-"))
    start = datetime(y, m, d, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    ob = read_range("orderbook", exchange, "ETHUSDT", start, end)
    if len(ob) == 0:
        return pd.DataFrame(columns=DEEP_COLS)

    ts = pd.to_datetime(ob["timestamp"], utc=True)
    ob = ob.set_index(ts)
    ob.index.name = "timestamp"

    n_levels = max([int(c.split("_")[1]) for c in ob.columns if c.startswith("bid_") and c.endswith("_size")] + [0]) + 1
    if n_levels < 25:
        log.warning(f"  {date_str}: only {n_levels} levels, deep features partial")

    out = pd.DataFrame(index=ob.index)

    # Mid-level imbalance (4)
    for (lo, hi, label) in [(5, 10, "L5to10"), (10, 15, "L10to15"),
                            (15, 20, "L15to20"), (20, 25, "L20to25")]:
        bcols = [f"bid_{i}_size" for i in range(lo, min(hi, n_levels)) if f"bid_{i}_size" in ob.columns]
        acols = [f"ask_{i}_size" for i in range(lo, min(hi, n_levels)) if f"ask_{i}_size" in ob.columns]
        if not bcols or not acols:
            out[f"deep_imb_{label}"] = np.nan; continue
        bsum = ob[bcols].sum(axis=1); asum = ob[acols].sum(axis=1)
        out[f"deep_imb_{label}"] = bsum / (bsum + asum + 1e-12) - 0.5

    # Price slope L0-L24 (3)
    eff_n = min(25, n_levels)
    if eff_n >= 5:
        bp_cols = [f"bid_{i}_price" for i in range(eff_n) if f"bid_{i}_price" in ob.columns]
        ap_cols = [f"ask_{i}_price" for i in range(eff_n) if f"ask_{i}_price" in ob.columns]
        x = np.arange(len(bp_cols), dtype=float); x_mean = x.mean(); x_var = ((x - x_mean) ** 2).sum()
        if x_var > 0:
            bp = ob[bp_cols].values
            ap = ob[ap_cols].values
            bp_c = bp - bp.mean(axis=1, keepdims=True)
            ap_c = ap - ap.mean(axis=1, keepdims=True)
            x_c = (x - x_mean)
            out["deep_bid_slope"] = (bp_c * x_c).sum(axis=1) / x_var
            out["deep_ask_slope"] = (ap_c * x_c).sum(axis=1) / x_var
            out["deep_mid_slope"] = (out["deep_bid_slope"] + out["deep_ask_slope"]) / 2.0

            # Size slope L0-L24 (2)
            bs_cols = [f"bid_{i}_size" for i in range(eff_n) if f"bid_{i}_size" in ob.columns]
            as_cols = [f"ask_{i}_size" for i in range(eff_n) if f"ask_{i}_size" in ob.columns]
            bs = ob[bs_cols].values
            asz = ob[as_cols].values
            bs_c = bs - bs.mean(axis=1, keepdims=True)
            as_c = asz - asz.mean(axis=1, keepdims=True)
            out["deep_bid_size_slope"] = (bs_c * x_c).sum(axis=1) / x_var
            out["deep_ask_size_slope"] = (as_c * x_c).sum(axis=1) / x_var

    # Resample to 1-min (last)
    out_1m = out.resample("1min").last()

    # OB pressure: top10 size + imb deltas (6)
    top10_b_cols = [f"bid_{i}_size" for i in range(10) if f"bid_{i}_size" in ob.columns]
    top10_a_cols = [f"ask_{i}_size" for i in range(10) if f"ask_{i}_size" in ob.columns]
    if top10_b_cols and top10_a_cols:
        bsum10 = ob[top10_b_cols].sum(axis=1).resample("1min").last()
        asum10 = ob[top10_a_cols].sum(axis=1).resample("1min").last()
        imb10 = bsum10 / (bsum10 + asum10 + 1e-12) - 0.5
        out_1m["deep_bid_size_delta_1m"] = bsum10.diff(1)
        out_1m["deep_bid_size_delta_5m"] = bsum10.diff(5)
        out_1m["deep_ask_size_delta_1m"] = asum10.diff(1)
        out_1m["deep_ask_size_delta_5m"] = asum10.diff(5)
        out_1m["deep_imb_top10_delta_1m"] = imb10.diff(1)
        out_1m["deep_imb_top10_delta_5m"] = imb10.diff(5)

    # Ensure all DEEP_COLS exist
    for c in DEEP_COLS:
        if c not in out_1m.columns:
            out_1m[c] = np.nan

    return out_1m[DEEP_COLS]


def add_deep_ob_features(df, exchange, log):
    if "_source_date" not in df.columns:
        log.warning("  no _source_date, skip deep OB")
        for c in DEEP_COLS: df[c] = np.nan
        return df
    ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in df.columns), None)
    if ts_col is None:
        log.warning("  no ts col, skip deep OB")
        for c in DEEP_COLS: df[c] = np.nan
        return df

    out_dfs = []
    for date_str in sorted(df["_source_date"].unique()):
        try:
            feat = compute_deep_ob(date_str, exchange, log)
        except Exception as e:
            log.error(f"  deep_ob {date_str}: {e}")
            sub = df[df["_source_date"] == date_str].copy()
            for c in DEEP_COLS: sub[c] = np.nan
            out_dfs.append(sub); continue

        sub = df[df["_source_date"] == date_str].copy()
        sub["_ts_min"] = pd.to_datetime(sub[ts_col], utc=True).dt.floor("1min")
        feat = feat.reset_index().rename(columns={"timestamp": "_ts_min"})
        feat["_ts_min"] = pd.to_datetime(feat["_ts_min"], utc=True)
        sub = sub.merge(feat, on="_ts_min", how="left").drop(columns=["_ts_min"])
        out_dfs.append(sub)

    if out_dfs:
        return pd.concat(out_dfs, ignore_index=True)
    return df


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
    log.info("시도 23c: Deep OB Features")
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

    log.info("\nAdding deep OB features...")
    log.info(f"  Tardis train (26 dates)..."); tardis_train_df = add_deep_ob_features(tardis_train_df, "bybit_tardis", log)
    log.info(f"  Tardis val (4 dates)...");    tardis_val_df = add_deep_ob_features(tardis_val_df, "bybit_tardis", log)
    log.info(f"  Tardis test (6 dates)...");   tardis_test_df = add_deep_ob_features(tardis_test_df, "bybit_tardis", log)
    log.info(f"  Self train (6 dates)...");    self_train_df = add_deep_ob_features(self_train_df, "bybit", log)
    log.info(f"  Self val (1 date)...");       self_val_df = add_deep_ob_features(self_val_df, "bybit", log)
    log.info(f"  Self test (3 dates)...");     self_test_df = add_deep_ob_features(self_test_df, "bybit", log)

    train_df = pd.concat([tardis_train_df, self_train_df], ignore_index=True)
    val_df = pd.concat([tardis_val_df, self_val_df], ignore_index=True)
    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
    for d in [train_df, val_df, tardis_test_df, self_test_df]:
        d.dropna(subset=[vol_target, dir_target], inplace=True)
    log.info(f"\nSizes: train {len(train_df)} / val {len(val_df)} / Self test {len(self_test_df)}")

    # NaN ratio of deep features
    log.info("\n--- Deep features NaN ratio (train) ---")
    for c in DEEP_COLS:
        if c in train_df.columns:
            log.info(f"  {c:<35} NaN {train_df[c].isna().mean():.1%}")

    # Single-feature corr w/ target_return_3600s on Self test
    log.info("\n--- Single-feature corr vs target_return_3600s (Self test) ---")
    for c in DEEP_COLS:
        if c in self_test_df.columns:
            corr = self_test_df[c].corr(self_test_df[dir_target])
            log.info(f"  {c:<35} corr {corr:+.4f}")

    base_features = get_feature_columns(train_df)
    new_E = [c for c in GROUP_E if c in train_df.columns]
    new_F = [c for c in GROUP_F if c in train_df.columns]
    new_G = [c for c in GROUP_G if c in train_df.columns]
    new_H = [c for c in GROUP_H if c in train_df.columns]
    log.info(f"\nFeature counts: Base {len(base_features)}  E {len(new_E)}  F {len(new_F)}  G {len(new_G)}  H {len(new_H)}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    def train_eval(feat_set, name):
        meds = train_df.reindex(columns=feat_set).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        def mx(df):
            X = df.reindex(columns=feat_set).copy()
            X = X.replace([np.inf, -np.inf], np.nan)
            return X.fillna(meds).fillna(0)
        Xt = mx(train_df); Xst = mx(self_test_df)
        train_vol_median = float(train_df[vol_target].median())
        y_vt = (train_df[vol_target] > train_vol_median).astype(int).values
        y_vs = (self_test_df[vol_target] > train_vol_median).astype(int).values
        sv = StandardScaler(); X_tv = sv.fit_transform(Xt); X_sv = sv.transform(Xst)
        lrv = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lrv.fit(X_tv, y_vt)
        v_auc = roc_auc_score(y_vs, lrv.predict_proba(X_sv)[:, 1])

        T = 0.20
        tm = train_df[dir_target].abs() > T
        sm = self_test_df[dir_target].abs() > T
        Xtd = Xt[tm].values; Xsd = Xst[sm].values
        ydt = (train_df.loc[tm, dir_target] > 0).astype(int).values
        yds = (self_test_df.loc[sm, dir_target] > 0).astype(int).values
        sd = StandardScaler(); X_td = sd.fit_transform(Xtd); X_sd = sd.transform(Xsd)
        lrd = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lrd.fit(X_td, ydt)
        d_auc_self = roc_auc_score(yds, lrd.predict_proba(X_sd)[:, 1])
        d_auc_train = roc_auc_score(ydt, lrd.predict_proba(X_td)[:, 1])

        coef_abs = np.abs(lrd.coef_[0])
        top5_idx = np.argsort(coef_abs)[-5:][::-1]
        top5 = [(feat_set[i], float(coef_abs[i])) for i in top5_idx]

        log.info(f"  {name:<28} n={len(feat_set):<4} | Vol {v_auc:.3f} | Dir self {d_auc_self:.3f} (train {d_auc_train:.3f})")
        return {"name": name, "n": len(feat_set),
                "vol_auc_self": v_auc, "dir_auc_self": d_auc_self, "dir_auc_train": d_auc_train,
                "feat_set": feat_set, "top5": top5,
                "lr_vol": lrv, "scaler_vol": sv, "lr_dir": lrd, "scaler_dir": sd,
                "train_medians": meds, "train_vol_median": train_vol_median, "T": T}

    log.info("\n--- Ablation ---")
    results = {}
    results["base"] = train_eval(base_features, "Base (시도 22)")
    if new_E: results["E"] = train_eval(base_features + new_E, "Base + E (mid imb)")
    if new_F: results["F"] = train_eval(base_features + new_F, "Base + F (price slope)")
    if new_G: results["G"] = train_eval(base_features + new_G, "Base + G (size slope)")
    if new_H: results["H"] = train_eval(base_features + new_H, "Base + H (OB pressure)")
    results["all"] = train_eval(base_features + new_E + new_F + new_G + new_H, "Base + ALL (deep)")

    print()
    print("=" * 100)
    print("DEEP OB ABLATION (Self test 4/28-4/30)")
    print("=" * 100)
    print(f"{'Set':<28} {'N':<6} {'Vol AUC':<10} {'Dir AUC':<10} {'Train Dir':<12} {'Δ vs base':<12}")
    print("-" * 100)
    base_dir = results["base"]["dir_auc_self"]
    for k, r in results.items():
        delta = r["dir_auc_self"] - base_dir
        mark = "***" if delta > 0.02 else ("**" if delta > 0.01 else ("=" if abs(delta) < 0.005 else ""))
        print(f"{r['name']:<28} {r['n']:<6} {r['vol_auc_self']:<10.3f} {r['dir_auc_self']:<10.3f} {r['dir_auc_train']:<12.3f} {delta:<+12.3f} {mark}")

    print()
    print("=" * 100)
    print("TOP 5 FEATURES BY |COEF| (Direction model)")
    print("=" * 100)
    base_set = set(base_features)
    for k, r in results.items():
        print(f"\n{r['name']}:")
        for f, c in r["top5"]:
            mark_new = "  [DEEP]" if f in DEEP_COLS else ("" if f in base_set else "  [NEW]")
            print(f"  {f:<38} {c:.4f}{mark_new}")

    best_k = max(results.keys(), key=lambda k: results[k]["dir_auc_self"])
    best = results[best_k]
    log.info(f"\nBest: {best['name']}, Dir AUC {best['dir_auc_self']:.3f}")

    # Backtest
    log.info(f"\nBacktest with {best['name']} (Drift)...")
    feat_set = best["feat_set"]
    def mx(df):
        X = df.reindex(columns=feat_set).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.fillna(best["train_medians"]).fillna(0)
    X_st = mx(self_test_df)
    self_test_df = self_test_df.copy().reset_index(drop=True)
    self_test_df["vol_proba"] = best["lr_vol"].predict_proba(best["scaler_vol"].transform(X_st))[:, 1]
    self_test_df["dir_proba"] = best["lr_dir"].predict_proba(best["scaler_dir"].transform(X_st.values))[:, 1]
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
    print(f"BACKTEST: {best['name']} (Self test, Drift)")
    for d in daily_results:
        print(f"  {d['date']}: {d['pnl_sum']:+.3f}% ({d['n_trades']} trades, win {d['win_rate']*100:.1f}%)")
    print(f"\nDaily avg: {daily_avg:+.3f}%  vs 시도 23: {daily_avg - 0.082:+.3f}p")

    out = {
        "lr_vol": best["lr_vol"], "scaler_vol": best["scaler_vol"],
        "lr_dir": best["lr_dir"], "scaler_dir": best["scaler_dir"],
        "feature_cols": best["feat_set"],
        "train_medians": best["train_medians"].to_dict(),
        "train_vol_median": best["train_vol_median"], "T": best["T"],
        "metadata": {
            "best_set": best["name"], "n_features": best["n"],
            "vol_auc_self": float(best["vol_auc_self"]),
            "dir_auc_self": float(best["dir_auc_self"]),
            "dir_auc_train": float(best["dir_auc_train"]),
            "self_daily_avg": daily_avg,
            "ablation": {k: {"name": r["name"], "n": r["n"],
                             "vol_auc_self": float(r["vol_auc_self"]),
                             "dir_auc_self": float(r["dir_auc_self"]),
                             "dir_auc_train": float(r["dir_auc_train"]),
                             "top5": r["top5"]}
                         for k, r in results.items()},
        },
    }
    model_path = Path("/Users/dohun/Desktop/Mark/mark19/models/mark23c_v1.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, model_path, compress=3)
    log.info(f"\nModel: {model_path}")

    json_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido23c_deep_ob.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"metadata": out["metadata"], "daily_results": daily_results}, f, indent=2, default=str)
    log.info(f"JSON:  {json_path}")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    if best["dir_auc_self"] >= 0.58:
        print(f"STRONG Dir AUC {best['dir_auc_self']:.3f} — {best['name']} 효과적")
    elif best["dir_auc_self"] >= 0.555:
        print(f"PARTIAL Dir AUC {best['dir_auc_self']:.3f} — 진전, LIVE 적용 검토")
    elif best["dir_auc_self"] >= 0.54:
        print(f"WEAK Dir AUC {best['dir_auc_self']:.3f} — 미미한 향상")
    else:
        print(f"NO IMPROVEMENT Dir AUC {best['dir_auc_self']:.3f} — 시도 27/28 검토")
    log.info("시도 23c complete")


if __name__ == "__main__":
    main()
