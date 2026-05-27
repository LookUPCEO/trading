"""시도 23: Feature Engineering — 4 group ablation."""
import sys, logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import joblib

from mark19.ml.data_prep import (
    DATES_TRAIN, DATES_VAL, DATES_TEST,
    build_split, get_feature_columns,
)
from mark19.storage import read_range

# Reuse self_data builder
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_backtest_self_data",
    Path(__file__).resolve().parent / "backtest_self_data.py",
)
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


def add_ofi_features(df, exchange, log):
    """Group A: 1-min OFI from trades. Joins by floor('1min')."""
    if "_source_date" not in df.columns:
        return df
    ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in df.columns), None)
    if ts_col is None:
        return df

    out_dfs = []
    for date_str in df["_source_date"].unique():
        y, m, d = map(int, date_str.split("-"))
        start = datetime(y, m, d, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        try:
            tr = read_range("trades", exchange, "ETHUSDT", start, end)
        except Exception as e:
            log.warning(f"  OFI {date_str}: trades read failed: {e}")
            sub = df[df["_source_date"] == date_str].copy()
            for c in ["ofi_1min", "ofi_5min", "ofi_15min", "ofi_ratio_1min",
                      "large_trade_imb_5min", "trade_intensity"]:
                sub[c] = 0.0
            out_dfs.append(sub); continue
        if len(tr) == 0:
            sub = df[df["_source_date"] == date_str].copy()
            for c in ["ofi_1min", "ofi_5min", "ofi_15min", "ofi_ratio_1min",
                      "large_trade_imb_5min", "trade_intensity"]:
                sub[c] = 0.0
            out_dfs.append(sub); continue

        tr["timestamp"] = pd.to_datetime(tr["timestamp"], utc=True)
        tr = tr.set_index("timestamp")
        size_col = "size" if "size" in tr.columns else ("amount" if "amount" in tr.columns else "qty")
        sign = np.where(tr["side"].astype(str).str.lower().isin(["buy", "b"]), 1, -1)
        tr["signed"] = sign * tr[size_col].astype(float)
        tr["abs_size"] = tr[size_col].astype(float)
        large_thr = tr["abs_size"].quantile(0.9) if len(tr) > 100 else tr["abs_size"].max()
        tr["large_signed"] = np.where(tr["abs_size"] > large_thr, tr["signed"], 0)

        tr_min = tr.resample("1min").agg(
            signed_sum=("signed", "sum"),
            abs_sum=("abs_size", "sum"),
            large_signed_sum=("large_signed", "sum"),
            cnt=("signed", "count"),
        )
        tr_min["ofi_1min"] = tr_min["signed_sum"]
        tr_min["ofi_5min"] = tr_min["signed_sum"].rolling(5).sum()
        tr_min["ofi_15min"] = tr_min["signed_sum"].rolling(15).sum()
        tr_min["ofi_ratio_1min"] = (tr_min["signed_sum"] / tr_min["abs_sum"].replace(0, np.nan)).fillna(0)
        tr_min["large_trade_imb_5min"] = tr_min["large_signed_sum"].rolling(5).sum()
        tr_min["trade_intensity"] = tr_min["cnt"]
        tr_min = tr_min[["ofi_1min", "ofi_5min", "ofi_15min", "ofi_ratio_1min",
                         "large_trade_imb_5min", "trade_intensity"]]

        sub = df[df["_source_date"] == date_str].copy()
        sub["_ts_min"] = pd.to_datetime(sub[ts_col], utc=True).dt.floor("1min")
        tr_min = tr_min.reset_index().rename(columns={"timestamp": "_ts_min"})
        sub = sub.merge(tr_min, on="_ts_min", how="left").drop(columns=["_ts_min"])
        out_dfs.append(sub)

    if out_dfs:
        return pd.concat(out_dfs, ignore_index=True)
    return df


def add_book_imbalance(df, log):
    """Group B: deep book imbalance (top1/5/10/25 + slope)."""
    bid_cols = sorted([c for c in df.columns if c.startswith("ob_bid_") and c.endswith("_size")])
    ask_cols = sorted([c for c in df.columns if c.startswith("ob_ask_") and c.endswith("_size")])
    if len(bid_cols) < 5 or len(ask_cols) < 5:
        log.warning(f"  imb: not enough levels (bid={len(bid_cols)}, ask={len(ask_cols)})")
        return df
    eps = 1e-9
    df = df.copy()
    df["imb_top1"] = df[bid_cols[0]] / (df[bid_cols[0]] + df[ask_cols[0]] + eps) - 0.5
    bid5 = df[bid_cols[:5]].sum(axis=1); ask5 = df[ask_cols[:5]].sum(axis=1)
    df["imb_top5"] = bid5 / (bid5 + ask5 + eps) - 0.5
    if len(bid_cols) >= 10:
        bid10 = df[bid_cols[:10]].sum(axis=1); ask10 = df[ask_cols[:10]].sum(axis=1)
        df["imb_top10"] = bid10 / (bid10 + ask10 + eps) - 0.5
    bidA = df[bid_cols].sum(axis=1); askA = df[ask_cols].sum(axis=1)
    df["imb_top25"] = bidA / (bidA + askA + eps) - 0.5
    df["imb_slope"] = df["imb_top1"] - df["imb_top25"]
    return df


def add_multi_tf(df, log):
    """Group C: multi-TF momentum + realized vol."""
    price_col = next((c for c in ["ob_mid_price", "mid"] if c in df.columns), None)
    if price_col is None:
        return df
    df = df.copy()
    for h in [5, 15, 30, 60, 120]:
        df[f"mom_{h}min"] = df[price_col].pct_change(h)
    df["mom_accel_5_15"] = df["mom_5min"] - df["mom_15min"]
    df["mom_accel_15_60"] = df["mom_15min"] - df["mom_60min"]
    log_ret = np.log(df[price_col] / df[price_col].shift(1))
    for h in [5, 30, 120]:
        df[f"rvol_{h}min"] = log_ret.rolling(h).std()
    return df


def add_session(df, log):
    """Group D: session features from timestamp."""
    ts_col = next((c for c in ["_ts", "ts", "timestamp"] if c in df.columns), None)
    if ts_col is None:
        return df
    df = df.copy()
    ts = pd.to_datetime(df[ts_col], utc=True)
    df["hour_of_day"] = ts.dt.hour
    df["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek
    df["session_asia"] = ((ts.dt.hour >= 0) & (ts.dt.hour < 8)).astype(int)
    df["session_eu"] = ((ts.dt.hour >= 8) & (ts.dt.hour < 16)).astype(int)
    df["session_us"] = ((ts.dt.hour >= 16) & (ts.dt.hour < 24)).astype(int)
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("시도 23: Feature Engineering")
    log.info("=" * 70)
    np.random.seed(42)

    SELF_TRAIN = [f"2026-04-{d:02d}" for d in range(21, 27)]
    SELF_VAL = ["2026-04-27"]
    SELF_TEST = ["2026-04-28", "2026-04-29", "2026-04-30"]

    # Build base + Tardis medians for self DT synth
    log.info("\nBuilding Tardis (for medians)...")
    tardis_train_df = build_split(DATES_TRAIN, log)
    tardis_val_df = build_split(DATES_VAL, log)
    tardis_test_df = build_split(DATES_TEST, log)

    feature_cols_base_pre = get_feature_columns(tardis_train_df)
    tt_clean = tardis_train_df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
    tardis_medians = tt_clean.reindex(columns=feature_cols_base_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    log.info("Building Self...")
    self_train_df = build_self_split(SELF_TRAIN, log, train_medians=tardis_medians)
    self_val_df = build_self_split(SELF_VAL, log, train_medians=tardis_medians)
    self_test_df = build_self_split(SELF_TEST, log, train_medians=tardis_medians)

    # Add new features
    def add_all(df, exchange, label):
        log.info(f"  adding features to {label} ({len(df)} rows)")
        df = add_ofi_features(df, exchange, log)
        df = add_book_imbalance(df, log)
        df = add_multi_tf(df, log)
        df = add_session(df, log)
        return df

    log.info("\nAdding feature groups...")
    tardis_train_df = add_all(tardis_train_df, "bybit_tardis", "tardis train")
    tardis_val_df = add_all(tardis_val_df, "bybit_tardis", "tardis val")
    tardis_test_df = add_all(tardis_test_df, "bybit_tardis", "tardis test")
    self_train_df = add_all(self_train_df, "bybit", "self train")
    self_val_df = add_all(self_val_df, "bybit", "self val")
    self_test_df = add_all(self_test_df, "bybit", "self test")

    train_df = pd.concat([tardis_train_df, self_train_df], ignore_index=True)
    val_df = pd.concat([tardis_val_df, self_val_df], ignore_index=True)
    vol_target = "target_volatility_300s"
    dir_target = "target_return_3600s"
    for d in [train_df, val_df, tardis_test_df, self_test_df]:
        d.dropna(subset=[vol_target, dir_target], inplace=True)
    log.info(f"\nSizes: train {len(train_df)} / val {len(val_df)} / Tardis test {len(tardis_test_df)} / Self test {len(self_test_df)}")

    base_features = get_feature_columns(train_df)
    new_A = [c for c in ["ofi_1min", "ofi_5min", "ofi_15min", "ofi_ratio_1min",
                         "large_trade_imb_5min", "trade_intensity"] if c in train_df.columns]
    new_B = [c for c in ["imb_top1", "imb_top5", "imb_top10", "imb_top25", "imb_slope"]
             if c in train_df.columns]
    new_C = [c for c in ["mom_5min", "mom_15min", "mom_30min", "mom_60min", "mom_120min",
                         "mom_accel_5_15", "mom_accel_15_60",
                         "rvol_5min", "rvol_30min", "rvol_120min"] if c in train_df.columns]
    new_D = [c for c in ["hour_of_day", "minute_of_day", "day_of_week",
                         "session_asia", "session_eu", "session_us"] if c in train_df.columns]
    log.info(f"\nBase {len(base_features)}  A {len(new_A)}  B {len(new_B)}  C {len(new_C)}  D {len(new_D)}")

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
        v_auc_self = roc_auc_score(y_vs, lrv.predict_proba(X_sv)[:, 1])

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
        log.info(f"  {name:<28} n={len(feat_set)} | Vol {v_auc_self:.3f} | Dir self {d_auc_self:.3f} (train {d_auc_train:.3f})")
        return {"name": name, "n": len(feat_set), "vol_auc_self": v_auc_self,
                "dir_auc_self": d_auc_self, "dir_auc_train": d_auc_train,
                "feat_set": feat_set, "lr_vol": lrv, "scaler_vol": sv,
                "lr_dir": lrd, "scaler_dir": sd, "train_medians": meds,
                "train_vol_median": train_vol_median, "T": T}

    log.info("\n--- Ablation ---")
    results = {}
    results["base"] = train_eval(base_features, "Base (시도 22)")
    results["A"] = train_eval(base_features + new_A, "Base + A (OFI)")
    results["B"] = train_eval(base_features + new_B, "Base + B (Book imb)")
    results["C"] = train_eval(base_features + new_C, "Base + C (Multi-TF)")
    results["D"] = train_eval(base_features + new_D, "Base + D (Session)")
    results["all"] = train_eval(base_features + new_A + new_B + new_C + new_D, "Base + ALL")

    print()
    print("=" * 95)
    print("ABLATION (Self test 4/28-4/30)")
    print("=" * 95)
    print(f"{'Set':<28} {'N':<6} {'Vol AUC':<10} {'Dir AUC':<10} {'Train Dir':<12} {'Δ vs base':<12}")
    print("-" * 95)
    base_dir = results["base"]["dir_auc_self"]
    for k, r in results.items():
        delta = r["dir_auc_self"] - base_dir
        mark = "*" if delta > 0.02 else ("+" if delta > 0 else "")
        print(f"{r['name']:<28} {r['n']:<6} {r['vol_auc_self']:<10.3f} {r['dir_auc_self']:<10.3f} {r['dir_auc_train']:<12.3f} {delta:<+12.3f} {mark}")

    best_k = max(results.keys(), key=lambda k: results[k]["dir_auc_self"])
    best = results[best_k]
    print(f"\nBest: {best['name']}, Dir AUC {best['dir_auc_self']:.3f}")

    # ---- Backtest best ----
    log.info(f"\nBacktest with {best['name']}...")
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
    print(f"\nDaily avg: {daily_avg:+.3f}%  vs 시도 22: {daily_avg - (-0.696):+.3f}p")

    # ---- Save ----
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
                             "dir_auc_train": float(r["dir_auc_train"])}
                         for k, r in results.items()},
        },
    }
    model_path = Path("/Users/dohun/Desktop/Mark/mark19/models/mark23_v1.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, model_path, compress=3)
    log.info(f"\nModel: {model_path}")

    json_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido23_feature_engineering.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"metadata": out["metadata"], "daily_results": daily_results}, f, indent=2, default=str)
    log.info(f"JSON:  {json_path}")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    if best["dir_auc_self"] >= 0.55:
        print(f"GOOD Dir AUC {best['dir_auc_self']:.3f} — {best['name']} 효과적")
    elif best["dir_auc_self"] >= 0.52:
        print(f"PARTIAL Dir AUC {best['dir_auc_self']:.3f} — 시도 27 또는 ensemble")
    else:
        print(f"FAIL Dir AUC {best['dir_auc_self']:.3f} — Feature 미미, 시도 27 권장")
    log.info("시도 23 complete")


if __name__ == "__main__":
    main()
