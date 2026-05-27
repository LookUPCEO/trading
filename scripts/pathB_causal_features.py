"""Path B: Causal regime-invariant features (rolling N-bar mean — no leak).

3 candidate causal norm methods + 2 scale-invariant ratio features added on top of canonical.
Compares to Path A (raw only) and audit_norm_lookahead CAUSAL/LEAKY baselines.

Methods (each 9-day walk-forward, mark36_v2 config n=100 d=6):
  ROLL_60    : feat_norm = feat / rolling_60min_mean (causal)
  ROLL_240   : feat_norm = feat / rolling_4h_mean
  ROLL_1440  : feat_norm = feat / rolling_24h_mean   ← closest to leaky day-mean but causal
  RATIOS     : add ob_imbalance + ob_depth_ratio (no division by mean — pure ratios)

Decision:
  Best ≥ +0.3%/day → causal regime-invariant features viable, mark36_v3 candidate
  Best ≤ 0         → norm features broadly useless (regardless of method)
"""
import sys, logging, json, importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.ml.data_prep import DATES_TRAIN, DATES_VAL, build_split, get_feature_columns

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_bsd", _HERE / "backtest_self_data.py")
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
build_self_date_dataset = _mod.build_self_date_dataset

_spec36 = importlib.util.spec_from_file_location("_sido36", _HERE / "sido36_regime_invariant.py")
_mod36 = importlib.util.module_from_spec(_spec36); _spec36.loader.exec_module(_mod36)
HIGH_SHIFT_FEATURES = _mod36.HIGH_SHIFT_FEATURES


# ---- Causal norm methods ----

def add_rolling_norm(df, window_bars, log):
    """Rolling N-bar causal mean per feature (within day so no cross-day contamination).
    Window in 1-min bars. Falls back to expanding-within-day for first window-1 rows.
    """
    if "_source_date" not in df.columns:
        log.warning("  no _source_date; skip"); return df
    df = df.copy()
    ts_col = "timestamp" if "timestamp" in df.columns else next((c for c in df.columns if "ts" in c.lower()), None)
    if ts_col is None:
        log.warning("  no timestamp col; skip"); return df
    df = df.sort_values(["_source_date", ts_col]).reset_index(drop=True)
    added = 0
    for feat in HIGH_SHIFT_FEATURES:
        if feat not in df.columns: continue
        # Rolling within day (min_periods=1 = expanding for first rows then full window)
        rolling_mean = df.groupby("_source_date")[feat].transform(
            lambda x: x.rolling(window=window_bars, min_periods=1).mean()
        )
        norm_col = f"{feat}_norm"
        df[norm_col] = (df[feat] / rolling_mean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        added += 1
    log.info(f"  Added {added} rolling-{window_bars} norm features")
    return df


def add_ratio_features(df, log):
    """Scale-invariant ratios — no division by aggregate."""
    df = df.copy()
    added = 0
    # OB imbalance (bounded -1..1)
    if "ob_bid_depth_50" in df.columns and "ob_ask_depth_50" in df.columns:
        denom = (df["ob_bid_depth_50"] + df["ob_ask_depth_50"]).replace(0, np.nan)
        df["ob_imbalance_50_ratio"] = ((df["ob_bid_depth_50"] - df["ob_ask_depth_50"]) / denom).fillna(0)
        added += 1
    if "ob_bid_depth_5" in df.columns and "ob_ask_depth_5" in df.columns:
        denom = (df["ob_bid_depth_5"] + df["ob_ask_depth_5"]).replace(0, np.nan)
        df["ob_imbalance_5_ratio"] = ((df["ob_bid_depth_5"] - df["ob_ask_depth_5"]) / denom).fillna(0)
        added += 1
    # Depth ratio (bounded 0..1)
    if "ob_bid_depth_5" in df.columns and "ob_bid_depth_50" in df.columns:
        df["ob_depth_concentration_bid"] = (df["ob_bid_depth_5"] / df["ob_bid_depth_50"].replace(0, np.nan)).fillna(0).clip(0, 1)
        added += 1
    if "ob_ask_depth_5" in df.columns and "ob_ask_depth_50" in df.columns:
        df["ob_depth_concentration_ask"] = (df["ob_ask_depth_5"] / df["ob_ask_depth_50"].replace(0, np.nan)).fillna(0).clip(0, 1)
        added += 1
    # Trade flow ratio (bounded -1..1)
    if "tr_buy_volume" in df.columns and "tr_sell_volume" in df.columns:
        denom = (df["tr_buy_volume"] + df["tr_sell_volume"]).replace(0, np.nan)
        df["tr_flow_ratio"] = ((df["tr_buy_volume"] - df["tr_sell_volume"]) / denom).fillna(0)
        added += 1
    log.info(f"  Added {added} ratio features (scale-invariant)")
    return df


# ---- Backtest ----

def run_walk_forward(method, feature_set_name, tardis_train, tardis_val, self_dfs, sorted_self,
                      test_dates, log):
    log.info(f"\n=== Method={method}, FeatSet={feature_set_name} ===")

    # Apply method to all dataframes (consistent train+test)
    if method == "ROLL_60":
        tardis_train_n = add_rolling_norm(tardis_train.copy(), 60, log)
        tardis_val_n = add_rolling_norm(tardis_val.copy(), 60, log)
        self_n = {d: add_rolling_norm(df.copy(), 60, log) for d, df in self_dfs.items()}
    elif method == "ROLL_240":
        tardis_train_n = add_rolling_norm(tardis_train.copy(), 240, log)
        tardis_val_n = add_rolling_norm(tardis_val.copy(), 240, log)
        self_n = {d: add_rolling_norm(df.copy(), 240, log) for d, df in self_dfs.items()}
    elif method == "ROLL_1440":
        tardis_train_n = add_rolling_norm(tardis_train.copy(), 1440, log)
        tardis_val_n = add_rolling_norm(tardis_val.copy(), 1440, log)
        self_n = {d: add_rolling_norm(df.copy(), 1440, log) for d, df in self_dfs.items()}
    elif method == "RATIOS":
        tardis_train_n = add_ratio_features(tardis_train.copy(), log)
        tardis_val_n = add_ratio_features(tardis_val.copy(), log)
        self_n = {d: add_ratio_features(df.copy(), log) for d, df in self_dfs.items()}
    else:
        raise ValueError(method)

    canonical = get_feature_columns(tardis_train_n)
    if method.startswith("ROLL"):
        norm_cols = [c for c in tardis_train_n.columns if c.endswith("_norm")]
        # Replace high-shift raw with their norm versions
        feat_set = [f for f in canonical if f not in HIGH_SHIFT_FEATURES] + norm_cols
    elif method == "RATIOS":
        ratio_cols = [c for c in tardis_train_n.columns if c.endswith("_ratio") or "concentration" in c]
        feat_set = canonical + ratio_cols
    log.info(f"  feat_set: {len(feat_set)}")

    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
    T = 0.20

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb_lib

    DIR_THS = [0.55, 0.58]
    VOL_TH = 0.6
    LOCKOUT = 60; SL = 1.5
    FEE_TAKER, FEE_MAKER = 0.055, -0.025; MAX_HOLD = 30

    cfg_p = dict(n_estimators=100, max_depth=6, learning_rate=0.03,
                  min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=1.0, reg_lambda=5.0,
                  random_state=42, n_jobs=4, eval_metric="auc")

    walk_results = []
    for step_idx, test_date in enumerate(test_dates, 1):
        idx = sorted_self.index(test_date)
        train_self_dates = sorted_self[:idx]
        log.info(f"  STEP {step_idx}/{len(test_dates)} test={test_date}")

        self_train_df = pd.concat([self_n[d] for d in train_self_dates], ignore_index=True)
        self_test_df = self_n[test_date].copy()
        train_df = pd.concat([tardis_train_n, self_train_df], ignore_index=True)

        meds = train_df.reindex(columns=feat_set).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        def mx(df_):
            X = df_.reindex(columns=feat_set).copy().replace([np.inf, -np.inf], np.nan)
            return X.fillna(meds).fillna(0)
        Xt = mx(train_df); Xst = mx(self_test_df)

        vol_med = float(train_df[vol_target].median())
        y_vt = (train_df[vol_target] > vol_med).astype(int).values
        sv = StandardScaler(); X_tv = sv.fit_transform(Xt)
        lrv = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
        lrv.fit(X_tv, y_vt)

        tm = train_df[dir_target].abs() > T
        sm = self_test_df[dir_target].abs() > T
        Xt_f = Xt[tm].values; Xst_f = Xst[sm].values
        y_dt = (train_df.loc[tm, dir_target] > 0).astype(int).values
        y_ds = (self_test_df.loc[sm, dir_target] > 0).astype(int).values
        if len(set(y_dt)) < 2 or len(set(y_ds)) < 2: continue

        clf = xgb_lib.XGBClassifier(**cfg_p)
        clf.fit(Xt_f, y_dt, verbose=False)
        auc_self = roc_auc_score(y_ds, clf.predict_proba(Xst_f)[:, 1])

        bt = self_test_df.copy().reset_index(drop=True)
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
            n_long = 0; n_short = 0
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
                    if sl: fee_e = FEE_TAKER
                    else:
                        filled = drift_fill(d_df, i + LOCKOUT, -direction)
                        fee_e = FEE_MAKER if filled else FEE_TAKER
                    trades.append({"net_pnl": ar - (FEE_TAKER + fee_e)})
                    i += LOCKOUT
                else:
                    i += 1
            n_total = len(trades)
            ps = sum(t["net_pnl"] for t in trades) if trades else 0
            wr = (sum(1 for t in trades if t["net_pnl"] > 0) / n_total) if n_total else 0
            per_th[str(dir_th)] = {"pnl": ps, "n_trades": n_total, "win_rate": wr,
                                    "n_long": n_long, "n_short": n_short}

        log.info(f"    AUC {auc_self:.3f}  TH0.58 {per_th['0.58']['pnl']:+.3f}% ({per_th['0.58']['n_trades']}t L{per_th['0.58']['n_long']}/S{per_th['0.58']['n_short']})")
        walk_results.append({"step": step_idx, "test_date": test_date,
                             "auc_self": float(auc_self), "per_th": per_th})

    return walk_results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("Path B: Causal regime-invariant features (rolling + ratios)")
    log.info("=" * 70)
    np.random.seed(42)

    log.info("\nBuilding Tardis...")
    tardis_train = build_split(DATES_TRAIN, log)
    tardis_val = build_split(DATES_VAL, log)
    feat_pre = get_feature_columns(tardis_train)
    tt_clean = tardis_train.dropna(subset=["target_volatility_300s", "target_return_3600s"])
    medians = tt_clean.reindex(columns=feat_pre).replace([np.inf, -np.inf], np.nan).median(numeric_only=True)

    SELF_ALL = [f"2026-04-{d:02d}" for d in range(21, 31)]
    self_dfs = {}
    log.info("Building Self...")
    for d in SELF_ALL:
        df = build_self_date_dataset(d, log, train_medians=medians)
        df = df.dropna(subset=["target_volatility_300s", "target_return_3600s"])
        self_dfs[d] = df
    sorted_self = sorted(self_dfs.keys())
    test_dates = sorted_self[1:]

    vol_target = "target_volatility_300s"; dir_target = "target_return_3600s"
    tardis_train.dropna(subset=[vol_target, dir_target], inplace=True)
    tardis_val.dropna(subset=[vol_target, dir_target], inplace=True)
    for d in self_dfs:
        self_dfs[d] = self_dfs[d].dropna(subset=[vol_target, dir_target])

    summaries = {}
    for method in ["ROLL_60", "ROLL_240", "ROLL_1440", "RATIOS"]:
        results = run_walk_forward(method, "default", tardis_train, tardis_val, self_dfs,
                                     sorted_self, test_dates, log)
        s = {"per_th": {}, "steps": results}
        for th in ["0.55", "0.58"]:
            pnls = [r["per_th"][th]["pnl"] for r in results]
            v = np.array(pnls)
            s["per_th"][th] = {
                "mean": float(v.mean()), "std": float(v.std()),
                "min": float(v.min()), "max": float(v.max()),
                "total": float(v.sum()), "positive": int((v > 0).sum()),
                "n": len(v),
            }
            log.info(f"  {method} TH {th}: mean {v.mean():+.3f}%/day  pos {(v>0).sum()}/{len(v)}")
        summaries[method] = s

    # Comparison
    print()
    print("=" * 90)
    print("Path B — CAUSAL feature methods (TH 0.58)")
    print("=" * 90)
    print(f"{'Method':<14} {'Mean PnL':<14} {'Std':<8} {'Positive':<10} {'Total':<10}")
    print("-" * 70)
    for method in summaries:
        d = summaries[method]["per_th"]["0.58"]
        print(f"{method:<14} {d['mean']:+.3f}%/day      {d['std']:.3f}    {d['positive']}/{d['n']:<8} {d['total']:+.3f}%")
    print()
    print("Reference baselines (TH 0.58):")
    print(f"  LEAKY (audit, with leak):       +1.074%/day  ❌ leak")
    print(f"  CAUSAL (audit, yesterday-norm):  -0.378%/day")
    print(f"  Path A (raw, no norm):           ?  (parallel run)")

    # Diagnosis
    print()
    print("=" * 90)
    print("DIAGNOSIS")
    print("=" * 90)
    best = max(summaries.items(), key=lambda kv: kv[1]["per_th"]["0.58"]["mean"])
    bn, bs = best
    bm = bs["per_th"]["0.58"]["mean"]
    print(f"\nBest causal method @ TH 0.58: {bn} → {bm:+.3f}%/day")
    if bm >= 0.3:
        print(f"  ✅ Causal regime-invariant features viable. mark36_v3 candidate.")
    elif bm >= 0:
        print(f"  🟡 Marginal positive. Direction modeling 한계 가능.")
    else:
        print(f"  ❌ All causal methods negative. Direction modeling 자체 한계.")

    out = {"approach": "Path B causal features", "summaries": summaries}
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/pathB_causal_features.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")
    log.info("Path B complete")


if __name__ == "__main__":
    main()
