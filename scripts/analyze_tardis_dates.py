"""
Tardis 4 dates: feature compute + Stage 2.5 analysis per date.

Computes order book, trades, liquidation features on Tardis-converted data.
Then runs detrended correlation analysis on each date.
Compares signals across 4 dates to identify robust patterns.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.storage import read_range
from mark19.features.orderbook import compute_all_pointwise
from mark19.features.orderbook_timeseries import compute_rolling_stats, compute_obi_persistence
from mark19.features.trades import aggregate_to_1s, compute_rolling_features as compute_trades_rolling
from mark19.features.liquidation import compute_liquidation_features


EXCHANGE = "bybit_tardis"
SYMBOL = "ETHUSDT"

DATES = [
    ("2025-04-01", "Recent 2025"),
    ("2024-08-01", "Pre-Yen unwind"),
    ("2022-11-01", "Pre-FTX"),
    ("2022-05-01", "Pre-LUNA"),
]


def compute_features_for_date(start: datetime, end: datetime, log) -> dict:
    out = {}

    log.info("  Computing orderbook features")
    ob_raw = read_range("orderbook", EXCHANGE, SYMBOL, start, end)
    if len(ob_raw) > 100:
        ob_pw = compute_all_pointwise(ob_raw)
        ob_rs = compute_rolling_stats(ob_pw, "mid_price", [60, 300, 900])
        ob_op = compute_obi_persistence(ob_pw, "obi_top5", [60, 300])
        ob_pw_idx = ob_pw.set_index("timestamp") if "timestamp" in ob_pw.columns else ob_pw
        ob_combined = pd.concat([ob_pw_idx, ob_rs, ob_op], axis=1).reset_index()
        out["orderbook"] = ob_combined
        log.info(f"    {len(ob_combined)} rows × {len(ob_combined.columns)} cols")

    log.info("  Computing trades features")
    tr_raw = read_range("trades", EXCHANGE, SYMBOL, start, end)
    if len(tr_raw) > 1000:
        tr_agg = aggregate_to_1s(tr_raw)
        tr_rolling = compute_trades_rolling(tr_agg, [60, 300, 900])
        tr_combined = pd.merge(tr_agg, tr_rolling, on="timestamp", how="outer")
        out["trades"] = tr_combined
        log.info(f"    {len(tr_combined)} rows × {len(tr_combined.columns)} cols")

    log.info("  Computing liquidation features")
    liq_raw = read_range("liquidation", EXCHANGE, SYMBOL, start, end)
    if len(liq_raw) > 5:
        liq_feat = compute_liquidation_features(liq_raw, [60, 300, 3600])
        out["liquidation"] = liq_feat
        log.info(f"    {len(liq_feat)} rows × {len(liq_feat.columns)} cols")

    log.info("  Computing derivative_ticker features")
    dt_raw = read_range("derivative_ticker", EXCHANGE, SYMBOL, start, end)
    if len(dt_raw) > 100:
        dt = dt_raw.copy().sort_values("timestamp")
        dt["timestamp"] = pd.to_datetime(dt["timestamp"], utc=True).dt.floor("1s")
        dt = dt.drop_duplicates("timestamp", keep="last")
        out["derivative_ticker"] = dt
        log.info(f"    {len(dt)} rows × {len(dt.columns)} cols")

    return out


def integrate_features(features: dict, log) -> pd.DataFrame:
    if "orderbook" not in features or len(features["orderbook"]) == 0:
        return pd.DataFrame()

    base = features["orderbook"].copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True).dt.floor("1s")
    base = base.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")

    full_idx = pd.date_range(base.index.min(), base.index.max(), freq="1s", tz="UTC")
    combined = base.reindex(full_idx)
    combined.columns = [f"ob_{c}" for c in combined.columns]

    if "trades" in features:
        t = features["trades"].copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True).dt.floor("1s")
        t = t.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")
        t = t.reindex(full_idx)
        t.columns = [f"tr_{c}" for c in t.columns]
        combined = combined.join(t)

    if "liquidation" in features:
        l = features["liquidation"].copy()
        l["timestamp"] = pd.to_datetime(l["timestamp"], utc=True).dt.floor("1s")
        l = l.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")
        l = l.reindex(full_idx, fill_value=0)
        l.columns = [f"liq_{c}" for c in l.columns]
        combined = combined.join(l)

    if "derivative_ticker" in features:
        d = features["derivative_ticker"].copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True).dt.floor("1s")
        d = d.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")
        d = d.reindex(full_idx, method="ffill", limit=300)
        keep_cols = ["funding_rate", "predicted_funding_rate", "open_interest",
                     "last_price", "index_price", "mark_price"]
        d = d[[c for c in keep_cols if c in d.columns]]
        d.columns = [f"dt_{c}" for c in d.columns]
        combined = combined.join(d)

    if "ob_mid_price" not in combined.columns:
        return pd.DataFrame()

    mid = combined["ob_mid_price"]
    for N in [300, 900, 3600]:
        min_p = max(N // 2, 1)
        future_mid = mid.shift(-N)
        combined[f"target_return_{N}s"] = (future_mid - mid) / mid * 100
        combined[f"target_volatility_{N}s"] = mid.rolling(N, min_periods=min_p).std().shift(-(N-1))
        combined[f"target_max_drawdown_{N}s"] = (mid.rolling(N, min_periods=min_p).min().shift(-(N-1)) - mid) / mid * 100
        combined[f"target_max_runup_{N}s"] = (mid.rolling(N, min_periods=min_p).max().shift(-(N-1)) - mid) / mid * 100

    return combined.reset_index().rename(columns={"index": "timestamp"})


PRICE_RAW_PATTERNS = [
    "ob_mid_price",
    "ob_mid_price_mean_",
    "ob_mid_price_std_",
    "tr_vwap",
    "dt_last_price",
    "dt_index_price",
    "dt_mark_price",
]


def is_price_raw(col_name: str) -> bool:
    for pat in PRICE_RAW_PATTERNS:
        if col_name == pat or col_name.startswith(pat):
            return True
    return False


def apply_bh_fdr(p_values: pd.Series, alpha: float = 0.05) -> pd.Series:
    n = len(p_values)
    if n == 0:
        return pd.Series([], dtype=bool)
    sorted_idx = p_values.sort_values().index
    sorted_p = p_values.loc[sorted_idx].values
    ranks = np.arange(1, n + 1)
    thresholds = (ranks / n) * alpha
    passes = sorted_p <= thresholds
    if not passes.any():
        return pd.Series(False, index=p_values.index)
    max_k = ranks[passes].max()
    result = pd.Series(False, index=p_values.index)
    pass_idx = sorted_idx[:max_k]
    result.loc[pass_idx] = True
    return result


def analyze_correlations(df: pd.DataFrame, log) -> pd.DataFrame:
    from scipy import stats as scistats

    df_1min = df.iloc[::60].copy().reset_index(drop=True)

    target_cols = [c for c in df_1min.columns if c.startswith("target_")]
    feature_cols_all = [c for c in df_1min.columns if c != "timestamp" and not c.startswith("target_")]
    feature_cols = [c for c in feature_cols_all if not is_price_raw(c)]

    log.info(f"  1min rows: {len(df_1min)}")
    log.info(f"  features: {len(feature_cols_all)} → {len(feature_cols)} after price-raw drop")
    log.info(f"  targets: {len(target_cols)}")

    results = []
    for feat in feature_cols:
        for targ in target_cols:
            valid = df_1min[[feat, targ]].dropna()
            if len(valid) < 50:
                continue
            if valid[feat].std() < 1e-10 or valid[targ].std() < 1e-10:
                continue
            try:
                pearson_r, pearson_p = scistats.pearsonr(valid[feat], valid[targ])
                spearman_r, spearman_p = scistats.spearmanr(valid[feat], valid[targ])
                results.append({
                    "feature": feat,
                    "target": targ,
                    "n": len(valid),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                })
            except Exception:
                continue

    res = pd.DataFrame(results)
    if len(res) == 0:
        return res

    n_tests = len(res)
    res["bonferroni_pass_pearson"] = res["pearson_p"] < (0.05 / n_tests)
    res["pearson_fdr_pass"] = apply_bh_fdr(res["pearson_p"])

    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    all_results = {}

    for date_str, label in DATES:
        log.info(f"")
        log.info(f"=" * 70)
        log.info(f"Processing {date_str} ({label})")
        log.info(f"=" * 70)

        y, m, d = map(int, date_str.split("-"))
        start = datetime(y, m, d, tzinfo=timezone.utc)
        end = datetime(y, m, d, tzinfo=timezone.utc) + timedelta(days=1)

        features = compute_features_for_date(start, end, log)
        if not features:
            log.warning(f"  No features for {date_str}")
            continue

        log.info(f"  Integrating to 1s grid")
        integrated = integrate_features(features, log)
        if len(integrated) == 0:
            log.warning(f"  Integration failed")
            continue
        log.info(f"  integrated: {len(integrated)} rows × {len(integrated.columns)} cols")

        log.info(f"  Running Stage 2.5 detrended correlation")
        res = analyze_correlations(integrated, log)
        if len(res) == 0:
            log.warning(f"  No valid pairs")
            continue

        all_results[date_str] = (label, res)

        n_tests = len(res)
        p05 = (res["pearson_p"] < 0.05).sum()
        log.info(f"  {date_str} pairs: {n_tests}, p<0.05: {p05} ({p05/n_tests*100:.1f}%), FDR: {res['pearson_fdr_pass'].sum()}, Bonf: {res['bonferroni_pass_pearson'].sum()}")

    print()
    print("=" * 80)
    print("PER-DATE TOP 15 by |Pearson r|")
    print("=" * 80)

    for date_str, (label, res) in all_results.items():
        print(f"\n--- {date_str} ({label}) ---")
        res_sorted = res.copy()
        res_sorted["abs_r"] = res_sorted["pearson_r"].abs()
        top = res_sorted.sort_values("abs_r", ascending=False).head(15)

        print(f"{'feature':<42} {'target':<28} {'r':>7} {'FDR':>5}")
        for _, row in top.iterrows():
            feat = row['feature'][:41]
            targ = row['target'][:27]
            fdr = "Y" if row['pearson_fdr_pass'] else " "
            print(f"{feat:<42} {targ:<28} {row['pearson_r']:+.4f} {fdr:>5}")

    print()
    print("=" * 80)
    print("CROSS-DATE ROBUSTNESS")
    print("=" * 80)

    pair_appearances = {}
    for date_str, (label, res) in all_results.items():
        sig = res[res["pearson_fdr_pass"]]
        for _, row in sig.iterrows():
            key = (row["feature"], row["target"])
            if key not in pair_appearances:
                pair_appearances[key] = []
            pair_appearances[key].append((date_str, row["pearson_r"]))

    robust = []
    for (feat, targ), apps in pair_appearances.items():
        n_dates = len(apps)
        rs = [r for _, r in apps]
        mean_r = np.mean(rs)
        all_pos = all(r > 0 for r in rs)
        all_neg = all(r < 0 for r in rs)
        sign_consistent = all_pos or all_neg
        robust.append({
            "feature": feat,
            "target": targ,
            "n_dates": n_dates,
            "mean_r": mean_r,
            "sign_consistent": sign_consistent,
            "rs": rs,
            "dates": [d for d, _ in apps],
        })

    robust_df = pd.DataFrame(robust)

    if len(robust_df) > 0:
        all_4 = robust_df[(robust_df["n_dates"] == 4) & (robust_df["sign_consistent"])]
        print(f"\n=== Pairs significant in ALL 4 dates with consistent sign ===")
        print(f"Total: {len(all_4)}")
        if len(all_4) > 0:
            all_4_sorted = all_4.copy()
            all_4_sorted["abs_mean_r"] = all_4_sorted["mean_r"].abs()
            all_4_sorted = all_4_sorted.sort_values("abs_mean_r", ascending=False).head(20)
            print(f"\n{'feature':<42} {'target':<28} {'mean_r':>8} {'rs':>40}")
            for _, row in all_4_sorted.iterrows():
                feat = row['feature'][:41]
                targ = row['target'][:27]
                rs_str = ", ".join(f"{r:+.2f}" for r in row['rs'])
                print(f"{feat:<42} {targ:<28} {row['mean_r']:+.4f} [{rs_str}]")

        in_3 = robust_df[(robust_df["n_dates"] == 3) & (robust_df["sign_consistent"])]
        print(f"\n=== Pairs in 3 dates (consistent sign) ===")
        print(f"Total: {len(in_3)}")
        if len(in_3) > 0:
            top_3 = in_3.copy()
            top_3["abs_mean_r"] = top_3["mean_r"].abs()
            top_3 = top_3.sort_values("abs_mean_r", ascending=False).head(15)
            for _, row in top_3.iterrows():
                feat = row['feature'][:41]
                targ = row['target'][:27]
                print(f"  {feat:<42} {targ:<28} {row['mean_r']:+.4f}  dates={row['dates']}")

    print()
    print("=" * 80)
    print("ROBUST SIGNAL SUMMARY BY CATEGORY")
    print("=" * 80)

    def get_prefix(col):
        for p in ["ob_", "tr_", "liq_", "dt_"]:
            if col.startswith(p):
                return p[:-1]
        return "other"

    if len(robust_df) > 0 and len(all_4) > 0:
        all_4_summary = all_4.copy()
        all_4_summary["prefix"] = all_4_summary["feature"].apply(get_prefix)
        cat_summary = all_4_summary.groupby("prefix").agg(
            count=("feature", "size"),
            mean_abs_r=("mean_r", lambda x: x.abs().mean()),
            max_abs_r=("mean_r", lambda x: x.abs().max()),
        )
        print("\nFeatures with all-4-date robust signals:")
        print(cat_summary.to_string())

    out_path = Path("data/analysis_results")
    out_path.mkdir(exist_ok=True, parents=True)

    for date_str, (label, res) in all_results.items():
        res.to_parquet(out_path / f"tardis_correlations_{date_str}.parquet")

    if len(robust_df) > 0:
        robust_save = robust_df.copy()
        robust_save["rs"] = robust_save["rs"].apply(lambda x: ",".join(f"{r:.4f}" for r in x))
        robust_save["dates"] = robust_save["dates"].apply(lambda x: ",".join(x))
        robust_save.to_csv(out_path / "tardis_robust_signals.csv", index=False)

    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
