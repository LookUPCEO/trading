"""Tardis 12 dates analysis with lagged features (Light Boost A)."""
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
from mark19.features.lagged import add_lagged_features


EXCHANGE = "bybit_tardis"
SYMBOL = "ETHUSDT"

DATES = [
    ("2025-04-01", "Recent 2025-04 (down)"),
    ("2025-01-01", "Jan 2025 (rest)"),
    ("2024-12-01", "Dec 2024 (top)"),
    ("2024-10-01", "Oct 2024 (volume peak)"),
    ("2024-09-01", "Sep 2024 (rally)"),
    ("2024-08-01", "Pre-Yen unwind"),
    ("2024-06-01", "Jun 2024 (sideways)"),
    ("2024-03-01", "Mar 2024 (bull)"),
    ("2023-10-01", "Oct 2023 (pre-ETF)"),
    ("2023-03-01", "Mar 2023 (pre-SVB)"),
    ("2022-11-01", "Pre-FTX"),
    ("2022-05-01", "Pre-LUNA"),
]

LAG_FEATURES = [
    "tr_trade_count_300s",
    "tr_trades_per_sec_300s",
    "tr_large_trade_count_300s",
    "tr_total_volume_300s",
    "liq_liq_count_300s",
    "liq_liq_notional_300s",
    "ob_obi_top5",
    "ob_spread",
    "dt_funding_rate",
    "dt_open_interest",
]
LAGS = [1, 5]


def compute_features_for_date(start, end, log):
    out = {}

    ob_raw = read_range("orderbook", EXCHANGE, SYMBOL, start, end)
    if len(ob_raw) > 100:
        ob_pw = compute_all_pointwise(ob_raw)
        ob_rs = compute_rolling_stats(ob_pw, "mid_price", [60, 300, 900])
        ob_op = compute_obi_persistence(ob_pw, "obi_top5", [60, 300])
        ob_pw_idx = ob_pw.set_index("timestamp") if "timestamp" in ob_pw.columns else ob_pw
        out["orderbook"] = pd.concat([ob_pw_idx, ob_rs, ob_op], axis=1).reset_index()

    tr_raw = read_range("trades", EXCHANGE, SYMBOL, start, end)
    if len(tr_raw) > 1000:
        tr_agg = aggregate_to_1s(tr_raw)
        tr_rolling = compute_trades_rolling(tr_agg, [60, 300, 900])
        out["trades"] = pd.merge(tr_agg, tr_rolling, on="timestamp", how="outer")

    liq_raw = read_range("liquidation", EXCHANGE, SYMBOL, start, end)
    if len(liq_raw) > 5:
        out["liquidation"] = compute_liquidation_features(liq_raw, [60, 300, 3600])

    dt_raw = read_range("derivative_ticker", EXCHANGE, SYMBOL, start, end)
    if len(dt_raw) > 100:
        dt = dt_raw.copy().sort_values("timestamp")
        dt["timestamp"] = pd.to_datetime(dt["timestamp"], utc=True).dt.floor("1s")
        dt = dt.drop_duplicates("timestamp", keep="last")
        out["derivative_ticker"] = dt

    return out


def integrate_features(features, log):
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
        keep = ["funding_rate", "predicted_funding_rate", "open_interest",
                "last_price", "index_price", "mark_price"]
        d = d[[c for c in keep if c in d.columns]]
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


def is_price_raw(col_name):
    for pat in PRICE_RAW_PATTERNS:
        if col_name == pat or col_name.startswith(pat):
            return True
    return False


def apply_bh_fdr(p_values, alpha=0.05):
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


def analyze_correlations(df, log):
    from scipy import stats as scistats

    df_1min = df.iloc[::60].copy().reset_index(drop=True)

    available_lag_features = [f for f in LAG_FEATURES if f in df_1min.columns]
    log.info(f"  Adding lagged features for {len(available_lag_features)} base features")
    df_1min = add_lagged_features(df_1min, available_lag_features, LAGS)
    log.info(f"  After lag: {len(df_1min.columns)} cols")

    target_cols = [c for c in df_1min.columns if c.startswith("target_")]
    feature_cols_all = [c for c in df_1min.columns if c != "timestamp" and not c.startswith("target_")]
    feature_cols = [c for c in feature_cols_all if not is_price_raw(c)]

    log.info(f"  1min rows: {len(df_1min)}, features: {len(feature_cols)}, targets: {len(target_cols)}")

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

    res["bonferroni_pass_pearson"] = res["pearson_p"] < (0.05 / len(res))
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
        end = start + timedelta(days=1)

        features = compute_features_for_date(start, end, log)
        if not features:
            continue

        integrated = integrate_features(features, log)
        if len(integrated) == 0:
            continue

        res = analyze_correlations(integrated, log)
        if len(res) == 0:
            continue

        all_results[date_str] = (label, res)

        n_tests = len(res)
        p05 = (res["pearson_p"] < 0.05).sum()
        log.info(f"  pairs: {n_tests}, p<0.05: {p05} ({p05/n_tests*100:.1f}%), FDR: {res['pearson_fdr_pass'].sum()}")

    n_processed = len(all_results)

    print()
    print("=" * 80)
    print(f"WITH LAGGED FEATURES — Cross-date Robustness ({n_processed} dates)")
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
        all_pos = all(r > 0 for r in rs)
        all_neg = all(r < 0 for r in rs)
        sign_consistent = all_pos or all_neg
        robust.append({
            "feature": feat,
            "target": targ,
            "n_dates": n_dates,
            "mean_r": np.mean(rs),
            "min_abs_r": min(abs(r) for r in rs) if rs else 0,
            "sign_consistent": sign_consistent,
            "rs": rs,
        })

    robust_df = pd.DataFrame(robust)

    sig_gold = robust_df[(robust_df["n_dates"] == n_processed) & (robust_df["sign_consistent"])]
    print(f"\n=== Gold standard ({n_processed}/{n_processed}): {len(sig_gold)} pairs ===")

    is_lag_feature = sig_gold["feature"].str.contains("_lag_|_change_|_pct_change_")
    new_signals = sig_gold[is_lag_feature]
    base_signals = sig_gold[~is_lag_feature]

    print(f"  Base features (existing): {len(base_signals)}")
    print(f"  Lagged/change features (NEW): {len(new_signals)}")

    if len(new_signals) > 0:
        print(f"\n=== NEW Lag/Change Signals (12/12 gold) Top 20 ===")
        top_new = new_signals.copy()
        top_new["abs_mean_r"] = top_new["mean_r"].abs()
        top_new = top_new.sort_values("abs_mean_r", ascending=False).head(20)

        print(f"{'feature':<46} {'target':<28} {'mean_r':>8} {'min|r|':>7}")
        for _, row in top_new.iterrows():
            feat = row['feature'][:45]
            targ = row['target'][:27]
            print(f"{feat:<46} {targ:<28} {row['mean_r']:+.4f} {row['min_abs_r']:>7.4f}")

    print(f"\n=== Overall Top 20 (12/12 gold) ===")
    top_all = sig_gold.copy()
    top_all["abs_mean_r"] = top_all["mean_r"].abs()
    top_all = top_all.sort_values("abs_mean_r", ascending=False).head(20)

    print(f"{'feature':<46} {'target':<28} {'mean_r':>8} {'min|r|':>7}")
    for _, row in top_all.iterrows():
        feat = row['feature'][:45]
        targ = row['target'][:27]
        print(f"{feat:<46} {targ:<28} {row['mean_r']:+.4f} {row['min_abs_r']:>7.4f}")

    out_path = Path("data/analysis_results")
    out_path.mkdir(exist_ok=True, parents=True)

    if len(robust_df) > 0:
        rs = robust_df.copy()
        rs["rs"] = rs["rs"].apply(lambda x: ",".join(f"{r:.4f}" for r in x))
        rs.to_csv(out_path / "tardis12_robust_with_lag.csv", index=False)

    log.info(f"Saved: {out_path}/tardis12_robust_with_lag.csv")
    print(f"\nProcessed {n_processed} / {len(DATES)} dates")


if __name__ == "__main__":
    main()
