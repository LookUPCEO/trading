"""Regime analysis: per-day metrics + train vs test distribution comparison.

Question: Is test data within the training distribution, or OOD?

Datasets:
  Self bybit (4/21-5/9, 19d) — used for sido36 walk-forward (train + test rotation)
  Tardis trial (4/28-5/7, 10d) — used for Track C/D test (Tardis source)
  Tardis original 2022-2024 — INACCESSIBLE (macOS permission issue)

Per-day regime metrics (5):
  daily_return     : (last_mid - first_mid) / first_mid (signed % daily move)
  daily_vol        : std of 1-min mid returns
  mean_spread_bp   : avg ob_spread / mid (bp)
  mean_depth_top5  : avg (ob_bid_depth_5 + ob_ask_depth_5)
  trades_per_sec   : len(trades) / 86400

Cuts:
  Train portion (sido36 walk-forward): Self 4/21-4/29 (9d, accumulating)
  Test portion (sido36 walk-forward):  Self 4/22-4/30 (9d step-by-step)
  Track C test:                        Tardis trial 4/29-5/7 (sub-window)

OOD analysis:
  - Per-feature: z-score test day vs train(prior days) mean+std
  - Multivariate: Mahalanobis distance test day vs train ellipsoid
  - Flag: any |z| > 2 = "outside" 1 metric
"""
import sys, logging, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.storage import read_range


def compute_day_metrics(date_str, exchange, log):
    """Compute regime metrics for one day. Returns dict or None on data miss."""
    start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1) - timedelta(seconds=1)

    try:
        df_ob = read_range("orderbook", exchange, "ETHUSDT", start, end)
        df_tr = read_range("trades", exchange, "ETHUSDT", start, end)
    except Exception as e:
        log.warning(f"  {date_str} ({exchange}): read fail {e}")
        return None
    if df_ob.empty:
        log.warning(f"  {date_str} ({exchange}): no OB data")
        return None

    # Identify mid column
    if "bid_0_price" in df_ob.columns and "ask_0_price" in df_ob.columns:
        df_ob["mid"] = (df_ob["bid_0_price"] + df_ob["ask_0_price"]) / 2
    elif "mid" in df_ob.columns:
        pass
    else:
        return None
    df_ob = df_ob.dropna(subset=["mid"])
    if len(df_ob) < 100:
        return None

    # Resample to 1-min for return/vol calc
    ts_col = next((c for c in df_ob.columns if "timestamp" in c.lower()), None)
    if ts_col:
        df_ob[ts_col] = pd.to_datetime(df_ob[ts_col], utc=True)
        df_1m = df_ob.set_index(ts_col)["mid"].resample("1min").last().ffill()
    else:
        df_1m = df_ob["mid"]

    daily_return = (float(df_1m.iloc[-1]) - float(df_1m.iloc[0])) / float(df_1m.iloc[0])
    rets_1m = df_1m.pct_change().dropna()
    daily_vol = float(rets_1m.std())

    # Spread (bp)
    if "bid_0_price" in df_ob.columns and "ask_0_price" in df_ob.columns:
        spread = (df_ob["ask_0_price"] - df_ob["bid_0_price"]).clip(lower=0)
        mid_arr = df_ob["mid"].clip(lower=1e-9)
        spread_bp = (spread / mid_arr * 10000).mean()
    else:
        spread_bp = float("nan")

    # Depth top 5 (sum bid + ask)
    depth_cols_b = [c for c in df_ob.columns if c.startswith("bid_") and "_size" in c]
    depth_cols_a = [c for c in df_ob.columns if c.startswith("ask_") and "_size" in c]
    if depth_cols_b and depth_cols_a:
        # Take first 5 levels
        bb = sum(df_ob[c] for c in sorted(depth_cols_b)[:5])
        aa = sum(df_ob[c] for c in sorted(depth_cols_a)[:5])
        mean_depth = float((bb + aa).mean())
    else:
        mean_depth = float("nan")

    # Trades per sec
    if not df_tr.empty:
        trades_per_sec = len(df_tr) / 86400.0
        # Buy/sell imbalance
        if "side" in df_tr.columns:
            buys = (df_tr["side"] == "Buy").sum()
            sells = (df_tr["side"] == "Sell").sum()
            tot = buys + sells
            buy_imbalance = (buys - sells) / max(tot, 1)
        else:
            buy_imbalance = 0
    else:
        trades_per_sec = 0.0
        buy_imbalance = 0

    return {
        "date": date_str, "exchange": exchange,
        "daily_return_pct": daily_return * 100,
        "daily_vol_bps": daily_vol * 10000,
        "mean_spread_bp": float(spread_bp),
        "mean_depth_top5": mean_depth,
        "trades_per_sec": trades_per_sec,
        "buy_imbalance": buy_imbalance,
        "n_ob_rows": len(df_ob),
    }


def zscore(x, train_mean, train_std):
    if train_std == 0 or np.isnan(train_std):
        return 0
    return (x - train_mean) / train_std


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("Regime analysis: train vs test distribution")
    log.info("=" * 70)

    # === Compute per-day metrics ===
    self_days = [f"2026-04-{d:02d}" for d in range(21, 31)] + [f"2026-05-{d:02d}" for d in range(1, 10)]
    trial_days = [f"2026-04-{d:02d}" for d in range(28, 31)] + [f"2026-05-{d:02d}" for d in range(1, 8)]

    log.info(f"\n=== Self bybit ({len(self_days)} days) ===")
    self_metrics = []
    for d in self_days:
        m = compute_day_metrics(d, "bybit", log)
        if m: self_metrics.append(m); log.info(f"  {d}: ret {m['daily_return_pct']:+.3f}%  vol {m['daily_vol_bps']:.1f}bp  spread {m['mean_spread_bp']:.2f}bp  trades/s {m['trades_per_sec']:.2f}")

    log.info(f"\n=== Tardis trial ({len(trial_days)} days) ===")
    trial_metrics = []
    for d in trial_days:
        m = compute_day_metrics(d, "bybit_tardis_trial", log)
        if m: trial_metrics.append(m); log.info(f"  {d}: ret {m['daily_return_pct']:+.3f}%  vol {m['daily_vol_bps']:.1f}bp  spread {m['mean_spread_bp']:.2f}bp  trades/s {m['trades_per_sec']:.2f}")

    if not self_metrics:
        log.error("No Self data — abort"); return

    # === Define train / test splits ===
    # sido36 walk-forward: train = Self 4/21-4/29, test = Self 4/22-4/30 (rotating)
    # For OOD, treat all 9 train days (4/21-4/29) as "train pool", each walk-forward test day as "test"
    df_self = pd.DataFrame(self_metrics).sort_values("date").reset_index(drop=True)
    df_trial = pd.DataFrame(trial_metrics).sort_values("date").reset_index(drop=True)

    metric_cols = ["daily_return_pct", "daily_vol_bps", "mean_spread_bp", "mean_depth_top5", "trades_per_sec"]

    print()
    print("=" * 110)
    print("Per-day metrics")
    print("=" * 110)
    print(f"{'Date':<12} {'Source':<8} {'Return%':<10} {'Vol bp':<10} {'Sprd bp':<10} {'Depth top5':<12} {'Trades/s':<10}")
    print("-" * 90)
    for _, r in df_self.iterrows():
        print(f"{r['date']:<12} {'self':<8} {r['daily_return_pct']:+.3f}    {r['daily_vol_bps']:.1f}      {r['mean_spread_bp']:.3f}     {r['mean_depth_top5']:.1f}        {r['trades_per_sec']:.2f}")
    print("- - -")
    for _, r in df_trial.iterrows():
        print(f"{r['date']:<12} {'trial':<8} {r['daily_return_pct']:+.3f}    {r['daily_vol_bps']:.1f}      {r['mean_spread_bp']:.3f}     {r['mean_depth_top5']:.1f}        {r['trades_per_sec']:.2f}")

    # Train pool: Self 4/21-4/29 (9 days)
    train_dates = self_days[:9]
    test_dates_self = self_days[1:10]   # walk-forward test 4/22-4/30
    test_dates_trial = trial_days        # Track C/D test pool

    train_df = df_self[df_self["date"].isin(train_dates)]
    test_self_df = df_self[df_self["date"].isin(test_dates_self)]
    test_trial_df = df_trial[df_trial["date"].isin(test_dates_trial)]

    print()
    print("=" * 110)
    print("Train (Self 4/21-4/29, 9d) distribution + test OOD analysis")
    print("=" * 110)
    print(f"{'Metric':<22} {'Train mean':<14} {'Train std':<14} {'Train min':<14} {'Train max':<14}")
    print("-" * 80)
    train_stats = {}
    for c in metric_cols:
        m = train_df[c].mean(); s = train_df[c].std(); mn = train_df[c].min(); mx = train_df[c].max()
        train_stats[c] = {"mean": m, "std": s, "min": mn, "max": mx}
        print(f"{c:<22} {m:+.4f}      {s:.4f}        {mn:+.4f}      {mx:+.4f}")

    # === OOD per test day ===
    print()
    print("=" * 110)
    print("Test days OOD analysis: |z|>2 means OOD on that metric")
    print("=" * 110)

    def report_test(title, df_test):
        print(f"\n{title}")
        header = f"{'Date':<12}"
        for c in metric_cols: header += f"  z({c[:8]:<8})"
        header += "  flag_ood"
        print(header)
        print("-" * (12 + 12*len(metric_cols) + 12))
        ood_count_per_metric = {c: 0 for c in metric_cols}
        n_total = len(df_test)
        for _, r in df_test.iterrows():
            line = f"{r['date']:<12}"
            ood = []
            for c in metric_cols:
                z = zscore(r[c], train_stats[c]["mean"], train_stats[c]["std"])
                line += f"  {z:+.2f}     "
                if abs(z) > 2:
                    ood.append(c[:8])
                    ood_count_per_metric[c] += 1
            flag = "OOD: " + ",".join(ood) if ood else "in-dist"
            line += f"  {flag}"
            print(line)
        print(f"\n  OOD events per metric (over {n_total} days): " +
              ", ".join(f"{c[:8]}={n}" for c, n in ood_count_per_metric.items()))
        return ood_count_per_metric

    ood_self = report_test("== Self test (4/22-4/30, walk-forward) ==", test_self_df)
    ood_trial = report_test("== Trial test (4/29-5/7, Track C/D pool) ==", test_trial_df)

    # === Multivariate distance (Mahalanobis-like, using train cov) ===
    print()
    print("=" * 110)
    print("Multivariate distance to train center (Euclidean of z-scores)")
    print("=" * 110)
    def md(row):
        zs = [zscore(row[c], train_stats[c]["mean"], train_stats[c]["std"]) for c in metric_cols]
        return float(np.sqrt(sum(z*z for z in zs)))
    test_self_df = test_self_df.copy()
    test_self_df["mvz"] = test_self_df.apply(md, axis=1)
    test_trial_df = test_trial_df.copy()
    test_trial_df["mvz"] = test_trial_df.apply(md, axis=1)

    print(f"{'Date':<12} {'Source':<8} {'mvz':<10}")
    for _, r in test_self_df.iterrows():
        print(f"{r['date']:<12} {'self':<8} {r['mvz']:.3f}  {'(in-dist <2)' if r['mvz']<2 else '(OOD ≥2)' if r['mvz']<3 else '(strong OOD ≥3)'}")
    for _, r in test_trial_df.iterrows():
        print(f"{r['date']:<12} {'trial':<8} {r['mvz']:.3f}  {'(in-dist <2)' if r['mvz']<2 else '(OOD ≥2)' if r['mvz']<3 else '(strong OOD ≥3)'}")

    # === Diagnosis ===
    print()
    print("=" * 110)
    print("DIAGNOSIS")
    print("=" * 110)
    n_self_test = len(test_self_df)
    n_trial_test = len(test_trial_df)
    n_self_ood = (test_self_df["mvz"] >= 2).sum()
    n_trial_ood = (test_trial_df["mvz"] >= 2).sum()

    print(f"\nSelf test (walk-forward) OOD: {n_self_ood}/{n_self_test}")
    print(f"Trial test (Track C/D)   OOD: {n_trial_ood}/{n_trial_test}")

    if n_self_ood / max(n_self_test, 1) > 0.5:
        print(f"\n  ❌ Most Self test days OOD → train data was 매우 다른 regime")
        print(f"  → 데이터 다양성 부족 (Case B)")
    elif n_self_ood / max(n_self_test, 1) > 0.2:
        print(f"\n  🟡 일부 Self test 가 OOD → 부분 일치 (Case C)")
        print(f"  → 더 다양한 시기 학습 필요")
    else:
        print(f"\n  ✅ Self test 대부분 in-distribution")
        print(f"  → 모델 능력 또는 본질 한계 (Case A)")
        print(f"  → 데이터 양 부족 가능 (학습 9d, 추가 학습 후 한계 향상 측정 필요)")

    if n_trial_ood / max(n_trial_test, 1) > 0.5:
        print(f"\n  ❌ Trial test (Track C/D) 대부분 OOD → 다른 source 사용 영향")
    elif n_trial_ood / max(n_trial_test, 1) > 0.2:
        print(f"\n  🟡 Trial test 일부 OOD → cross-exchange regime drift")

    out = {
        "self_metrics": self_metrics,
        "trial_metrics": trial_metrics,
        "train_stats": train_stats,
        "test_self_mvz": test_self_df[["date","mvz"]].to_dict(orient="records"),
        "test_trial_mvz": test_trial_df[["date","mvz"]].to_dict(orient="records"),
        "ood_counts_self": {k: int(v) for k, v in ood_self.items()},
        "ood_counts_trial": {k: int(v) for k, v in ood_trial.items()},
        "summary": {
            "n_self_ood": int(n_self_ood),
            "n_self_total": int(n_self_test),
            "n_trial_ood": int(n_trial_ood),
            "n_trial_total": int(n_trial_test),
        },
        "note": "Tardis original 2022-2024 inaccessible (macOS permission). Train pool = Self 4/21-4/29.",
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/regime_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")


if __name__ == "__main__":
    main()
