"""시도 33: Cross-exchange spread analysis (Bybit vs Binance vs OKX)."""
import sys, logging, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("시도 33: Cross-Exchange Spread Analysis")
    log.info("=" * 70)

    # ---- 1. Load cross_exchange_prices ----
    cx_dir = Path("/Users/dohun/Desktop/Mark/mark19/data/cross_exchange_prices/combined/ETHUSDT")
    files = sorted(cx_dir.glob("*.parquet"))
    log.info(f"\nFiles: {len(files)} ({files[0].stem if files else 'none'} ~ {files[-1].stem if files else 'none'})")

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            log.warning(f"  Skip {f.stem}: {e}")
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    log.info(f"Total rows: {len(df)}")
    if len(df) == 0:
        return

    log.info(f"Columns: {list(df.columns)}")

    # Identify mid columns. Likely: bybit_mid, binance_mid, okx_mid
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Schema: {ex}_eth_usd (single price columns), no bid/ask
    exchanges = ["bybit", "binance", "okx"]
    for ex in exchanges:
        src = f"{ex}_eth_usd"
        if src in df.columns:
            df[f"{ex}_mid"] = df[src]

    have_ex = [ex for ex in exchanges if f"{ex}_mid" in df.columns and df[f"{ex}_mid"].notna().any()]
    log.info(f"Available exchange mids: {have_ex}")

    if len(have_ex) < 2:
        log.error(f"Need at least 2 exchanges with mid prices. Got {len(have_ex)}.")
        # show sample
        log.info(f"Sample row: {df.iloc[0].to_dict()}")
        return

    # ---- 2. Spread distribution per pair ----
    print()
    print("=" * 80)
    print("SPREAD DISTRIBUTION (per pair, % of mid)")
    print("=" * 80)

    pairs = []
    for i, e1 in enumerate(have_ex):
        for e2 in have_ex[i+1:]:
            pairs.append((e1, e2))

    spread_stats = {}
    for e1, e2 in pairs:
        m1 = df[f"{e1}_mid"]; m2 = df[f"{e2}_mid"]
        # filter valid
        valid = m1.notna() & m2.notna() & (m1 > 0) & (m2 > 0)
        sp = ((m1 - m2) / ((m1 + m2) / 2) * 100)[valid]  # % spread, signed
        abs_sp = sp.abs()
        n = len(sp)
        if n == 0: continue

        stats = {
            "n": int(n),
            "mean_signed": float(sp.mean()),
            "std": float(sp.std()),
            "abs_mean": float(abs_sp.mean()),
            "abs_max": float(abs_sp.max()),
            "abs_q50": float(abs_sp.quantile(0.5)),
            "abs_q75": float(abs_sp.quantile(0.75)),
            "abs_q95": float(abs_sp.quantile(0.95)),
            "abs_q99": float(abs_sp.quantile(0.99)),
        }
        # Threshold counts
        for thr_label, thr in [("0.01", 0.01), ("0.05", 0.05), ("0.10", 0.10), ("0.20", 0.20), ("0.50", 0.50)]:
            stats[f"above_{thr_label}pct"] = float((abs_sp > thr).mean())
        spread_stats[f"{e1}_vs_{e2}"] = stats

    print(f"\n{'Pair':<22} {'N':<10} {'Mean':<10} {'Std':<10} {'|S| q50':<10} {'|S| q95':<10} {'|S| q99':<10} {'|S| max':<10}")
    print("-" * 100)
    for pair, s in spread_stats.items():
        print(f"{pair:<22} {s['n']:<10} {s['mean_signed']:<+10.5f} {s['std']:<10.5f} {s['abs_q50']:<10.5f} {s['abs_q95']:<10.5f} {s['abs_q99']:<10.5f} {s['abs_max']:<10.5f}")

    print(f"\n{'Pair':<22} {'>0.01%':<10} {'>0.05%':<10} {'>0.10%':<10} {'>0.20%':<10} {'>0.50%':<10}")
    print("-" * 80)
    for pair, s in spread_stats.items():
        print(f"{pair:<22} {s['above_0.01pct']*100:<10.2f} {s['above_0.05pct']*100:<10.2f} {s['above_0.10pct']*100:<10.2f} {s['above_0.20pct']*100:<10.2f} {s['above_0.50pct']*100:<10.2f}")

    # ---- 3. Mean reversion analysis ----
    print()
    print("=" * 80)
    print("MEAN REVERSION (autocorr lag1, signed spread)")
    print("=" * 80)
    print("Lag 1 reflects sample interval (~1s for self collector). High autocorr = persistent.")
    for e1, e2 in pairs:
        m1 = df[f"{e1}_mid"]; m2 = df[f"{e2}_mid"]
        valid = m1.notna() & m2.notna() & (m1 > 0) & (m2 > 0)
        sp = ((m1 - m2) / ((m1 + m2) / 2) * 100)[valid]
        ac1 = sp.autocorr(1)
        ac10 = sp.autocorr(10)
        ac60_str = f"{sp.autocorr(60):.3f}" if len(sp) > 100 else "N/A"
        print(f"  {e1}_vs_{e2}: AC lag1 {ac1:.3f}  lag10 {ac10:.3f}  lag60 {ac60_str}")

    # ---- 4. Funding spread analysis ----
    print()
    print("=" * 80)
    print("FUNDING SPREAD (bybit / binance / okx 8h funding)")
    print("=" * 80)

    fc_dir = Path("/Users/dohun/Desktop/Mark/mark19/data/funding_current/combined/ETHUSDT")
    fc_files = sorted(fc_dir.glob("*.parquet"))
    if fc_files:
        fc_dfs = [pd.read_parquet(f) for f in fc_files]
        fc = pd.concat(fc_dfs, ignore_index=True)
        log.info(f"Funding snapshots: {len(fc)}")

        # Take last snapshot per next_time per exchange to extract paid rates
        fc["timestamp"] = pd.to_datetime(fc["timestamp"], utc=True)
        for ex in ["bybit", "binance", "okx"]:
            fr_col = f"{ex}_funding"
            nt_col = f"{ex}_next_time"
            if fr_col in fc.columns and nt_col in fc.columns:
                fc[nt_col] = pd.to_datetime(fc[nt_col], utc=True)

        # Distribution
        for ex in ["bybit", "binance", "okx"]:
            fr_col = f"{ex}_funding"
            if fr_col in fc.columns:
                rates = fc[fr_col].dropna() * 100  # to %
                if len(rates) > 0:
                    print(f"  {ex}: mean {rates.mean():+.4f}%  std {rates.std():.4f}  range [{rates.min():+.4f}, {rates.max():+.4f}]")

        # Funding spread bybit vs binance vs okx (use predicted rate snapshots, all rows)
        print()
        print("Funding rate spreads (snapshot-level):")
        for e1, e2 in pairs:
            f1 = fc.get(f"{e1}_funding")
            f2 = fc.get(f"{e2}_funding")
            if f1 is None or f2 is None: continue
            valid = f1.notna() & f2.notna()
            diff = (f1[valid] - f2[valid]) * 100  # to %
            if len(diff) > 0:
                print(f"  {e1}_vs_{e2} funding diff: mean {diff.mean():+.4f}%  std {diff.std():.4f}  abs max {diff.abs().max():.4f}%")

    # ---- 5. Statistical arbitrage potential ----
    print()
    print("=" * 80)
    print("STATISTICAL ARBITRAGE POTENTIAL")
    print("=" * 80)
    print("\nBreakeven analysis (round-trip taker fee = 2 × 0.055% = 0.110%, maker = -0.025% × 2 = -0.05%):")
    print("\nRequired |spread| to profit (entry+exit):")
    print(f"  Both taker:    {0.110:.3f}% per leg")
    print(f"  Mixed (1 mkt): {0.030:.3f}%  (taker entry + maker exit)")
    print(f"  Both maker:    {-0.050:.3f}% (rebate, profit even at zero spread!)")

    print()
    print("Rough opportunity count (per pair):")
    for pair, s in spread_stats.items():
        n = s["n"]
        ops_010 = s["above_0.10pct"] * n
        ops_020 = s["above_0.20pct"] * n
        ops_050 = s["above_0.50pct"] * n
        print(f"  {pair}: |spread|>0.10% {ops_010:.0f} times,  >0.20% {ops_020:.0f},  >0.50% {ops_050:.0f}  (over 11 days)")

    # ---- Save ----
    out = {
        "spread_stats": spread_stats,
        "exchanges_available": have_ex,
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido33_cross_exchange_spread.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")

    # ---- Verdict ----
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    fee_round_taker = 0.110  # %
    fee_mixed = 0.030  # %
    # Find best pair
    best_pair = max(spread_stats.items(), key=lambda kv: kv[1]["above_0.10pct"]) if spread_stats else None
    # Recompute opportunity counts with consistent keys
    pass
    if best_pair:
        name, s = best_pair
        print(f"\nBest pair (most >0.10% events): {name}  ({s['above_0.10pct']*100:.1f}% of snapshots)")
        if s["abs_q95"] > fee_round_taker:
            print(f"  ✅ q95 |spread| {s['abs_q95']:.3f}% > taker round-trip {fee_round_taker:.3f}%")
            print(f"  → tail events can profit even with taker. But infrequent.")
        elif s["abs_q95"] > fee_mixed:
            print(f"  🟡 q95 |spread| {s['abs_q95']:.3f}% > maker-mixed {fee_mixed:.3f}%")
            print(f"  → Some opportunity if maker-only fills feasible.")
        else:
            print(f"  ❌ q95 |spread| {s['abs_q95']:.3f}% < maker-mixed {fee_mixed:.3f}%")
            print(f"  → Even with rebate, spreads too small to profit.")

    print("\nLimitations:")
    print("  - Self mid only (no actual orderbook depth on Binance/OKX)")
    print("  - Real arb needs Binance/OKX OB collector (currently missing)")
    print("  - Sub-second spread persistence not measurable (sample interval ~1s)")
    print("  - Cross-exchange settlement fees, withdrawal fees not modeled")

    log.info("시도 33 complete")


if __name__ == "__main__":
    main()
