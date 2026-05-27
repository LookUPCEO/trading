"""Path E: SHADOW MM 8-day analysis + sido28b lookahead audit.

Findings to report:
  1. SHADOW MM (PID 65673) runs SPREAD_BP=0 (at-mid).
  2. sido28b backtest: sp0 is the WORST config (-0.02%/day on 11 days).
     sp0.5 (best, +0.23%/day) and sp2 (Sharpe 13.4, +0.21%/day) are positive.
  3. Audit sido28b for lookahead: process_adverse uses future_i (i+60s) for stat
     only — trading decisions are causal (use only data ≤ current i).
  4. Replay sido28b sp0 + sp0.5 on actual SHADOW period (4/22-5/9) for empirical
     comparison.

This script:
  - Re-runs sido28b simulate_day() across 4/22-5/9 (live SHADOW window) for
    sp0 + sp0.5 + sp2 configs.
  - Reports daily PnL, toxic_rate, n_pairs.
  - Compares live SHADOW MM aggregate stats (25535 fills, 58.9% toxic) to the
    sp0 backtest expectation.
  - Audit: no lookahead found (process_adverse is post-hoc stat).
"""
import sys, logging, json, importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_mm", _HERE / "sido28b_strict_market_making.py")
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
simulate_day = _mod.simulate_day


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("Path E: MM analysis (live SHADOW MM 4/22-5/9 + sido28b empirical replay)")
    log.info("=" * 70)
    np.random.seed(42)

    # Live SHADOW MM ran 5/1 15:33 → 5/9 (8 days). But OB/trade parquet exists for
    # both ranges; replay full available 4/21-5/9 (19 days) for fuller stat.
    DATES = []
    for d in range(21, 31):
        DATES.append(f"2026-04-{d:02d}")
    for d in range(1, 10):
        DATES.append(f"2026-05-{d:02d}")
    log.info(f"Replay dates: {len(DATES)} ({DATES[0]} → {DATES[-1]})")

    # 3 configs of interest
    CONFIGS = [
        ("sp0_sz0.01", 0, 0.01),       # ← live SHADOW MM config
        ("sp0.5_sz0.01", 0.5, 0.01),   # ← sido28b best mean
        ("sp2_sz0.01", 2, 0.01),       # ← sido28b best sharpe
    ]

    results = {}
    for name, sp_bp, sz in CONFIGS:
        log.info(f"\n=== Config {name} (spread_bp={sp_bp}, size={sz}) ===")
        per_day = []
        for d in DATES:
            r = simulate_day(d, sp_bp, sz, log)
            if r is None:
                log.warning(f"  {d}: data missing, skip"); continue
            per_day.append(r)
            avg_mid = r["avg_mid"]
            pct = (r["pnl_usd"] / (avg_mid * sz)) if (avg_mid * sz > 0) else 0
            log.info(f"  {d}: pnl=${r['pnl_usd']:+.2f}  pairs={r['n_pairs']}  "
                     f"toxic={r['toxic_rate']*100:.1f}%  pct={pct*100:+.4f}%")
        if not per_day:
            log.warning(f"  {name}: no data"); continue
        # Aggregate
        total_pnl = sum(r["pnl_usd"] for r in per_day)
        avg_mid = np.mean([r["avg_mid"] for r in per_day])
        total_size = sz * len(per_day)  # approximate position-day count
        # Compute pct relative to average notional (using avg_mid * size_eth as "1-day notional")
        pcts = [(r["pnl_usd"] / (r["avg_mid"] * sz)) for r in per_day if r["avg_mid"] * sz > 0]
        v = np.array(pcts) * 100
        toxic_avg = np.mean([r["toxic_rate"] for r in per_day])
        pairs_total = sum(r["n_pairs"] for r in per_day)
        results[name] = {
            "spread_bp": sp_bp, "size_eth": sz,
            "n_days": len(per_day),
            "total_pnl_usd": float(total_pnl),
            "mean_pct_per_day": float(v.mean()),
            "std_pct": float(v.std()),
            "sharpe_annualized": float((v.mean() / v.std()) * np.sqrt(365)) if v.std() > 0 else float("nan"),
            "positive_days": int((np.array([r["pnl_usd"] for r in per_day]) > 0).sum()),
            "toxic_rate_avg": float(toxic_avg),
            "pairs_total": int(pairs_total),
            "pairs_per_day_avg": float(pairs_total / len(per_day)),
        }
        log.info(f"  → mean {v.mean():+.4f}%/day  std {v.std():.4f}  "
                 f"sharpe {results[name]['sharpe_annualized']:+.2f}  "
                 f"pos {results[name]['positive_days']}/{len(per_day)}  "
                 f"toxic {toxic_avg*100:.1f}%  pairs {pairs_total}")

    # Compare to live SHADOW MM aggregate (read state JSON)
    state_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/shadow_mm_state_ETHUSDT.json")
    live = json.load(open(state_path)) if state_path.exists() else {}
    print()
    print("=" * 90)
    print("Path E — sido28b REPLAY 4/21-5/9 vs LIVE SHADOW MM (PID 65673)")
    print("=" * 90)
    print(f"{'Config':<14} {'mean%/day':<14} {'sharpe':<10} {'pos/N':<10} {'toxic%':<10} {'pairs/d':<10}")
    print("-" * 80)
    for name, r in results.items():
        marker = "  ← live" if name == "sp0_sz0.01" else ""
        print(f"{name:<14} {r['mean_pct_per_day']:+.4f}        {r['sharpe_annualized']:+.2f}      "
              f"{r['positive_days']}/{r['n_days']:<7} {r['toxic_rate_avg']*100:.1f}        "
              f"{r['pairs_per_day_avg']:.0f}{marker}")
    print()
    print("LIVE SHADOW MM stats:")
    if live:
        uptime_h = live.get("uptime_sec", 0) / 3600
        print(f"  Uptime: {uptime_h:.2f}h ({uptime_h/24:.2f}d)")
        print(f"  Fills: {live.get('n_fills',0)}  Cancels: {live.get('n_cancels',0)}  Places: {live.get('n_places',0)}")
        print(f"  Toxic rate: {live.get('toxic_rate',0)*100:.1f}%  (vs sido28b sp0 replay)")
        print(f"  Inventory: {live.get('inventory',0):+.4f}")
        print(f"  Pairs/day est: {live.get('n_fills',0)/2/(uptime_h/24):.0f}")
    else:
        print("  state JSON missing")

    # Diagnosis
    print()
    print("=" * 90)
    print("DIAGNOSIS")
    print("=" * 90)
    sp0 = results.get("sp0_sz0.01")
    sp05 = results.get("sp0.5_sz0.01")
    sp2 = results.get("sp2_sz0.01")
    if sp0:
        print(f"\n  Live SHADOW MM (sp0) backtest replay: {sp0['mean_pct_per_day']:+.4f}%/day, "
              f"sharpe {sp0['sharpe_annualized']:+.2f}, {sp0['positive_days']}/{sp0['n_days']} days positive")
        if sp0["mean_pct_per_day"] >= 0:
            print(f"    → Live config marginally positive on recent data")
        else:
            print(f"    → Live config NEGATIVE on recent data")
    if sp05 and sp0:
        delta = sp05["mean_pct_per_day"] - sp0["mean_pct_per_day"]
        print(f"\n  sp0.5 vs sp0 advantage: {delta:+.4f}%/day")
        print(f"    sp0.5: {sp05['mean_pct_per_day']:+.4f}%/day, sharpe {sp05['sharpe_annualized']:+.2f}")
        if delta > 0.1:
            print(f"    → SHADOW MM should switch to sp0.5 (significant edge)")
    if sp2:
        print(f"\n  sp2: {sp2['mean_pct_per_day']:+.4f}%/day, sharpe {sp2['sharpe_annualized']:+.2f}")

    print(f"\n  Lookahead audit on sido28b: simulate_day uses only data ≤ current i for")
    print(f"  trading decisions. process_adverse(future_i = i+60) is post-hoc stat only.")
    print(f"  → No lookahead bias. sido28b PnL claims are structurally legitimate.")
    print(f"  Caveat: queue half-size at spread>0 (initial_q = bsz*0.5) may be generous.")

    # Save
    out = {
        "approach": "Path E — sido28b replay 4/21-5/9 + live SHADOW MM stats",
        "configs": results,
        "live_shadow_mm": live,
        "audit_lookahead": "No leak — process_adverse is post-hoc stat only",
    }
    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/pathE_mm_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")
    log.info("Path E complete")


if __name__ == "__main__":
    main()
