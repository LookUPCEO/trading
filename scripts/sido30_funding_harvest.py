"""시도 30: Funding Rate Harvesting analysis."""
import sys, logging, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mark19.storage import read_range


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("시도 30: Funding Rate Harvesting")
    log.info("=" * 70)

    # ---- 1. Historical funding distribution (Tardis dates) ----
    log.info("\n[1] Historical funding rate distribution (bybit ETHUSDT)")
    h_dir = Path("/Users/dohun/Desktop/Mark/mark19/data/funding_historical/bybit/ETHUSDT")
    files = sorted(h_dir.glob("*.parquet"))
    log.info(f"  Files: {len(files)}")
    if len(files) > 0:
        log.info(f"  Range: {files[0].stem} → {files[-1].stem}")

    samples = []
    for f in files[::30]:  # every 30th file for speed
        try:
            df = pd.read_parquet(f)
            samples.append(df)
        except Exception:
            continue
    h_df = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    log.info(f"  Sampled rows: {len(h_df)}")
    if len(h_df) > 0:
        rates = h_df["funding_rate"].astype(float) * 100  # to percent
        print()
        print("Historical funding rate distribution (% per 8h):")
        print(f"  N: {len(rates)}")
        print(f"  Mean: {rates.mean():+.4f}%  Std: {rates.std():.4f}")
        print(f"  Min: {rates.min():+.4f}  Max: {rates.max():+.4f}")
        for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
            print(f"  q{int(q*100):02d}: {rates.quantile(q):+.4f}%")
        thr_counts = {}
        for t in [0.005, 0.01, 0.02, 0.05]:
            thr_counts[t] = {
                "abs_above": float((rates.abs() > t).mean()),
                "above_pos": float((rates > t).mean()),
                "below_neg": float((rates < -t).mean()),
            }
        print("\nThreshold ratios (per 8h period):")
        print(f"  {'thr':<10} {'|R|>thr':<12} {'R>thr':<12} {'R<-thr':<12}")
        for t, d in thr_counts.items():
            print(f"  ±{t:<.3f}%   {d['abs_above']:<12.1%} {d['above_pos']:<12.1%} {d['below_neg']:<12.1%}")

    # ---- 2. Self funding (recent 9 days) — extract from funding_current/combined ----
    log.info("\n[2] Self funding analysis (9 days 2026-04-22~30)")
    SELF_DATES = [f"2026-04-{d:02d}" for d in range(22, 31)]
    cur_dir = Path("/Users/dohun/Desktop/Mark/mark19/data/funding_current/combined/ETHUSDT")
    raw_snaps = []
    # Need previous day too because funding event 00:00 has snapshots ending at 00:00 (some come from 23:50 prior)
    pd_self = SELF_DATES[:1]  # we'll also include 4/21 and the first SELF date
    all_dates_for_snaps = ["2026-04-21"] + SELF_DATES
    for d in all_dates_for_snaps:
        p = cur_dir / f"{d}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            raw_snaps.append(df)
    snap_df = pd.concat(raw_snaps, ignore_index=True) if raw_snaps else pd.DataFrame()
    log.info(f"  Self funding snapshots loaded: {len(snap_df)}")

    self_f_df = pd.DataFrame()
    if len(snap_df) > 0:
        snap_df["timestamp"] = pd.to_datetime(snap_df["timestamp"], utc=True)
        snap_df["bybit_next_time"] = pd.to_datetime(snap_df["bybit_next_time"], utc=True)
        snap_df = snap_df.dropna(subset=["bybit_funding", "bybit_next_time"])
        # For each unique next_time, take the snapshot whose timestamp is closest to (just before) next_time
        # That's the final predicted rate that becomes the actual paid rate
        snap_df["delta"] = (snap_df["bybit_next_time"] - snap_df["timestamp"]).dt.total_seconds()
        # Keep only snapshots within 30 min before next_time (final prediction window)
        final_snaps = snap_df[(snap_df["delta"] > 0) & (snap_df["delta"] < 1800)]
        # Take the last snapshot per next_time (smallest positive delta)
        final_snaps = final_snaps.sort_values("delta")
        events = final_snaps.groupby("bybit_next_time").first().reset_index()
        # Drop original snapshot timestamp before renaming to avoid duplicate column
        if "timestamp" in events.columns:
            events = events.drop(columns=["timestamp"])
        events = events.rename(columns={"bybit_next_time": "timestamp", "bybit_funding": "funding_rate"})
        events = events[["timestamp", "funding_rate"]].sort_values("timestamp")
        # Filter to only events whose timestamp date is in SELF_DATES
        events = events[events["timestamp"].dt.strftime("%Y-%m-%d").isin(SELF_DATES)]
        self_f_df = events.reset_index(drop=True)
        log.info(f"  Self funding events extracted: {len(self_f_df)}")

        if len(self_f_df) > 0:
            print("\nSelf funding events:")
            for _, r in self_f_df.iterrows():
                ts = r["timestamp"]
                rate = float(r["funding_rate"]) * 100
                print(f"  {ts.strftime('%Y-%m-%d %H:%M UTC')}  rate {rate:+.4f}%")

    # ---- 3. Theoretical perfect harvest ----
    log.info("\n[3] Theoretical perfect harvest")
    if len(self_f_df) > 0:
        rates_pct = self_f_df["funding_rate"].astype(float) * 100
        # Perfect: always be on side that receives funding
        perfect_per_event = rates_pct.abs()
        total_perfect = perfect_per_event.sum()
        n_events = len(rates_pct)
        n_days = len(SELF_DATES)
        per_day = total_perfect / n_days
        print(f"  Events: {n_events}  ({n_events/n_days:.1f} per day)")
        print(f"  Perfect per-event harvest avg: {perfect_per_event.mean():+.4f}%")
        print(f"  Total perfect (9 days): {total_perfect:+.4f}%")
        print(f"  Per day: {per_day:+.4f}%")
        print(f"  Annualized (perfect): {per_day*365:+.1f}%/year")

    # ---- 4. Backtest with direction risk (real OB price) ----
    log.info("\n[4] Backtest with direction risk (1h hold around funding time)")
    log.info("  Strategy: 30 min before funding, open opposite-of-funding direction. Exit 30 min after.")
    if len(self_f_df) > 0:
        # Read OB for self period
        start = datetime(2026, 4, 22, tzinfo=timezone.utc)
        end = datetime(2026, 4, 30, tzinfo=timezone.utc) + timedelta(days=1)
        log.info("  Loading OB...")
        ob = read_range("orderbook", "bybit", "ETHUSDT", start, end)
        if "bid_0_price" in ob.columns and "ask_0_price" in ob.columns:
            ob["mid"] = (ob["bid_0_price"] + ob["ask_0_price"]) / 2
        ob = ob.set_index(pd.to_datetime(ob["timestamp"], utc=True)).sort_index()
        mid_1m = ob["mid"].resample("1min").last().ffill()
        log.info(f"  Mid 1-min rows: {len(mid_1m)}")

        # For each funding event, compute strategy PnL
        FEE_TAKER = 0.055
        FEE_MAKER = -0.025
        HOLD_BEFORE_MIN = 30
        HOLD_AFTER_MIN = 30
        FUNDING_THR_PCT = 0.005  # 0.005% = 0.00005 rate

        results = []
        for _, r in self_f_df.iterrows():
            ts = pd.to_datetime(r["timestamp"])
            rate_pct = float(r["funding_rate"]) * 100  # %
            entry_ts = ts - pd.Timedelta(minutes=HOLD_BEFORE_MIN)
            exit_ts = ts + pd.Timedelta(minutes=HOLD_AFTER_MIN)

            # Get prices
            try:
                p_entry = float(mid_1m.asof(entry_ts))
                p_funding = float(mid_1m.asof(ts))
                p_exit = float(mid_1m.asof(exit_ts))
            except Exception:
                results.append(None); continue
            if any(pd.isna(p) for p in [p_entry, p_funding, p_exit]):
                results.append(None); continue

            # Skip if rate too small
            if abs(rate_pct) < FUNDING_THR_PCT:
                results.append({"ts": ts, "rate_pct": rate_pct, "skipped": True,
                                "trade_pnl_pct": 0, "funding_pnl_pct": 0,
                                "fee_pct": 0, "net_pnl_pct": 0,
                                "p_entry": p_entry, "p_exit": p_exit})
                continue

            # Direction: opposite of funding sign
            # rate > 0: longs pay shorts → SHORT to receive
            # rate < 0: shorts pay longs → LONG to receive
            direction = -1 if rate_pct > 0 else 1
            trade_pnl_pct = direction * (p_exit - p_entry) / p_entry * 100
            funding_pnl_pct = abs(rate_pct)  # received
            # Both legs taker (we're harvesting, no time to maker)
            fee_pct = FEE_TAKER + FEE_TAKER  # entry + exit
            net = trade_pnl_pct + funding_pnl_pct - fee_pct
            results.append({"ts": ts, "rate_pct": rate_pct, "skipped": False,
                            "direction": direction,
                            "trade_pnl_pct": trade_pnl_pct,
                            "funding_pnl_pct": funding_pnl_pct,
                            "fee_pct": fee_pct,
                            "net_pnl_pct": net,
                            "p_entry": p_entry, "p_exit": p_exit})

        valid = [r for r in results if r is not None]
        executed = [r for r in valid if not r["skipped"]]
        skipped = [r for r in valid if r["skipped"]]
        log.info(f"  Valid events: {len(valid)}, executed: {len(executed)}, skipped: {len(skipped)}")

        print("\nBacktest events:")
        print(f"{'TS UTC':<22} {'Rate%':<8} {'Dir':<5} {'Trade':<10} {'Fund':<10} {'Fee':<8} {'Net':<10}")
        for r in executed:
            ts_str = r["ts"].strftime("%Y-%m-%d %H:%M")
            print(f"{ts_str:<22} {r['rate_pct']:<+8.4f} {r['direction']:<5} {r['trade_pnl_pct']:<+10.3f} {r['funding_pnl_pct']:<+10.4f} {r['fee_pct']:<8.3f} {r['net_pnl_pct']:<+10.3f}")

        if executed:
            net_arr = np.array([r["net_pnl_pct"] for r in executed])
            trade_arr = np.array([r["trade_pnl_pct"] for r in executed])
            fund_arr = np.array([r["funding_pnl_pct"] for r in executed])
            fee_arr = np.array([r["fee_pct"] for r in executed])
            print(f"\nAggregate ({len(executed)} executed events):")
            print(f"  Net PnL total: {net_arr.sum():+.3f}%  (avg/event {net_arr.mean():+.4f}%)")
            print(f"  Trade direction PnL: {trade_arr.sum():+.3f}%")
            print(f"  Funding received: {fund_arr.sum():+.3f}%")
            print(f"  Fees: {fee_arr.sum():.3f}%")
            print(f"  Per day: {net_arr.sum()/9:+.4f}%/day")
            print(f"  Annualized: {net_arr.sum()/9*365:+.1f}%/year")

            # Sharpe
            sharpe = (net_arr.mean() / net_arr.std()) * np.sqrt(len(net_arr) / 9 * 365) if net_arr.std() > 0 else 0
            print(f"  Sharpe (annualized): {sharpe:.2f}")

    # ---- 5. Combined strategy idea ----
    print()
    print("=" * 80)
    print("COMBINED STRATEGY IDEA")
    print("=" * 80)
    print("""
시도 29f Direction (mark29f) + 시도 30 Funding harvesting:
  - 시도 29f: 일 +0.541% (4 days subset robust, 9 days -0.046%)
  - 시도 30 (위 backtest): real-time direction risk 포함 후 결과 판정
  - 두 source 가 독립적이라면 합산 가능

LIVE 적용 시 운용:
  1. Direction trades: vol_filter + ENS dir_proba > 0.55
  2. Funding harvest: 30분 전 진입, 30분 후 청산 (큰 funding 만)
  3. 충돌: funding 시간대 직전 direction trade 회피 권장
""")

    out_path = Path("/Users/dohun/Desktop/Mark/mark19/data/analysis_results/sido30_funding_harvest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {"approach": "Funding harvesting analysis"}
    if len(self_f_df) > 0 and 'executed' in dir() and executed:
        out["self_summary"] = {
            "total_events": len(valid),
            "executed_events": len(executed),
            "skipped_events": len(skipped),
            "net_total_pct": float(np.array([r["net_pnl_pct"] for r in executed]).sum()),
            "per_day_pct": float(np.array([r["net_pnl_pct"] for r in executed]).sum() / 9),
            "annualized_pct": float(np.array([r["net_pnl_pct"] for r in executed]).sum() / 9 * 365),
            "trade_direction_total": float(np.array([r["trade_pnl_pct"] for r in executed]).sum()),
            "funding_received_total": float(np.array([r["funding_pnl_pct"] for r in executed]).sum()),
            "fees_total": float(np.array([r["fee_pct"] for r in executed]).sum()),
        }
        out["events"] = [{
            "ts": str(r["ts"]), "rate_pct": r["rate_pct"], "direction": r["direction"],
            "trade_pnl": r["trade_pnl_pct"], "funding_pnl": r["funding_pnl_pct"],
            "fee": r["fee_pct"], "net_pnl": r["net_pnl_pct"],
        } for r in executed]
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nJSON: {out_path}")
    log.info("\n시도 30 complete")


if __name__ == "__main__":
    main()
