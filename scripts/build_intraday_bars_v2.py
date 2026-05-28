"""
Intraday 5min bars v2 — full 50-level orderbook utilization + intra-bar dynamics.

Adds on top of v1:
  - Multi-level OBI: top1/5/10/20/50
  - Multi-depth: bid/ask depth at 5/10/20/50 levels
  - Multi-span slope: span 5/10/20
  - Concentration: top5_depth / top50_depth (how much liquidity clusters at top)
  - Microprice & microprice_deviation: ((ask*bid_size + bid*ask_size)/(bid_size+ask_size) - mid) / mid * 10000
  - Intra-bar dynamics (1Hz aggregated):
      * mid velocity (log-return per second, intra-bar mean/std)
      * OBI change std (volatility of imbalance signal)
      * Spread regime (mean spread / max spread in bar)
  - Wall detection: max_size_top10 / mean_size_top10 (concentration in single level)
"""
from __future__ import annotations
import argparse, logging, sys, time, re
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd


DEFAULT_IN_ROOT = Path("/Users/mark/mark19_data")
DEFAULT_OUT_ROOT = Path("/Users/mark/mark19_data/bars_5min_v2")
BAR_SECONDS = 300
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.parquet$")


def build_bars_for_day(in_path: Path) -> pd.DataFrame:
    cols_needed = (["timestamp", "update_id"] +
                   [f"bid_{k}_price" for k in range(50)] +
                   [f"bid_{k}_size" for k in range(50)] +
                   [f"ask_{k}_price" for k in range(50)] +
                   [f"ask_{k}_size" for k in range(50)])
    df = pd.read_parquet(in_path, columns=cols_needed)
    if len(df) == 0:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # FIX (day-boundary wrap-around): raw files include the next day's first
    # snapshot (00:00:0X). bar_idx uses sec_of_day only (date-agnostic), so that
    # row wraps into bar_idx 0 and overwrites bar 0's mid_close with the NEXT
    # day's opening mid — corrupting mom_*/dist_ma/rv/cumflow near every day
    # boundary. Drop rows not belonging to this calendar day.
    _day = pd.Timestamp(in_path.stem).date()
    df = df[df["timestamp"].dt.date == _day].reset_index(drop=True)
    if len(df) == 0:
        return pd.DataFrame()

    bid_p0 = df["bid_0_price"]; ask_p0 = df["ask_0_price"]
    mid = (bid_p0 + ask_p0) / 2
    spread = ask_p0 - bid_p0
    spread_bp = spread / mid * 10000

    # Multi-depth sums + multi-OBI (vectorized)
    depth_pieces = {}
    obi_pieces = {}
    for k in [1, 5, 10, 20, 50]:
        bid_k = df[[f"bid_{i}_size" for i in range(k)]].sum(axis=1, skipna=True)
        ask_k = df[[f"ask_{i}_size" for i in range(k)]].sum(axis=1, skipna=True)
        depth_pieces[f"bid_d{k}"] = bid_k
        depth_pieces[f"ask_d{k}"] = ask_k
        depth_pieces[f"tot_d{k}"] = bid_k + ask_k
        tot = bid_k + ask_k
        obi_pieces[f"obi{k}"] = np.where(tot > 0, (bid_k - ask_k) / tot, 0.0)

    # Slopes at multiple spans
    slope_pieces = {}
    for span in [5, 10, 20]:
        slope_pieces[f"bid_sl{span}"] = (bid_p0 - df[f"bid_{span-1}_price"]) / span
        slope_pieces[f"ask_sl{span}"] = (df[f"ask_{span-1}_price"] - ask_p0) / span

    # Concentration: top5 / top50
    concentration = depth_pieces["tot_d5"] / depth_pieces["tot_d50"].replace(0, np.nan)

    # Microprice (top-of-book weighted by sizes on opposite side)
    bid_s0 = df["bid_0_size"].fillna(0); ask_s0 = df["ask_0_size"].fillna(0)
    micro_tot = bid_s0 + ask_s0
    microprice = np.where(micro_tot > 0,
                           (ask_p0 * bid_s0 + bid_p0 * ask_s0) / micro_tot,
                           mid)
    microprice_dev_bp = (microprice - mid) / mid * 10000

    # Wall detection on top-10: max single-level size / mean top-10 size
    top10_sizes_bid = df[[f"bid_{i}_size" for i in range(10)]]
    top10_sizes_ask = df[[f"ask_{i}_size" for i in range(10)]]
    bid_wall = top10_sizes_bid.max(axis=1) / top10_sizes_bid.mean(axis=1).replace(0, np.nan)
    ask_wall = top10_sizes_ask.max(axis=1) / top10_sizes_ask.mean(axis=1).replace(0, np.nan)

    # OFI proxy (top imbalance change)
    top_imb = bid_s0 - ask_s0
    ofi_step = top_imb.diff().fillna(0)

    # Mid log-return per snapshot (intra-bar velocity)
    log_mid = np.log(mid)
    log_ret_step = log_mid.diff().fillna(0)

    # OBI top5 change (signal volatility)
    obi5_step = pd.Series(obi_pieces["obi5"]).diff().fillna(0)

    work = pd.DataFrame({
        "timestamp": df["timestamp"],
        "mid": mid.values, "log_mid": log_mid.values,
        "spread_bp": spread_bp.values,
        "microprice_dev_bp": microprice_dev_bp,
        "bid_wall_ratio": bid_wall.values, "ask_wall_ratio": ask_wall.values,
        "concentration_5_50": concentration.values,
        "ofi_step": ofi_step.values,
        "log_ret_step": log_ret_step.values,
        "obi5_step": obi5_step.values,
    })
    for k, v in obi_pieces.items(): work[k] = v
    for k, v in depth_pieces.items(): work[k] = v.values
    for k, v in slope_pieces.items(): work[k] = v.values

    sec_of_day = (work["timestamp"].dt.hour * 3600 +
                  work["timestamp"].dt.minute * 60 +
                  work["timestamp"].dt.second)
    work["bar_idx"] = (sec_of_day // BAR_SECONDS).astype(int)
    g = work.groupby("bar_idx", sort=True)

    bars = pd.DataFrame({
        "bar_idx": g.size().index,
        "bar_open_ts": g["timestamp"].first().values,
        "bar_close_ts": g["timestamp"].last().values,
        "mid_open": g["mid"].first().values,
        "mid_close": g["mid"].last().values,
        "mid_high": g["mid"].max().values,
        "mid_low": g["mid"].min().values,
        # realized vol of intra-bar log-returns
        "rv_bar_bp": (g["log_ret_step"].std().fillna(0).values * 10000),
        # mid velocity stats
        "vel_mean_bp": g["log_ret_step"].mean().values * 10000,
        "vel_abs_mean_bp": g["log_ret_step"].apply(lambda x: x.abs().mean()).values * 10000,
        # spread
        "spread_bp_mean": g["spread_bp"].mean().values,
        "spread_bp_max": g["spread_bp"].max().values,
        "spread_bp_last": g["spread_bp"].last().values,
        # microprice
        "micro_dev_bp_mean": g["microprice_dev_bp"].mean().values,
        "micro_dev_bp_last": g["microprice_dev_bp"].last().values,
        # OBI multi-level
        "obi1_last": g["obi1"].last().values,
        "obi5_mean": g["obi5"].mean().values,
        "obi5_last": g["obi5"].last().values,
        "obi5_std": g["obi5"].std().fillna(0).values,
        "obi10_last": g["obi10"].last().values,
        "obi20_last": g["obi20"].last().values,
        "obi50_last": g["obi50"].last().values,
        "obi5_step_std": g["obi5_step"].std().fillna(0).values,
        # Depth multi-level
        "tot_d5_mean": g["tot_d5"].mean().values,
        "tot_d10_mean": g["tot_d10"].mean().values,
        "tot_d20_mean": g["tot_d20"].mean().values,
        "tot_d50_mean": g["tot_d50"].mean().values,
        "bid_d5_last": g["bid_d5"].last().values,
        "ask_d5_last": g["ask_d5"].last().values,
        "bid_d50_last": g["bid_d50"].last().values,
        "ask_d50_last": g["ask_d50"].last().values,
        # Concentration / asymmetry
        "concentration_5_50_mean": g["concentration_5_50"].mean().values,
        "depth_asym_50": (g["bid_d50"].last().values /
                         g["ask_d50"].last().replace(0, np.nan).values),
        # Slopes
        "bid_sl5_mean": g["bid_sl5"].mean().values,
        "ask_sl5_mean": g["ask_sl5"].mean().values,
        "bid_sl10_mean": g["bid_sl10"].mean().values,
        "ask_sl10_mean": g["ask_sl10"].mean().values,
        "bid_sl20_mean": g["bid_sl20"].mean().values,
        "ask_sl20_mean": g["ask_sl20"].mean().values,
        # Wall detection
        "bid_wall_mean": g["bid_wall_ratio"].mean().values,
        "ask_wall_mean": g["ask_wall_ratio"].mean().values,
        # OFI
        "ofi_proxy": g["ofi_step"].sum().values,
        # Counts
        "n_updates": g.size().values,
    })
    bars["return_5m_bar_bp"] = (np.log(bars["mid_close"]) - np.log(bars["mid_open"])) * 10000

    # Forward targets
    mc = bars["mid_close"]
    for n in [1, 2, 3, 6, 12, 24, 48]:
        future = mc.shift(-n)
        bars[f"target_return_{n}bar_bp"] = (np.log(future) - np.log(mc)) * 10000

    return bars


def worker(task):
    sym, ds, in_path, out_path = task
    t0 = time.time()
    try:
        bars = build_bars_for_day(in_path)
        if len(bars) == 0:
            return (sym, ds, time.time() - t0, 0, 0, "EMPTY")
        bars["symbol"] = sym; bars["date"] = ds
        bars.to_parquet(out_path, compression="zstd", index=False)
        return (sym, ds, time.time() - t0, len(bars), out_path.stat().st_size, "ok")
    except Exception as e:
        return (sym, ds, time.time() - t0, 0, 0, f"FAIL: {type(e).__name__}: {e}")


def discover(in_root: Path, out_root: Path, symbols: list) -> list:
    tasks = []
    for sym in symbols:
        in_dir = in_root / sym
        if not in_dir.exists():
            continue
        out_dir = out_root / sym
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(in_dir.iterdir()):
            m = DATE_RE.match(f.name)
            if not m:
                continue
            ds = m.group(1)
            out_path = out_dir / f.name
            if out_path.exists() and out_path.stat().st_size > 0:
                continue
            tasks.append((sym, ds, f, out_path))
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SOLUSDT")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--in-root", type=Path, default=DEFAULT_IN_ROOT)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tasks = discover(args.in_root, args.out_root, symbols)
    log.info(f"Discovered {len(tasks)} files")
    if args.limit > 0:
        tasks = tasks[:args.limit]

    if not tasks:
        log.info("Nothing to do."); return

    t0 = time.time(); done = failed = 0; total_bars = total_bytes = 0
    n = len(tasks)
    with Pool(processes=args.workers, maxtasksperchild=50) as pool:
        for sym, ds, dt, n_bars, sz, status in pool.imap_unordered(worker, tasks):
            done += 1
            if status != "ok":
                failed += 1
                log.warning(f"  [{done}/{n}] {sym} {ds} {status} ({dt:.1f}s)")
            else:
                total_bars += n_bars; total_bytes += sz
                if done % 50 == 0 or done == n:
                    rate = done / (time.time() - t0)
                    log.info(f"  [{done}/{n}] {sym} {ds} {n_bars}bars ({dt:.1f}s) | rate {rate*60:.0f}/min | size {total_bytes/1024/1024:.0f}MB")
    log.info(f"DONE. {done-failed} ok / {failed} fail | {total_bars} bars / {total_bytes/1024/1024:.0f}MB | {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
