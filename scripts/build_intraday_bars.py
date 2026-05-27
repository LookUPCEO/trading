"""
Build intraday 5min bars from 1Hz mark19 orderbook snapshots.

Per day:
  Input  : ~/mark19_data/{SYMBOL}/{YYYY-MM-DD}.parquet  (86401 rows × 203 cols)
  Output : {OUT_ROOT}/{SYMBOL}/{YYYY-MM-DD}.parquet     (~288 bars × ~22 cols)

Bar features (orderbook-only, causal):
  - bar_open_ts, bar_close_ts
  - mid_open, mid_close, mid_high, mid_low, mid_realized_vol_bp (std of log-returns within bar)
  - spread_bp_mean, spread_bp_last
  - obi_top5_mean, obi_top5_last, obi_top1_last
  - depth_top5_mean (total = bid+ask)
  - bid_depth_top5_last, ask_depth_top5_last
  - bid_slope_10_mean, ask_slope_10_mean
  - n_updates (count of 1Hz snapshots in bar)
  - ofi_proxy (top-of-book imbalance change sum)
  - return_5m_bar (mid close vs open in bp)

Targets (forward-looking, causal target NOT a feature):
  - target_return_1bar_bp   (5min ahead)
  - target_return_3bar_bp   (15min ahead)
  - target_return_12bar_bp  (1h ahead)
"""
from __future__ import annotations
import argparse, logging, sys, time, re
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd


DEFAULT_IN_ROOT = Path("/Users/mark/mark19_data")
DEFAULT_OUT_ROOT = Path("/Users/mark/mark19_data/bars_5min")
BAR_SECONDS = 300  # 5 minutes
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.parquet$")


def build_bars_for_day(in_path: Path) -> pd.DataFrame:
    """Read 1-day 1Hz orderbook, produce 5min bars with features + 3 forward targets."""
    cols_needed = (["timestamp", "update_id"] +
                   [f"bid_{k}_price" for k in range(10)] +
                   [f"bid_{k}_size" for k in range(5)] +
                   [f"ask_{k}_price" for k in range(10)] +
                   [f"ask_{k}_size" for k in range(5)])
    df = pd.read_parquet(in_path, columns=cols_needed)
    if len(df) == 0:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Point-wise features (vectorized)
    mid = (df["bid_0_price"] + df["ask_0_price"]) / 2
    spread = df["ask_0_price"] - df["bid_0_price"]
    spread_bp = spread / mid * 10000

    bid5 = df[[f"bid_{k}_size" for k in range(5)]].sum(axis=1, skipna=True)
    ask5 = df[[f"ask_{k}_size" for k in range(5)]].sum(axis=1, skipna=True)
    total5 = bid5 + ask5
    obi5 = np.where(total5 > 0, (bid5 - ask5) / total5, 0.0)

    bid1 = df["bid_0_size"].fillna(0)
    ask1 = df["ask_0_size"].fillna(0)
    total1 = bid1 + ask1
    obi1 = np.where(total1 > 0, (bid1 - ask1) / total1, 0.0)

    bid_slope10 = (df["bid_0_price"] - df["bid_9_price"]) / 10
    ask_slope10 = (df["ask_9_price"] - df["ask_0_price"]) / 10

    # OFI proxy: change in (bid_0_size - ask_0_size) per snapshot
    top_imb = bid1 - ask1
    ofi_step = top_imb.diff().fillna(0)

    work = pd.DataFrame({
        "timestamp": df["timestamp"],
        "mid": mid,
        "log_mid": np.log(mid),
        "spread_bp": spread_bp,
        "obi5": obi5,
        "obi1": obi1,
        "total5": total5,
        "bid5": bid5,
        "ask5": ask5,
        "bid_slope10": bid_slope10,
        "ask_slope10": ask_slope10,
        "ofi_step": ofi_step,
    })
    # Bucket index = floor seconds since day-start / 300
    sec_of_day = (work["timestamp"].dt.hour * 3600 +
                  work["timestamp"].dt.minute * 60 +
                  work["timestamp"].dt.second)
    work["bar_idx"] = (sec_of_day // BAR_SECONDS).astype(int)
    # log returns within bar (consecutive snapshots) for realized vol
    work["log_ret_step"] = work["log_mid"].diff()

    g = work.groupby("bar_idx", sort=True)
    bars = pd.DataFrame({
        "bar_idx": g.size().index,
        "bar_open_ts": g["timestamp"].first().values,
        "bar_close_ts": g["timestamp"].last().values,
        "mid_open": g["mid"].first().values,
        "mid_close": g["mid"].last().values,
        "mid_high": g["mid"].max().values,
        "mid_low": g["mid"].min().values,
        "mid_realized_vol_bp": (g["log_ret_step"].std().fillna(0).values * 10000),
        "spread_bp_mean": g["spread_bp"].mean().values,
        "spread_bp_last": g["spread_bp"].last().values,
        "obi5_mean": g["obi5"].mean().values,
        "obi5_last": g["obi5"].last().values,
        "obi1_last": g["obi1"].last().values,
        "depth_top5_mean": g["total5"].mean().values,
        "bid_depth_top5_last": g["bid5"].last().values,
        "ask_depth_top5_last": g["ask5"].last().values,
        "bid_slope10_mean": g["bid_slope10"].mean().values,
        "ask_slope10_mean": g["ask_slope10"].mean().values,
        "n_updates": g.size().values,
        "ofi_proxy": g["ofi_step"].sum().values,
    })
    bars["return_5m_bar_bp"] = (np.log(bars["mid_close"]) - np.log(bars["mid_open"])) * 10000

    # Forward targets (mid_close shift)
    mid_close = bars["mid_close"]
    for n in [1, 3, 12]:
        future = mid_close.shift(-n)
        bars[f"target_return_{n}bar_bp"] = (np.log(future) - np.log(mid_close)) * 10000

    return bars


def worker(task):
    sym, date_str, in_path, out_path = task
    t0 = time.time()
    try:
        bars = build_bars_for_day(in_path)
        if len(bars) == 0:
            return (sym, date_str, time.time() - t0, 0, 0, "EMPTY")
        bars["symbol"] = sym
        bars["date"] = date_str
        bars.to_parquet(out_path, compression="zstd", index=False)
        return (sym, date_str, time.time() - t0, len(bars),
                out_path.stat().st_size, "ok")
    except Exception as e:
        return (sym, date_str, time.time() - t0, 0, 0,
                f"FAIL: {type(e).__name__}: {e}")


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
            date_str = m.group(1)
            out_path = out_dir / f.name
            if out_path.exists() and out_path.stat().st_size > 0:
                continue
            tasks.append((sym, date_str, f, out_path))
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SOLUSDT")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--in-root", type=Path, default=DEFAULT_IN_ROOT)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tasks = discover(args.in_root, args.out_root, symbols)
    log.info(f"Discovered {len(tasks)} day-files to process")
    if args.limit > 0:
        tasks = tasks[:args.limit]
        log.info(f"  limited to {len(tasks)}")
    if not tasks:
        log.info("Nothing to do.")
        return

    t0 = time.time()
    done = failed = 0
    total_bars = total_bytes = 0
    n = len(tasks)
    with Pool(processes=args.workers, maxtasksperchild=50) as pool:
        for sym, ds, dt, n_bars, sz, status in pool.imap_unordered(worker, tasks):
            done += 1
            if status != "ok":
                failed += 1
                log.warning(f"  [{done}/{n}] {sym} {ds} {status} ({dt:.1f}s)")
            else:
                total_bars += n_bars; total_bytes += sz
                if done % 20 == 0 or done == n:
                    rate = done / (time.time() - t0)
                    eta_h = (n - done) / rate / 3600 if rate > 0 else 0
                    log.info(f"  [{done}/{n}] {sym} {ds} {n_bars} bars ({dt:.1f}s) | "
                             f"rate {rate*60:.0f}/min | ETA {eta_h:.2f}h | total_size {total_bytes/1024/1024:.0f}MB")

    log.info(f"DONE. {done-failed} ok / {failed} fail | total {total_bars} bars / "
             f"{total_bytes/1024/1024:.0f}MB | elapsed {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
