"""
Intraday 5min bars v3 = v2 orderbook features + Bybit perpetual trades features.

Adds on top of v2 (53 cols):
  Per-bar trades aggregates (causal — only uses trades within the bar):
    - tr_count, tr_buy_count, tr_sell_count
    - tr_buy_size, tr_sell_size, tr_net_size  (buy - sell, signed flow)
    - tr_total_notional, tr_buy_notional, tr_sell_notional
    - tr_vwap_bp_dev  (volume-weighted price - mid_close, in bp)
    - tr_large_count  (count of trades with size > p95 of that day)
    - tr_plus_count, tr_minus_count, tr_tick_imb  (tickDirection momentum)
    - tr_intensity (trades per second)
    - tr_max_trade_size, tr_size_p95_in_bar

Trades source : ~/mark19_data/trades_perp/{SYMBOL}/{date}.parquet
Bar template  : ~/mark19_data/bars_5min_v2/{SYMBOL}/{date}.parquet  (existing v2 bars)
Output        : ~/mark19_data/bars_5min_v3/{SYMBOL}/{date}.parquet
"""
from __future__ import annotations
import argparse, logging, re, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd


DEFAULT_V2_ROOT = Path("/Users/mark/mark19_data/bars_5min_v2")
DEFAULT_TRADES_ROOT = Path("/Users/mark/mark19_data/trades_perp")
DEFAULT_OUT_ROOT = Path("/Users/mark/mark19_data/bars_5min_v3")
BAR_SECONDS = 300
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.parquet$")


def trades_features_for_day(trades_path: Path, day_str: str) -> pd.DataFrame:
    """Aggregate trades into 5min bars. Returns DF indexed by bar_idx 0..287."""
    df = pd.read_parquet(trades_path,
        columns=["timestamp","side","size","price","tickDirection","homeNotional"])
    if len(df) == 0:
        return pd.DataFrame()
    # timestamp is seconds-since-epoch float
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    # second-of-day → bar_idx
    sec = (df["ts"].dt.hour * 3600 + df["ts"].dt.minute * 60 + df["ts"].dt.second).astype("int64")
    df["bar_idx"] = (sec // BAR_SECONDS).astype("int32")
    df["is_buy"] = (df["side"] == "Buy").astype("int32")
    df["is_sell"] = (df["side"] == "Sell").astype("int32")
    df["notional"] = df["size"] * df["price"]  # USDT notional
    # per-day large threshold (p95 size)
    p95 = df["size"].quantile(0.95)
    df["is_large"] = (df["size"] > p95).astype("int32")
    df["is_plus"] = df["tickDirection"].isin(["PlusTick", "ZeroPlusTick"]).astype("int32")
    df["is_minus"] = df["tickDirection"].isin(["MinusTick", "ZeroMinusTick"]).astype("int32")
    df["buy_size"] = df["size"] * df["is_buy"]
    df["sell_size"] = df["size"] * df["is_sell"]
    df["buy_notional"] = df["notional"] * df["is_buy"]
    df["sell_notional"] = df["notional"] * df["is_sell"]
    df["sz_price"] = df["size"] * df["price"]

    g = df.groupby("bar_idx", sort=True)
    out = pd.DataFrame({
        "bar_idx": g.size().index.astype("int32"),
        "tr_count": g.size().values.astype("int64"),
        "tr_buy_count": g["is_buy"].sum().values.astype("int64"),
        "tr_sell_count": g["is_sell"].sum().values.astype("int64"),
        "tr_buy_size": g["buy_size"].sum().values,
        "tr_sell_size": g["sell_size"].sum().values,
        "tr_total_notional": g["notional"].sum().values,
        "tr_buy_notional": g["buy_notional"].sum().values,
        "tr_sell_notional": g["sell_notional"].sum().values,
        "tr_vwap": (g["sz_price"].sum() / g["size"].sum().replace(0, np.nan)).values,
        "tr_large_count": g["is_large"].sum().values.astype("int64"),
        "tr_plus_count": g["is_plus"].sum().values.astype("int64"),
        "tr_minus_count": g["is_minus"].sum().values.astype("int64"),
        "tr_max_size": g["size"].max().values,
        "tr_size_p95_in_bar": g["size"].quantile(0.95).values,
    })
    out["tr_net_size"] = out["tr_buy_size"] - out["tr_sell_size"]
    out["tr_net_notional"] = out["tr_buy_notional"] - out["tr_sell_notional"]
    out["tr_tick_imb"] = (out["tr_plus_count"] - out["tr_minus_count"]) / \
                         (out["tr_plus_count"] + out["tr_minus_count"]).replace(0, np.nan)
    out["tr_intensity"] = out["tr_count"] / BAR_SECONDS
    out["tr_buy_ratio"] = out["tr_buy_size"] / (out["tr_buy_size"] + out["tr_sell_size"]).replace(0, np.nan)
    return out


def build_v3_for_day(v2_path: Path, trades_path: Path, day_str: str) -> pd.DataFrame:
    v2 = pd.read_parquet(v2_path)
    if not trades_path.exists():
        # mark trades cols as NaN, return v2 unchanged
        v2["tr_count"] = np.nan
        return v2
    tr = trades_features_for_day(trades_path, day_str)
    # Merge on bar_idx; compute tr_vwap_bp_dev using v2's mid_close
    merged = v2.merge(tr, on="bar_idx", how="left")
    merged["tr_vwap_bp_dev"] = ((merged["tr_vwap"] - merged["mid_close"]) /
                                 merged["mid_close"].replace(0, np.nan) * 10000)
    merged["tr_vwap_bp_dev"] = merged["tr_vwap_bp_dev"].replace([np.inf, -np.inf], np.nan)
    # Fill trades cols with 0 for bars with no trades (rare for ETH)
    tr_cols = [c for c in merged.columns if c.startswith("tr_")]
    merged[tr_cols] = merged[tr_cols].fillna(0)
    return merged


def worker(task):
    sym, ds, v2_path, trades_path, out_path = task
    t0 = time.time()
    try:
        df = build_v3_for_day(v2_path, trades_path, ds)
        df.to_parquet(out_path, compression="zstd", index=False)
        return (sym, ds, time.time() - t0, len(df), out_path.stat().st_size, "ok")
    except Exception as e:
        return (sym, ds, time.time() - t0, 0, 0, f"FAIL: {type(e).__name__}: {e}")


def discover(v2_root: Path, trades_root: Path, out_root: Path, symbols: list) -> list:
    tasks = []
    for sym in symbols:
        v2_dir = v2_root / sym
        tr_dir = trades_root / sym
        out_dir = out_root / sym
        out_dir.mkdir(parents=True, exist_ok=True)
        if not v2_dir.exists():
            continue
        for f in sorted(v2_dir.iterdir()):
            m = DATE_RE.match(f.name)
            if not m: continue
            ds = m.group(1)
            out_path = out_dir / f.name
            if out_path.exists() and out_path.stat().st_size > 0:
                continue
            trades_path = tr_dir / f.name
            tasks.append((sym, ds, f, trades_path, out_path))
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SOLUSDT")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    p.add_argument("--trades-root", type=Path, default=DEFAULT_TRADES_ROOT)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--require-trades", action="store_true",
                   help="Skip dates where trades parquet missing")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tasks = discover(args.v2_root, args.trades_root, args.out_root, syms)
    if args.require_trades:
        tasks = [t for t in tasks if t[3].exists()]
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
                if done % 100 == 0 or done == n:
                    rate = done / (time.time() - t0)
                    log.info(f"  [{done}/{n}] {sym} {ds} {n_bars}bars ({dt:.1f}s) | "
                             f"rate {rate*60:.0f}/min | size {total_bytes/1024/1024:.0f}MB")
    log.info(f"DONE. {done-failed} ok / {failed} fail | {total_bars} bars / {total_bytes/1024/1024:.0f}MB | {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
