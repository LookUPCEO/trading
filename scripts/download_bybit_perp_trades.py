"""
Download Bybit USDT Perpetual trades from public archive + convert to parquet.

URL : https://public.bybit.com/trading/{SYMBOL}/{SYMBOL}{date}.csv.gz
Schema: timestamp, symbol, side, size, price, tickDirection, trdMatchID,
        grossValue, homeNotional, foreignNotional

Output: {out_root}/{SYMBOL}/{YYYY-MM-DD}.parquet  (zstd)

- Discovers dates from existing 1Hz orderbook (~/mark19_data/{SYMBOL}/) so the
  trades coverage matches.
- Idempotent skip (existing parquet).
- Parallel download via ThreadPoolExecutor (I/O bound).
- gzip + parse + write parquet (zstd) in worker.
"""
from __future__ import annotations
import argparse, gzip, io, logging, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request, urllib.error

import pandas as pd


DEFAULT_OB_ROOT = Path("/Users/mark/mark19_data")
DEFAULT_OUT_ROOT = Path("/Users/mark/mark19_data/trades_perp")
URL_TMPL = "https://public.bybit.com/trading/{sym}/{sym}{date}.csv.gz"


def fetch_one(sym: str, date_str: str, out_path: Path,
              timeout: int = 60, retries: int = 3) -> tuple[bool, int, str]:
    """Download + decompress + write parquet. Returns (ok, size_bytes, msg)."""
    url = URL_TMPL.format(sym=sym, date=date_str)
    last_err = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mark19/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            # decompress
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                df = pd.read_csv(gz)
            # write parquet
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, compression="zstd", index=False)
            return True, out_path.stat().st_size, f"{len(df)} trades"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False, 0, "not_found"
            last_err = f"HTTP {e.code}"
            time.sleep(2)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2)
    return False, 0, f"FAIL: {last_err}"


def discover_tasks(ob_root: Path, out_root: Path, symbols: list,
                   start: str | None, end: str | None) -> list:
    tasks = []
    for sym in symbols:
        in_dir = ob_root / sym
        out_dir = out_root / sym
        if not in_dir.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(in_dir.iterdir()):
            m = re.match(r"^(\d{4}-\d{2}-\d{2})\.parquet$", f.name)
            if not m: continue
            date_str = m.group(1)
            if start and date_str < start: continue
            if end and date_str > end: continue
            out_path = out_dir / f.name
            if out_path.exists() and out_path.stat().st_size > 0:
                continue
            tasks.append((sym, date_str, out_path))
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SOLUSDT")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--start", default=None, help="YYYY-MM-DD inclusive lower bound")
    p.add_argument("--end", default=None, help="YYYY-MM-DD inclusive upper bound")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ob-root", type=Path, default=DEFAULT_OB_ROOT,
                   help="1Hz orderbook root to mirror date set from")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger()

    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tasks = discover_tasks(args.ob_root, args.out_root, syms, args.start, args.end)
    log.info(f"Discovered {len(tasks)} files (mirroring orderbook coverage)")
    if args.limit > 0:
        tasks = tasks[:args.limit]; log.info(f"  limited to {len(tasks)}")
    if not tasks:
        log.info("Nothing to do."); return

    t0 = time.time(); done = failed = nf = 0; bytes_total = 0
    n = len(tasks)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fut = {pool.submit(fetch_one, sym, ds, out): (sym, ds) for sym, ds, out in tasks}
        for f in as_completed(fut):
            sym, ds = fut[f]
            ok, sz, msg = f.result()
            done += 1
            if ok:
                bytes_total += sz
            elif msg == "not_found":
                nf += 1
            else:
                failed += 1
                log.warning(f"  {sym} {ds}: {msg}")
            if done % 25 == 0 or done == n:
                rate = done / (time.time() - t0)
                eta_min = (n - done) / rate / 60 if rate > 0 else 0
                log.info(f"  [{done}/{n}] ok={done-failed-nf} fail={failed} nf={nf} | "
                         f"rate {rate*60:.0f}/min | ETA {eta_min:.1f}min | size {bytes_total/1024/1024:.0f}MB")
    log.info(f"DONE. ok={done-failed-nf} fail={failed} nf={nf} | total {bytes_total/1024/1024:.0f}MB | {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
