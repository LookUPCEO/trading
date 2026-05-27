"""
Batch convert Bybit raw delta parquet → Mark19 schema (parallel, idempotent).

- Input  : /Volumes/PortableSSD/bybit_data/parquet/{SYMBOL}/{YYYY-MM-DD}_{SYMBOL}_ob{200|500}.parquet
- Output : /Volumes/PortableSSD/bybit_data/parquet_mark19/{SYMBOL}/{YYYY-MM-DD}.parquet

Recent-first ordering (--priority-recent N) processes last N days per symbol first.
"""
from __future__ import annotations

import argparse, logging, os, sys, time, re
from datetime import datetime
from multiprocessing import Pool, current_process
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_bybit_raw_to_mark19 import reconstruct_day  # type: ignore


DEFAULT_RAW_ROOT = Path("/Volumes/PortableSSD/bybit_data/parquet")
DEFAULT_OUT_ROOT = Path("/Volumes/PortableSSD/bybit_data/parquet_mark19")
DEFAULT_LOG_DIR = Path("/Volumes/PortableSSD/bybit_data/logs_convert")
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_([A-Z]+)_ob\d+\.parquet$")


def discover_tasks(symbols: list[str], raw_root: Path, out_root: Path) -> list[tuple[str, str, Path, Path]]:
    """Return [(symbol, date_str, in_path, out_path), ...] for files needing conversion."""
    tasks = []
    for sym in symbols:
        in_dir = raw_root / sym
        out_dir = out_root / sym
        if not in_dir.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in in_dir.iterdir():
            m = DATE_RE.match(f.name)
            if not m:
                continue
            date_str, file_sym = m.group(1), m.group(2)
            if file_sym != sym:
                continue
            out_path = out_dir / f"{date_str}.parquet"
            if out_path.exists() and out_path.stat().st_size > 0:
                continue  # idempotent skip
            tasks.append((sym, date_str, f, out_path))
    return tasks


def order_tasks(tasks: list, priority_recent: int) -> list:
    """Sort: per-symbol last N days first (descending), then remainder ascending."""
    by_sym: dict[str, list] = {}
    for t in tasks:
        by_sym.setdefault(t[0], []).append(t)
    for sym in by_sym:
        by_sym[sym].sort(key=lambda t: t[1])  # ascending date

    priority = []
    remainder = []
    for sym in sorted(by_sym):
        days = by_sym[sym]
        if priority_recent > 0 and len(days) > priority_recent:
            priority.extend(reversed(days[-priority_recent:]))  # newest first
            remainder.extend(days[:-priority_recent])
        else:
            priority.extend(reversed(days))
    # interleave priority across symbols by zipping rounds for parallel work distribution
    return priority + remainder


def worker(task: tuple[str, str, Path, Path]) -> tuple[str, str, float, int, str]:
    sym, date_str, in_path, out_path = task
    t0 = time.time()
    try:
        df = reconstruct_day(in_path)
        df.to_parquet(out_path, compression="zstd", index=False)
        elapsed = time.time() - t0
        size_mb = out_path.stat().st_size / 1024 / 1024
        return (sym, date_str, elapsed, int(size_mb * 1024), "ok")
    except Exception as e:
        return (sym, date_str, time.time() - t0, 0, f"FAIL: {type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SOLUSDT")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--priority-recent", type=int, default=60,
                   help="Convert last N days first per symbol (0 = no priority)")
    p.add_argument("--limit", type=int, default=0, help="Stop after N tasks (0 = no limit)")
    p.add_argument("--log-name", default=None, help="Log filename prefix")
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help="Input root directory")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output root directory")
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Log directory")
    args = p.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    log_name = args.log_name or f"convert_{stamp}"
    log_path = args.log_dir / f"{log_name}.log"
    fail_path = args.log_dir / f"{log_name}.failed.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    log.info(f"Discovery: symbols={symbols} workers={args.workers} priority_recent={args.priority_recent}")
    log.info(f"  raw_root={args.raw_root}")
    log.info(f"  out_root={args.out_root}")
    log.info(f"  log_dir={args.log_dir}")
    tasks = discover_tasks(symbols, args.raw_root, args.out_root)
    log.info(f"  found {len(tasks)} tasks (pending conversion)")
    tasks = order_tasks(tasks, args.priority_recent)
    if args.limit > 0:
        tasks = tasks[:args.limit]
        log.info(f"  limited to first {len(tasks)}")
    if not tasks:
        log.info("Nothing to do."); return

    log.info(f"  first 5: {[(t[0],t[1]) for t in tasks[:5]]}")
    log.info(f"  last 5:  {[(t[0],t[1]) for t in tasks[-5:]]}")

    t_start = time.time()
    completed, failed, total_mb = 0, 0, 0
    n_total = len(tasks)

    with Pool(processes=args.workers, maxtasksperchild=20) as pool, \
         open(fail_path, "a") as fail_log:
        for sym, date_str, elapsed, kb, status in pool.imap_unordered(worker, tasks):
            completed += 1
            if status != "ok":
                failed += 1
                fail_log.write(f"{sym},{date_str},{status}\n")
                fail_log.flush()
                log.error(f"  [{completed}/{n_total}] {sym} {date_str} FAILED in {elapsed:.1f}s — {status}")
            else:
                total_mb += kb / 1024
                if completed % 5 == 0 or completed == n_total:
                    rate = completed / (time.time() - t_start)
                    eta_h = (n_total - completed) / rate / 3600 if rate > 0 else 0
                    log.info(f"  [{completed}/{n_total}] {sym} {date_str} ok in {elapsed:.1f}s "
                             f"({kb/1024:.1f} MB) | total {total_mb:.1f} MB | "
                             f"rate {rate*60:.1f}/min | ETA {eta_h:.1f}h")

    log.info(f"DONE. completed={completed}/{n_total} failed={failed} total_size={total_mb:.1f} MB "
             f"elapsed={(time.time()-t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
