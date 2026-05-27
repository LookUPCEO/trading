"""
Convert Bybit raw delta-stream parquet → Mark19 50-level 1Hz snapshot schema.

Input  (per file, 1 day, 1 symbol):
    Columns: ts, cts, type ('snapshot'|'delta'), u, seq, bids (JSON), asks (JSON)
    Depth: 200 or 500 levels per side
    Rows : ~850k events/day

Output (Mark19 compatible):
    Columns: timestamp, update_id, sequence, bid_0_price..bid_49_size, ask_0_price..ask_49_size
    Rows   : 86400 (one per second; last event in each second)
    50 levels each side; levels > book depth → NaN
"""
from __future__ import annotations

import argparse, json, logging, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


N_LEVELS = 50


def reconstruct_day(in_path: Path) -> pd.DataFrame:
    """Read 1-day delta stream, reconstruct book, emit 1Hz top-50 snapshots."""
    log = logging.getLogger(__name__)

    t0 = time.time()
    tbl = pq.read_table(in_path, columns=["ts", "type", "u", "seq", "bids", "asks"])
    n = tbl.num_rows
    ts_arr = tbl.column("ts").to_numpy()
    type_arr = tbl.column("type").to_numpy(zero_copy_only=False)
    u_arr = tbl.column("u").to_numpy()
    seq_arr = tbl.column("seq").to_numpy()
    bids_arr = tbl.column("bids").to_pylist()
    asks_arr = tbl.column("asks").to_pylist()
    log.info(f"  loaded {n:,} events in {time.time()-t0:.1f}s")

    bid_book: Dict[float, float] = {}  # price → size
    ask_book: Dict[float, float] = {}

    # group by 1-second bucket; keep last event of each second
    sec_buckets: Dict[int, int] = {}  # second_ts → last row idx
    for i in range(n):
        sec = ts_arr[i] // 1000
        sec_buckets[int(sec)] = i

    keep_idx = set(sec_buckets.values())

    out_records = []
    last_emit_sec = None

    t0 = time.time()
    for i in range(n):
        kind = type_arr[i]
        bids_levels = json.loads(bids_arr[i]) if bids_arr[i] else []
        asks_levels = json.loads(asks_arr[i]) if asks_arr[i] else []

        if kind == "snapshot":
            bid_book.clear()
            ask_book.clear()
            for p, s in bids_levels:
                pf = float(p); sf = float(s)
                if sf > 0:
                    bid_book[pf] = sf
            for p, s in asks_levels:
                pf = float(p); sf = float(s)
                if sf > 0:
                    ask_book[pf] = sf
        else:  # delta
            for p, s in bids_levels:
                pf = float(p); sf = float(s)
                if sf == 0:
                    bid_book.pop(pf, None)
                else:
                    bid_book[pf] = sf
            for p, s in asks_levels:
                pf = float(p); sf = float(s)
                if sf == 0:
                    ask_book.pop(pf, None)
                else:
                    ask_book[pf] = sf

        if i in keep_idx:
            # sort bids desc, asks asc; take top N
            top_bids = sorted(bid_book.items(), key=lambda kv: -kv[0])[:N_LEVELS]
            top_asks = sorted(ask_book.items(), key=lambda kv: kv[0])[:N_LEVELS]

            rec: dict = {
                "timestamp": pd.Timestamp(int(ts_arr[i]), unit="ms", tz="UTC"),
                "update_id": int(u_arr[i]),
                "sequence": int(seq_arr[i]),
            }
            for k in range(N_LEVELS):
                if k < len(top_bids):
                    rec[f"bid_{k}_price"] = top_bids[k][0]
                    rec[f"bid_{k}_size"] = top_bids[k][1]
                else:
                    rec[f"bid_{k}_price"] = np.nan
                    rec[f"bid_{k}_size"] = np.nan
                if k < len(top_asks):
                    rec[f"ask_{k}_price"] = top_asks[k][0]
                    rec[f"ask_{k}_size"] = top_asks[k][1]
                else:
                    rec[f"ask_{k}_price"] = np.nan
                    rec[f"ask_{k}_size"] = np.nan
            out_records.append(rec)

    log.info(f"  reconstructed in {time.time()-t0:.1f}s, emitted {len(out_records):,} 1Hz snapshots")

    df = pd.DataFrame(out_records)
    # column order to match mark19 schema exactly
    cols = ["timestamp", "update_id", "sequence"]
    for k in range(N_LEVELS):
        cols += [f"bid_{k}_price", f"bid_{k}_size", f"ask_{k}_price", f"ask_{k}_size"]
    df = df[cols]
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to Bybit raw parquet (1 day)")
    p.add_argument("--output", required=True, help="Path to write Mark19-schema parquet")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Converting: {in_path.name}")
    df = reconstruct_day(in_path)
    log.info(f"  shape: {df.shape}")
    log.info(f"  ts range: {df.timestamp.min()} .. {df.timestamp.max()}")

    df.to_parquet(out_path, compression="zstd", index=False)
    out_size_mb = out_path.stat().st_size / 1024 / 1024
    log.info(f"  wrote {out_path} ({out_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
