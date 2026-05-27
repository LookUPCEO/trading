"""
Convert Tardis CSV files to Mark19 parquet format.

Usage:
    python scripts/convert_tardis_to_mark19.py [--date YYYY-MM-DD] [--input-dir PATH]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mark19.storage import write_append
from mark19.converters.tardis import (
    convert_trades,
    convert_liquidations,
    convert_book_snapshot_25,
    convert_derivative_ticker,
)


CONVERSION_MAP = {
    "trades": (convert_trades, "trades", "bybit_tardis", "ETHUSDT"),
    "book_snapshot_25": (convert_book_snapshot_25, "orderbook", "bybit_tardis", "ETHUSDT"),
    "liquidations": (convert_liquidations, "liquidation", "bybit_tardis", "ETHUSDT"),
    "derivative_ticker": (convert_derivative_ticker, "derivative_ticker", "bybit_tardis", "ETHUSDT"),
}


def find_tardis_file(input_dir: Path, datatype: str, date: str, symbol: str) -> Path | None:
    pattern = f"bybit_{datatype}_{date}_{symbol}.csv.gz"
    candidate = input_dir / pattern
    if candidate.exists():
        return candidate
    return None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    p = argparse.ArgumentParser()
    p.add_argument("--date", default="2025-04-01")
    p.add_argument("--input-dir", default="/Users/dohun/Desktop/Mark/Mark18-R/data")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--types", default="all")
    args = p.parse_args()

    input_dir = Path(args.input_dir)

    if args.types == "all":
        datatypes = list(CONVERSION_MAP.keys())
    else:
        datatypes = [t.strip() for t in args.types.split(",")]

    print()
    print("=" * 70)
    print(f"Tardis → Mark19 Conversion")
    print(f"  Date: {args.date}")
    print(f"  Symbol: {args.symbol}")
    print(f"  Input: {input_dir}")
    print(f"  Types: {datatypes}")
    print("=" * 70)
    print()

    for dtype in datatypes:
        if dtype not in CONVERSION_MAP:
            log.warning(f"Unknown type: {dtype}")
            continue

        convert_fn, mark19_dtype, mark19_exchange, _ = CONVERSION_MAP[dtype]

        gz_file = find_tardis_file(input_dir, dtype, args.date, args.symbol)
        if gz_file is None:
            log.warning(f"File not found: {dtype} for {args.date}")
            continue

        log.info(f"Processing: {gz_file.name}")
        log.info(f"  Size: {gz_file.stat().st_size / 1024 / 1024:.1f} MB")

        log.info(f"  Reading CSV...")
        df_raw = pd.read_csv(gz_file, compression="gzip")
        log.info(f"  Loaded: {len(df_raw)} rows × {len(df_raw.columns)} cols")

        log.info(f"  Converting...")
        df_converted = convert_fn(df_raw)
        log.info(f"  Converted: {len(df_converted)} rows × {len(df_converted.columns)} cols")

        log.info(f"  Saving to {mark19_dtype}/{mark19_exchange}/{args.symbol}/")
        write_append(
            df_converted,
            data_type=mark19_dtype,
            exchange=mark19_exchange,
            symbol=args.symbol,
        )
        log.info(f"  Saved")

        print(f"\n  === {dtype} sanity ===")
        print(f"  shape: {df_converted.shape}")
        print(f"  time range: {df_converted['timestamp'].min()} → {df_converted['timestamp'].max()}")
        print(f"  columns: {list(df_converted.columns)[:8]}{'...' if len(df_converted.columns) > 8 else ''}")
        if "side" in df_converted.columns:
            print(f"  side dist: {df_converted['side'].value_counts().to_dict()}")
        if "price" in df_converted.columns:
            print(f"  price range: [{df_converted['price'].min():.2f}, {df_converted['price'].max():.2f}]")
        if dtype == "derivative_ticker":
            print(f"  funding_rate range: [{df_converted['funding_rate'].min():.6f}, {df_converted['funding_rate'].max():.6f}]")
            print(f"  open_interest range: [{df_converted['open_interest'].min():.0f}, {df_converted['open_interest'].max():.0f}]")
        if dtype == "book_snapshot_25":
            print(f"  bid_0_price range: [{df_converted['bid_0_price'].min():.2f}, {df_converted['bid_0_price'].max():.2f}]")
            print(f"  bid_25_price NaN: {df_converted['bid_25_price'].isna().all()} (expected True)")
        print()

    print("=" * 70)
    print("Done")
    print("=" * 70)


if __name__ == "__main__":
    main()
