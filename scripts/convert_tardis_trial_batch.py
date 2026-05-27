"""Batch convert Tardis trial CSVs (4/29-5/7, 3 symbols × 4 types = 108 files) to mark19 parquet."""
import sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mark19.storage import write_append
from mark19.converters.tardis import (
    convert_trades, convert_liquidations,
    convert_book_snapshot_25, convert_derivative_ticker,
)

# (tardis name, convert fn, mark19 datatype)
CONVERSION_MAP = {
    "trades": (convert_trades, "trades"),
    "book_snapshot_25": (convert_book_snapshot_25, "orderbook"),
    "liquidations": (convert_liquidations, "liquidation"),
    "derivative_ticker": (convert_derivative_ticker, "derivative_ticker"),
}

INPUT_DIR = Path("/Users/dohun/Desktop/Mark/mark19/data/tardis_trial_raw")
SYMBOLS = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]
DATES = [f"2026-04-{d:02d}" for d in (29, 30)] + [f"2026-05-{d:02d}" for d in range(1, 8)]
EXCHANGE = "bybit_tardis_trial"  # new path, avoid macOS perm conflicts with existing bybit_tardis


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info(f"Tardis trial batch convert: {len(DATES)} days × {len(SYMBOLS)} symbols × {len(CONVERSION_MAP)} types")
    log.info("=" * 70)

    n_done = 0; n_skip = 0; n_fail = 0
    for sym in SYMBOLS:
        for date in DATES:
            for tardis_dt, (fn, mark19_dt) in CONVERSION_MAP.items():
                fname = f"bybit_{tardis_dt}_{date}_{sym}.csv.gz"
                fpath = INPUT_DIR / fname
                if not fpath.exists():
                    log.warning(f"  missing: {fname}")
                    n_fail += 1; continue
                try:
                    df_raw = pd.read_csv(fpath, compression="gzip")
                    df_conv = fn(df_raw)
                    write_append(df_conv, data_type=mark19_dt, exchange=EXCHANGE, symbol=sym)
                    log.info(f"  OK  {sym}/{tardis_dt}/{date}: {len(df_raw)} → {len(df_conv)} rows")
                    n_done += 1
                except Exception as e:
                    log.error(f"  FAIL {fname}: {e}")
                    n_fail += 1

    log.info(f"\nDone. {n_done} converted, {n_skip} skipped, {n_fail} failed (of {len(DATES)*len(SYMBOLS)*len(CONVERSION_MAP)})")


if __name__ == "__main__":
    main()
