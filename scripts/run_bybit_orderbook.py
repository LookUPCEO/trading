"""Entry point for Bybit order book collector."""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mark19.collectors.bybit_orderbook import run


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", type=int, default=60, help="rows before flush (default 60 = 1 min)")
    p.add_argument("--interval", type=float, default=1.0, help="snapshot interval seconds")
    p.add_argument("--max-runtime", type=float, default=None, help="stop after N seconds (testing)")
    args = p.parse_args()

    try:
        asyncio.run(run(
            max_runtime=args.max_runtime,
            buffer_size=args.buffer,
            snapshot_interval=args.interval,
        ))
    except KeyboardInterrupt:
        print("\nstopped by user")


if __name__ == "__main__":
    main()
