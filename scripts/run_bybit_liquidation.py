"""Entry point for Bybit liquidation collector."""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mark19.collectors.bybit_liquidation import run


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", type=int, default=50)
    p.add_argument("--flush-interval", type=float, default=60.0)
    p.add_argument("--max-runtime", type=float, default=None)
    args = p.parse_args()

    try:
        asyncio.run(run(
            max_runtime=args.max_runtime,
            buffer_size=args.buffer,
            flush_interval=args.flush_interval,
        ))
    except KeyboardInterrupt:
        print("\nstopped by user")


if __name__ == "__main__":
    main()
