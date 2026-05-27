"""Entry point for funding current snapshot collector."""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mark19.collectors.funding_rates import run_current


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=600.0)
    p.add_argument("--buffer", type=int, default=6)
    p.add_argument("--max-cycles", type=int, default=None)
    args = p.parse_args()

    try:
        asyncio.run(run_current(
            poll_interval=args.interval,
            buffer_size=args.buffer,
            max_cycles=args.max_cycles,
        ))
    except KeyboardInterrupt:
        print("\nstopped by user")


if __name__ == "__main__":
    main()
