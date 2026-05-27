"""Entry point for cross-exchange price collector."""
import asyncio
import logging
import sys
from pathlib import Path

# Ensure mark19 is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mark19.collectors.cross_exchange_prices import run


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=10.0,
                        help="poll interval seconds")
    parser.add_argument("--buffer", type=int, default=60,
                        help="buffer size before flush")
    parser.add_argument("--max-cycles", type=int, default=None,
                        help="stop after N cycles (for testing)")
    args = parser.parse_args()

    try:
        asyncio.run(run(
            poll_interval=args.interval,
            buffer_size=args.buffer,
            max_cycles=args.max_cycles,
        ))
    except KeyboardInterrupt:
        print("\nstopped by user")


if __name__ == "__main__":
    main()
