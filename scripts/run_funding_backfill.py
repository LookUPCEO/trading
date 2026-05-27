"""Entry point for funding historical backfill."""
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mark19.collectors.funding_rates import run_backfill


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    stats = asyncio.run(run_backfill(start, end))
    print(f"\nBackfill complete: {stats}")


if __name__ == "__main__":
    main()
