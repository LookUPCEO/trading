"""Mark19 Live Trading Bot Entry Point.

Usage:
    python -m live_bot.main --mode PAPER --max-minutes 5
    python -m live_bot.main --mode LIVE_SHADOW
    python -m live_bot.main --mode LIVE_SMALL_CAPITAL
    python -m live_bot.main --mode LIVE        # requires BOT_LIVE_OK=1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Project root on sys.path so that `live_bot.*` and `mark19.*` resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_bot.config import CFG, Mode
from live_bot.trading_bot import Mark19TradingBot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[Mode.PAPER, Mode.LIVE_SHADOW, Mode.LIVE_SMALL_CAPITAL, Mode.LIVE],
        default=Mode.PAPER,
        help="Trading mode (default PAPER)",
    )
    parser.add_argument("--model-path", default="models/mark17_v1.joblib")
    parser.add_argument("--max-minutes", type=int, default=None,
                        help="Stop after N minutes (None = forever)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    errors = CFG.validate()
    if errors and args.mode in [Mode.LIVE_SHADOW, Mode.LIVE_SMALL_CAPITAL, Mode.LIVE]:
        log.error("Config errors (required for live modes):")
        for e in errors:
            log.error(f"  {e}")
        sys.exit(1)

    bot = Mark19TradingBot(mode=args.mode, model_path=args.model_path)
    bot.run(max_minutes=args.max_minutes)


if __name__ == "__main__":
    main()
