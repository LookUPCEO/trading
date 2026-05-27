"""Heartbeat watchdog — runs every 10 min from a systemd timer.

Reads state/heartbeat.csv, checks per-symbol freshness, writes an alert to the
audit table when a collector goes stale. Exit code 0 always (timer does not
need to page); alerts are surfaced via the dashboard's alert feed.

Optional: TELEGRAM_TOKEN + TELEGRAM_CHAT_ID env vars, if set, also push a
notification. Missing env = silent (no-op, no error).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live_bot.state_store import heartbeat    # noqa: E402
from live_bot.state_store.db import open_db    # noqa: E402

log = logging.getLogger("heartbeat_monitor")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")


def _notify_telegram(msg: str) -> None:
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg}, timeout=5,
        )
    except Exception as e:
        log.warning("telegram notify failed: %s", e)


def main() -> int:
    ap = argparse.ArgumentParser(description="Liquidation heartbeat monitor")
    ap.add_argument("--symbols", nargs="+", required=True)
    args = ap.parse_args()

    stale = heartbeat.stale_collectors(args.symbols)
    last = heartbeat.tail_last_per_symbol()
    log.info("heartbeat check: symbols=%s  stale=%s", args.symbols, stale)

    # Always audit the snapshot (useful for dashboard & post-mortems).
    with open_db() as db:
        db.audit("heartbeat_check", {"symbols": args.symbols,
                                     "stale": stale, "last": last})
        if stale:
            db.audit("collector_stale", {"stale": stale})

    if stale:
        _notify_telegram(f"WS liquidations stale: {', '.join(stale)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
