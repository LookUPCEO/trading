"""Safe broker instantiation.

Rules (enforced here — cannot be bypassed by callers):
  * mode == "backtest" → PaperBroker (deterministic simulator)
  * mode == "paper"    → PaperBroker with live-stream-driven bars
  * mode == "live"     → LiveBroker, ONLY IF:
       - env var `BOT_LIVE_OK` is set to "1"
       - strategy's sqlite status is LIVE_SHADOW or LIVE_SMALL_CAPITAL
       - API credentials exist in env (BYBIT_API_KEY / BYBIT_API_SECRET)
    Any missing precondition raises — refuses to run.
"""
from __future__ import annotations

import os
from typing import Optional

from live_bot.state_store.db import open_db

from .base import Broker
from .paper import PaperBroker


_VALID_LIVE_STATUSES = ("LIVE_SHADOW", "LIVE_SMALL_CAPITAL")


def get_broker(mode: str, symbol: str, strategy_name: Optional[str] = None,
               initial_equity: float = 10_000.0) -> Broker:
    mode = mode.lower()
    if mode in ("backtest", "paper"):
        return PaperBroker(symbol=symbol, initial_equity=initial_equity)

    if mode != "live":
        raise ValueError(f"unknown broker mode: {mode!r}")

    # ---- live path — all of these must hold ----
    if os.getenv("BOT_LIVE_OK") != "1":
        raise PermissionError(
            "live broker refused: BOT_LIVE_OK env var is not set to '1'. "
            "This interlock prevents accidental live deployments."
        )
    if strategy_name is None:
        raise ValueError("live broker requires strategy_name for status lookup")
    with open_db() as db:
        row = db.get_strategy(strategy_name)
    if row is None:
        raise ValueError(f"unknown strategy {strategy_name!r} — register it first")
    if row["status"] not in _VALID_LIVE_STATUSES:
        raise PermissionError(
            f"live broker refused: strategy {strategy_name!r} status={row['status']}. "
            f"Only {_VALID_LIVE_STATUSES} may use the live broker."
        )
    if not (os.getenv("BYBIT_API_KEY") and os.getenv("BYBIT_API_SECRET")):
        raise PermissionError("live broker refused: missing BYBIT_API_KEY / BYBIT_API_SECRET")

    # Deferred import: live path depends on `requests` (already in project),
    # but we avoid importing execution.py into the paper path to keep it simple.
    from .live import LiveBroker
    return LiveBroker(symbol=symbol)
