"""
Bybit Trades WebSocket collector.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import websockets

from mark19.storage import write_append

logger = logging.getLogger(__name__)

WS_URL = "wss://stream.bybit.com/v5/public/linear"
SYMBOL = os.environ.get("BYBIT_SYMBOL", "ETHUSDT")
TOPIC = f"publicTrade.{SYMBOL}"
DATA_TYPE = "trades"
EXCHANGE = "bybit"

BUFFER_SIZE = 500
FLUSH_INTERVAL = 30.0
PING_INTERVAL = 20.0


def ms_to_utc(ms) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


def parse_trade(t: dict) -> dict:
    return {
        "timestamp": ms_to_utc(t["T"]),
        "symbol": t.get("s", SYMBOL),
        "side": t["S"],
        "price": float(t["p"]),
        "size": float(t["v"]),
        "trade_id": str(t["i"]),
        "tick_direction": t.get("L"),
        "block_trade": bool(t.get("BT", False)),
    }


async def flush_buffer(buffer: list[dict]) -> None:
    if not buffer:
        return
    df = pd.DataFrame(buffer)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    try:
        write_append(
            df, DATA_TYPE, EXCHANGE, SYMBOL,
            dedup_cols=["trade_id"],
        )
        logger.info(f"flushed {len(buffer)} trades")
    except Exception as e:
        logger.error(f"flush failed: {e}")


async def ws_handler(buffer: list, stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            logger.info(f"connecting to {WS_URL}")
            async with websockets.connect(
                WS_URL,
                ping_interval=PING_INTERVAL,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                await ws.send(json.dumps({
                    "op": "subscribe",
                    "args": [TOPIC],
                }))
                logger.info(f"subscribed to {TOPIC}")

                async for raw_msg in ws:
                    if stop_event.is_set():
                        break

                    try:
                        msg = json.loads(raw_msg)
                    except Exception:
                        continue

                    if msg.get("op") == "subscribe":
                        logger.info(f"subscribe ack: success={msg.get('success')}")
                        continue

                    if msg.get("topic") != TOPIC:
                        continue

                    trades = msg.get("data", [])
                    for t in trades:
                        try:
                            buffer.append(parse_trade(t))
                        except Exception as e:
                            logger.warning(f"parse error: {e}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"WS error, reconnecting in 3s: {e}")
            await asyncio.sleep(3)


async def flush_scheduler(
    buffer: list,
    stop_event: asyncio.Event,
    buffer_size: int = BUFFER_SIZE,
    flush_interval: float = FLUSH_INTERVAL,
):
    """Cancel-safe periodic flush."""
    last_flush = asyncio.get_running_loop().time()
    flush_count = 0

    try:
        while not stop_event.is_set():
            await asyncio.sleep(1.0)
            now = asyncio.get_running_loop().time()

            should_flush = (
                len(buffer) >= buffer_size or
                (len(buffer) > 0 and now - last_flush >= flush_interval)
            )

            if should_flush:
                to_flush = buffer[:]
                buffer.clear()
                await flush_buffer(to_flush)
                flush_count += 1
                last_flush = now
                buy_count = sum(1 for t in to_flush if t["side"] == "Buy")
                sell_count = len(to_flush) - buy_count
                total = buy_count + sell_count
                ratio = buy_count / total if total > 0 else 0
                logger.info(
                    f"flush #{flush_count}: buy={buy_count} sell={sell_count} "
                    f"ratio={ratio:.3f}"
                )
    finally:
        # Always flush remaining on exit (including CancelledError)
        if buffer:
            to_flush = buffer[:]
            buffer.clear()
            try:
                await flush_buffer(to_flush)
                logger.info(f"final flush {len(to_flush)} trades")
            except Exception as e:
                logger.error(f"final flush failed: {e}")


async def run(
    max_runtime: Optional[float] = None,
    buffer_size: int = BUFFER_SIZE,
    flush_interval: float = FLUSH_INTERVAL,
):
    buffer: list[dict] = []
    stop_event = asyncio.Event()

    tasks = [
        asyncio.create_task(ws_handler(buffer, stop_event)),
        asyncio.create_task(flush_scheduler(buffer, stop_event, buffer_size, flush_interval)),
    ]

    try:
        if max_runtime is not None:
            await asyncio.sleep(max_runtime)
            logger.info(f"max_runtime {max_runtime}s reached, stopping")
            stop_event.set()
        else:
            await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
