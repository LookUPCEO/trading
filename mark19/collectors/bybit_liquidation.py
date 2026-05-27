"""
Bybit Liquidation WebSocket collector.

Note on 'side' field:
  Bybit documentation is inconsistent about interpretation.
  We store the raw 'side' value. Interpret during analysis by
  cross-checking with price movement (short liquidation → upward squeeze).
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
TOPIC = f"allLiquidation.{SYMBOL}"
DATA_TYPE = "liquidation"
EXCHANGE = "bybit"

BUFFER_SIZE = 50
FLUSH_INTERVAL = 60.0
PING_INTERVAL = 20.0


def ms_to_utc(ms) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


def parse_liquidation(d: dict) -> dict:
    """
    Parse allLiquidation event.

    Bybit allLiquidation fields (short-form):
      T: timestamp (ms)
      s: symbol
      S: side (Buy/Sell)
      v: size (volume)
      p: price
    """
    ts_ms = d.get("T")
    if ts_ms and int(ts_ms) > 0:
        event_ts = ms_to_utc(ts_ms)
    else:
        event_ts = datetime.now(timezone.utc)

    side = d.get("S", "")
    price = float(d["p"])
    size = float(d["v"])

    return {
        "timestamp": event_ts,
        "symbol": d.get("s", SYMBOL),
        "side": side,
        "price": price,
        "size": size,
        "notional": price * size,
        "received_at": datetime.now(timezone.utc),
    }


async def flush_buffer(buffer: list[dict]) -> None:
    if not buffer:
        return
    df = pd.DataFrame(buffer)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    try:
        write_append(
            df, DATA_TYPE, EXCHANGE, SYMBOL,
            dedup_cols=["timestamp", "side", "size", "price"],
        )
        total_notional = df["notional"].sum()
        logger.info(
            f"flushed {len(buffer)} liquidations, "
            f"total notional ${total_notional:,.0f}"
        )
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

                    data = msg.get("data")
                    if data is None:
                        continue

                    # Bybit may send dict (single) or list (batch)
                    if isinstance(data, dict):
                        events = [data]
                    elif isinstance(data, list):
                        events = data
                    else:
                        logger.warning(f"unexpected data type: {type(data)}")
                        continue

                    for event in events:
                        try:
                            parsed = parse_liquidation(event)
                            buffer.append(parsed)
                            logger.info(
                                f"liquidation: side={parsed['side']} "
                                f"size={parsed['size']:.4f} "
                                f"price={parsed['price']:.2f} "
                                f"notional=${parsed['notional']:,.0f}"
                            )
                        except Exception as e:
                            logger.warning(f"parse error: {e}, event={event}")

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
    finally:
        if buffer:
            to_flush = buffer[:]
            buffer.clear()
            try:
                await flush_buffer(to_flush)
                logger.info(f"final flush {len(to_flush)} liquidations")
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
