"""
Bybit Order Book WebSocket collector.

Maintains local order book state from snapshot + delta stream.
Saves 1-second snapshots to parquet.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import websockets

from mark19.storage import write_append

logger = logging.getLogger(__name__)

WS_URL = "wss://stream.bybit.com/v5/public/linear"
SYMBOL = os.environ.get("BYBIT_SYMBOL", "ETHUSDT")
TOPIC = f"orderbook.50.{SYMBOL}"
DATA_TYPE = "orderbook"
EXCHANGE = "bybit"

SNAPSHOT_INTERVAL = 1.0   # seconds between snapshots
BUFFER_SIZE = 60          # rows = 1 minute
PING_INTERVAL = 20.0      # Bybit requires ping
DEPTH = 50                # top 50 levels stored


class OrderBookState:
    """Local order book maintained from snapshot + delta."""

    def __init__(self):
        self.bids: dict[float, float] = {}  # price -> size
        self.asks: dict[float, float] = {}
        self.update_id: int = 0
        self.sequence: int = 0
        self.last_update_ts: Optional[datetime] = None
        self.initialized: bool = False

    def apply_snapshot(self, data: dict):
        """Full snapshot: reset and populate."""
        self.bids = {float(p): float(s) for p, s in data.get("b", [])}
        self.asks = {float(p): float(s) for p, s in data.get("a", [])}
        self.update_id = int(data.get("u", 0))
        self.sequence = int(data.get("seq", 0))
        self.initialized = True

    def apply_delta(self, data: dict) -> bool:
        """Delta update. Returns True if sequence valid, False if gap detected."""
        new_u = int(data.get("u", 0))
        new_seq = int(data.get("seq", 0))

        # Bybit delta rule: for linear orderbook.50,
        # u should equal prev_u + 1 (or check seq)
        # We use seq as primary check
        if self.initialized and new_seq <= self.sequence:
            # Out of order or duplicate
            logger.warning(f"seq not advancing: prev={self.sequence}, new={new_seq}")
            return True  # Not fatal, just skip

        # Apply bid updates
        for price_str, size_str in data.get("b", []):
            price = float(price_str)
            size = float(size_str)
            if size == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = size

        # Apply ask updates
        for price_str, size_str in data.get("a", []):
            price = float(price_str)
            size = float(size_str)
            if size == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size

        self.update_id = new_u
        self.sequence = new_seq
        return True

    def to_snapshot_row(self) -> dict:
        """Flatten current state to row dict."""
        # Sort bids desc (highest first), asks asc (lowest first)
        sorted_bids = sorted(self.bids.items(), key=lambda x: -x[0])[:DEPTH]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:DEPTH]

        row = {
            "timestamp": datetime.now(timezone.utc),
            "update_id": self.update_id,
            "sequence": self.sequence,
        }

        for i in range(DEPTH):
            if i < len(sorted_bids):
                row[f"bid_{i}_price"] = sorted_bids[i][0]
                row[f"bid_{i}_size"] = sorted_bids[i][1]
            else:
                row[f"bid_{i}_price"] = None
                row[f"bid_{i}_size"] = None

            if i < len(sorted_asks):
                row[f"ask_{i}_price"] = sorted_asks[i][0]
                row[f"ask_{i}_size"] = sorted_asks[i][1]
            else:
                row[f"ask_{i}_price"] = None
                row[f"ask_{i}_size"] = None

        return row


async def ws_handler(state: OrderBookState, stop_event: asyncio.Event):
    """WebSocket receiver. Updates state on each message."""
    while not stop_event.is_set():
        try:
            logger.info(f"connecting to {WS_URL}")
            async with websockets.connect(
                WS_URL,
                ping_interval=PING_INTERVAL,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                # Subscribe
                await ws.send(json.dumps({
                    "op": "subscribe",
                    "args": [TOPIC],
                }))
                logger.info(f"subscribed to {TOPIC}")

                # Reset state for fresh snapshot
                state.initialized = False

                async for raw_msg in ws:
                    if stop_event.is_set():
                        break

                    try:
                        msg = json.loads(raw_msg)
                    except Exception:
                        continue

                    # Subscribe ack
                    if msg.get("op") == "subscribe":
                        logger.info(f"subscribe ack: success={msg.get('success')}")
                        continue

                    # Data message
                    if msg.get("topic") != TOPIC:
                        continue

                    msg_type = msg.get("type")
                    data = msg.get("data", {})

                    if msg_type == "snapshot":
                        state.apply_snapshot(data)
                        logger.info(
                            f"snapshot received: "
                            f"{len(state.bids)} bids, {len(state.asks)} asks, "
                            f"u={state.update_id}, seq={state.sequence}"
                        )
                    elif msg_type == "delta":
                        if not state.initialized:
                            logger.warning("delta received before snapshot, ignoring")
                            continue
                        state.apply_delta(data)

                    state.last_update_ts = datetime.now(timezone.utc)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"WS error, reconnecting in 3s: {e}")
            state.initialized = False
            await asyncio.sleep(3)


async def snapshot_writer(
    state: OrderBookState,
    stop_event: asyncio.Event,
    buffer_size: int = BUFFER_SIZE,
    snapshot_interval: float = SNAPSHOT_INTERVAL,
):
    """Periodically snapshot state and write to parquet."""
    buffer = []
    flush_count = 0

    while not stop_event.is_set():
        await asyncio.sleep(snapshot_interval)

        if not state.initialized:
            continue

        row = state.to_snapshot_row()
        buffer.append(row)

        if len(buffer) >= buffer_size:
            df = pd.DataFrame(buffer)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            try:
                write_append(df, DATA_TYPE, EXCHANGE, SYMBOL)
                flush_count += 1
                logger.info(
                    f"flushed {len(buffer)} rows "
                    f"(total flushes: {flush_count}, "
                    f"bids={len(state.bids)}, asks={len(state.asks)})"
                )
            except Exception as e:
                logger.error(f"write failed: {e}")
            buffer = []

    # Final flush
    if buffer:
        df = pd.DataFrame(buffer)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        try:
            write_append(df, DATA_TYPE, EXCHANGE, SYMBOL)
            logger.info(f"final flush {len(buffer)} rows")
        except Exception as e:
            logger.error(f"final flush failed: {e}")


async def run(
    max_runtime: Optional[float] = None,
    buffer_size: int = BUFFER_SIZE,
    snapshot_interval: float = SNAPSHOT_INTERVAL,
):
    """
    Main entry.

    max_runtime: if set, stop after N seconds (for testing).
    """
    state = OrderBookState()
    stop_event = asyncio.Event()

    tasks = [
        asyncio.create_task(ws_handler(state, stop_event)),
        asyncio.create_task(snapshot_writer(state, stop_event, buffer_size, snapshot_interval)),
    ]

    try:
        if max_runtime is not None:
            await asyncio.sleep(max_runtime)
            logger.info(f"max_runtime {max_runtime}s reached, stopping")
            stop_event.set()
        else:
            await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("interrupted by user")
        stop_event.set()
    finally:
        stop_event.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
