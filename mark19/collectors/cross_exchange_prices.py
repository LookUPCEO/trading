"""
Cross-exchange ETH price collector.

Polls 4 exchanges every POLL_INTERVAL seconds, buffers BUFFER_SIZE rows,
then writes to parquet via storage.write_append.

Design:
  - Each exchange has its own fetcher function (isolated errors)
  - Single collection cycle: gather all 4 in parallel
  - Failed exchange → null in that column
  - Timestamp = cycle start time (shared across row)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
import pandas as pd

from mark19.storage import write_append

logger = logging.getLogger(__name__)

POLL_INTERVAL = 10.0
BUFFER_SIZE = 60
HTTP_TIMEOUT = 5.0
SYMBOL = "ETHUSDT"
DATA_TYPE = "cross_exchange_prices"


async def fetch_bybit(client: httpx.AsyncClient) -> Optional[float]:
    try:
        r = await client.get(
            "https://api.bybit.com/v5/market/tickers",
            params={"category": "linear", "symbol": "ETHUSDT"},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        return float(data["result"]["list"][0]["lastPrice"])
    except Exception as e:
        logger.warning(f"bybit fetch failed: {e}")
        return None


async def fetch_binance(client: httpx.AsyncClient) -> Optional[float]:
    try:
        r = await client.get(
            "https://fapi.binance.com/fapi/v1/ticker/price",
            params={"symbol": "ETHUSDT"},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception as e:
        logger.warning(f"binance fetch failed: {e}")
        return None


async def fetch_okx(client: httpx.AsyncClient) -> Optional[float]:
    try:
        r = await client.get(
            "https://www.okx.com/api/v5/market/ticker",
            params={"instId": "ETH-USDT-SWAP"},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return float(r.json()["data"][0]["last"])
    except Exception as e:
        logger.warning(f"okx fetch failed: {e}")
        return None


async def fetch_upbit(client: httpx.AsyncClient) -> Optional[float]:
    try:
        r = await client.get(
            "https://api.upbit.com/v1/ticker",
            params={"markets": "KRW-ETH"},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return float(r.json()[0]["trade_price"])
    except Exception as e:
        logger.warning(f"upbit fetch failed: {e}")
        return None


async def collect_once(client: httpx.AsyncClient) -> dict:
    """One collection cycle: fetch all 4 in parallel."""
    timestamp = datetime.now(timezone.utc)

    bybit, binance, okx, upbit = await asyncio.gather(
        fetch_bybit(client),
        fetch_binance(client),
        fetch_okx(client),
        fetch_upbit(client),
        return_exceptions=False,
    )

    return {
        "timestamp": timestamp,
        "bybit_eth_usd": bybit,
        "binance_eth_usd": binance,
        "okx_eth_usd": okx,
        "upbit_eth_krw": upbit,
    }


def flush_buffer(buffer: list[dict]) -> None:
    """Write buffered rows to parquet."""
    if not buffer:
        return
    df = pd.DataFrame(buffer)
    write_append(
        df=df,
        data_type=DATA_TYPE,
        exchange="combined",
        symbol=SYMBOL,
    )
    logger.info(f"flushed {len(buffer)} rows")


async def run(
    poll_interval: float = POLL_INTERVAL,
    buffer_size: int = BUFFER_SIZE,
    max_cycles: Optional[int] = None,
) -> None:
    """
    Main collection loop with hang protection.

    - Per-cycle timeout (asyncio.wait_for)
    - Periodic httpx client recycling (every 100 cycles)
    """
    logger.info(
        f"starting cross-exchange price collector: "
        f"interval={poll_interval}s, buffer={buffer_size}"
    )

    buffer = []
    cycles = 0
    client = None
    cycles_with_current_client = 0
    CLIENT_LIFETIME = 100  # cycles
    CYCLE_TIMEOUT = 15.0   # seconds

    try:
        while True:
            cycle_start = asyncio.get_running_loop().time()

            # Recycle client periodically or if missing
            if client is None or cycles_with_current_client >= CLIENT_LIFETIME:
                if client is not None:
                    try:
                        await client.aclose()
                    except Exception:
                        pass
                client = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT))
                cycles_with_current_client = 0
                logger.info("httpx client recycled")

            # Fetch with hard timeout
            try:
                row = await asyncio.wait_for(
                    collect_once(client),
                    timeout=CYCLE_TIMEOUT,
                )
                buffer.append(row)
                cycles += 1
                cycles_with_current_client += 1

                non_null = sum(
                    1 for v in row.values()
                    if v is not None and not isinstance(v, datetime)
                )
                logger.info(
                    f"cycle {cycles}: ts={row['timestamp'].isoformat()} "
                    f"bybit={row['bybit_eth_usd']} "
                    f"binance={row['binance_eth_usd']} "
                    f"okx={row['okx_eth_usd']} "
                    f"upbit={row['upbit_eth_krw']} "
                    f"({non_null}/4 exchanges)"
                )
            except asyncio.TimeoutError:
                logger.warning(f"cycle timeout at {CYCLE_TIMEOUT}s, recycling client")
                # Force client recycle on next iter
                try:
                    await client.aclose()
                except Exception:
                    pass
                client = None
                cycles += 1  # count attempt
            except Exception as e:
                logger.warning(f"cycle error: {e}")
                cycles += 1

            if len(buffer) >= buffer_size:
                flush_buffer(buffer)
                buffer = []

            if max_cycles is not None and cycles >= max_cycles:
                break

            elapsed = asyncio.get_running_loop().time() - cycle_start
            sleep_for = max(0, poll_interval - elapsed)
            await asyncio.sleep(sleep_for)
    finally:
        if buffer:
            flush_buffer(buffer)
            logger.info(f"final flush {len(buffer)} rows")
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass
        logger.info(f"stopped after {cycles} cycles")
