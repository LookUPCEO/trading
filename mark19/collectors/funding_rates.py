"""
Funding rate collector for Bybit, Binance, OKX.
Two functions:
  - collect_current(): snapshot current funding (every 10 min)
  - backfill_historical(start, end): fetch historical funding
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

HTTP_TIMEOUT = 10.0
SYMBOL = "ETHUSDT"
RETRY_SLEEP = 2.0


def ms_to_utc(ms) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


# ============================================================
# CURRENT (snapshot)
# ============================================================

async def fetch_bybit_current(client: httpx.AsyncClient) -> tuple[Optional[float], Optional[datetime]]:
    try:
        r = await client.get(
            "https://api.bybit.com/v5/market/tickers",
            params={"category": "linear", "symbol": "ETHUSDT"},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        d = r.json()["result"]["list"][0]
        return float(d["fundingRate"]), ms_to_utc(d["nextFundingTime"])
    except Exception as e:
        logger.warning(f"bybit current failed: {e}")
        return None, None


async def fetch_binance_current(client: httpx.AsyncClient) -> tuple[Optional[float], Optional[datetime]]:
    try:
        r = await client.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": "ETHUSDT"},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        d = r.json()
        return float(d["lastFundingRate"]), ms_to_utc(d["nextFundingTime"])
    except Exception as e:
        logger.warning(f"binance current failed: {e}")
        return None, None


async def fetch_okx_current(client: httpx.AsyncClient) -> tuple[Optional[float], Optional[datetime]]:
    try:
        r = await client.get(
            "https://www.okx.com/api/v5/public/funding-rate",
            params={"instId": "ETH-USDT-SWAP"},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        j = r.json()
        if j.get("code") != "0":
            logger.warning(f"okx api error: {j.get('msg')}")
            return None, None
        d = j["data"][0]
        return float(d["fundingRate"]), ms_to_utc(d["nextFundingTime"])
    except Exception as e:
        logger.warning(f"okx current failed: {e}")
        return None, None


async def collect_current_once(client: httpx.AsyncClient) -> dict:
    timestamp = datetime.now(timezone.utc)
    (by_f, by_t), (bn_f, bn_t), (ok_f, ok_t) = await asyncio.gather(
        fetch_bybit_current(client),
        fetch_binance_current(client),
        fetch_okx_current(client),
    )
    return {
        "timestamp": timestamp,
        "symbol": SYMBOL,
        "bybit_funding": by_f,
        "bybit_next_time": by_t,
        "binance_funding": bn_f,
        "binance_next_time": bn_t,
        "okx_funding": ok_f,
        "okx_next_time": ok_t,
    }


def _prepare_current_df(buffer: list[dict]) -> pd.DataFrame:
    """Ensure datetime columns are proper dtype (handles None)."""
    df = pd.DataFrame(buffer)
    for col in ["timestamp", "bybit_next_time", "binance_next_time", "okx_next_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


async def run_current(
    poll_interval: float = 600.0,
    buffer_size: int = 6,
    max_cycles: Optional[int] = None,
) -> None:
    """
    Main loop with hang protection.

    - Per-cycle timeout (asyncio.wait_for)
    - Periodic httpx client recycling
    """
    logger.info(f"starting funding current collector: interval={poll_interval}s")

    buffer = []
    cycles = 0
    client = None
    cycles_with_current_client = 0
    CLIENT_LIFETIME = 100  # cycles (~17 hours at 10min interval)
    CYCLE_TIMEOUT = 30.0   # seconds (longer than cross-exchange's 15s)

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
                    collect_current_once(client),
                    timeout=CYCLE_TIMEOUT,
                )
                buffer.append(row)
                cycles += 1
                cycles_with_current_client += 1

                logger.info(
                    f"current cycle {cycles}: "
                    f"bybit={row['bybit_funding']} "
                    f"binance={row['binance_funding']} "
                    f"okx={row['okx_funding']}"
                )
            except asyncio.TimeoutError:
                logger.warning(f"cycle timeout at {CYCLE_TIMEOUT}s, recycling client")
                try:
                    await client.aclose()
                except Exception:
                    pass
                client = None
                cycles += 1
            except Exception as e:
                logger.warning(f"cycle error: {e}")
                cycles += 1

            if len(buffer) >= buffer_size:
                df = _prepare_current_df(buffer)
                write_append(df, "funding_current", "combined", SYMBOL)
                logger.info(f"flushed {len(buffer)} rows")
                buffer = []

            if max_cycles is not None and cycles >= max_cycles:
                break

            elapsed = asyncio.get_running_loop().time() - cycle_start
            sleep_for = max(0, poll_interval - elapsed)
            await asyncio.sleep(sleep_for)
    finally:
        if buffer:
            df = _prepare_current_df(buffer)
            write_append(df, "funding_current", "combined", SYMBOL)
            logger.info(f"final flush {len(buffer)} rows")
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass
        logger.info(f"stopped after {cycles} cycles")


# ============================================================
# HISTORICAL (backfill)
# ============================================================

async def backfill_bybit(
    client: httpx.AsyncClient,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """Bybit: newest-first pagination."""
    results = []
    cursor_end = int(end.timestamp() * 1000)
    start_ms = int(start.timestamp() * 1000)
    max_calls = 200

    for call_i in range(max_calls):
        try:
            r = await client.get(
                "https://api.bybit.com/v5/market/funding/history",
                params={
                    "category": "linear",
                    "symbol": "ETHUSDT",
                    "startTime": start_ms,
                    "endTime": cursor_end,
                    "limit": 200,
                },
                timeout=HTTP_TIMEOUT,
            )
            r.raise_for_status()
            j = r.json()
            if j.get("retCode") != 0:
                logger.warning(f"bybit api error: {j.get('retMsg')}")
                break
            rows = j.get("result", {}).get("list", [])
        except Exception as e:
            logger.warning(f"bybit backfill call {call_i}: {e}")
            await asyncio.sleep(RETRY_SLEEP)
            continue

        if not rows:
            break

        # Defensive parsing: try both possible field names
        timestamps_collected = []
        for row in rows:
            ts_ms = row.get("fundingRateTimestamp") or row.get("timestamp")
            if ts_ms is None:
                logger.warning(f"bybit row missing timestamp: {row}")
                continue
            ts_ms = int(ts_ms)
            timestamps_collected.append(ts_ms)
            results.append({
                "timestamp": ms_to_utc(ts_ms),
                "exchange": "bybit",
                "symbol": SYMBOL,
                "funding_rate": float(row["fundingRate"]),
            })

        if not timestamps_collected:
            break

        oldest_ms = min(timestamps_collected)
        if oldest_ms <= start_ms:
            break
        cursor_end = oldest_ms - 1
        await asyncio.sleep(0.2)

    return results


async def backfill_binance(
    client: httpx.AsyncClient,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """Binance: oldest-first, paginate forward."""
    results = []
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    cursor_start = start_ms
    max_calls = 200

    for call_i in range(max_calls):
        try:
            r = await client.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={
                    "symbol": "ETHUSDT",
                    "startTime": cursor_start,
                    "endTime": end_ms,
                    "limit": 1000,
                },
                timeout=HTTP_TIMEOUT,
            )
            r.raise_for_status()
            rows = r.json()
        except Exception as e:
            logger.warning(f"binance backfill call {call_i}: {e}")
            await asyncio.sleep(RETRY_SLEEP)
            continue

        if not rows:
            break

        for row in rows:
            results.append({
                "timestamp": ms_to_utc(row["fundingTime"]),
                "exchange": "binance",
                "symbol": SYMBOL,
                "funding_rate": float(row["fundingRate"]),
            })

        newest_ms = max(int(row["fundingTime"]) for row in rows)
        if len(rows) < 1000 or newest_ms >= end_ms:
            break
        cursor_start = newest_ms + 1
        await asyncio.sleep(0.2)

    return results


async def backfill_okx(
    client: httpx.AsyncClient,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """
    OKX: paginate backward with 'after' (moves to older).
    Note: OKX historical funding may not reach beyond ~3 months back.
    """
    results = []
    start_ms = int(start.timestamp() * 1000)
    cursor_after = int(end.timestamp() * 1000)
    max_calls = 500

    for call_i in range(max_calls):
        try:
            r = await client.get(
                "https://www.okx.com/api/v5/public/funding-rate-history",
                params={
                    "instId": "ETH-USDT-SWAP",
                    "before": start_ms,
                    "after": cursor_after,
                    "limit": 100,
                },
                timeout=HTTP_TIMEOUT,
            )
            r.raise_for_status()
            j = r.json()
            if j.get("code") != "0":
                logger.warning(f"okx api error: {j.get('msg')}")
                break
            data = j.get("data", [])
        except Exception as e:
            logger.warning(f"okx backfill call {call_i}: {e}")
            await asyncio.sleep(RETRY_SLEEP)
            continue

        if not data:
            break

        for row in data:
            results.append({
                "timestamp": ms_to_utc(row["fundingTime"]),
                "exchange": "okx",
                "symbol": SYMBOL,
                "funding_rate": float(row["fundingRate"]),
            })

        timestamps = [int(row["fundingTime"]) for row in data]
        oldest_ms = min(timestamps)
        if oldest_ms <= start_ms:
            break
        cursor_after = oldest_ms - 1
        await asyncio.sleep(0.3)

    return results


async def run_backfill(start: datetime, end: datetime) -> dict:
    """Backfill historical funding for all 3 exchanges."""
    logger.info(f"backfill: {start.isoformat()} to {end.isoformat()}")

    async with httpx.AsyncClient() as client:
        bybit, binance, okx = await asyncio.gather(
            backfill_bybit(client, start, end),
            backfill_binance(client, start, end),
            backfill_okx(client, start, end),
        )

    logger.info(f"fetched: bybit={len(bybit)}, binance={len(binance)}, okx={len(okx)}")

    stats = {}
    for ex_name, rows in [("bybit", bybit), ("binance", binance), ("okx", okx)]:
        if not rows:
            stats[ex_name] = 0
            continue
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        write_append(
            df, "funding_historical", ex_name, SYMBOL,
            dedup_cols=["timestamp", "exchange"],
        )
        stats[ex_name] = len(df)
        logger.info(f"{ex_name}: wrote {len(df)} rows")

    return stats
