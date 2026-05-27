"""
Bybit V5 REST wrapper — READ-ONLY first, write functions exist but DO NOT call.

⚠️ Schema risk: field names / units / side conventions in V5 can differ from docs.
Approach: log raw JSON for every signed call so we can VERIFY mapping against
expected ExchangeState format. Mapping assumptions are flagged with ⚠️ — confirm
against real responses before using in any decision.

Auth: HMAC-SHA256 signature on (timestamp + apiKey + recvWindow + payload).
  https://bybit-exchange.github.io/docs/v5/intro#authentication

Endpoints used:
  GET  /v5/account/wallet-balance?accountType=UNIFIED
  GET  /v5/position/list?category=linear&symbol=ETHUSDT
  GET  /v5/order/realtime?category=linear&symbol=ETHUSDT
  POST /v5/order/create   (WRITE — implemented but DO NOT call until verified)
  POST /v5/order/cancel   (WRITE — implemented but DO NOT call until verified)

Env vars:
  MARK19_BYBIT_KEY     = API key
  MARK19_BYBIT_SECRET  = secret
  MARK19_BYBIT_BASE    = api.bybit.com  (default; mainnet only, no testnet)
  MARK19_RAW_LOG_DIR   = where to dump raw JSON responses (default ~/mark19_data/bybit_raw_logs)
"""
from __future__ import annotations
import hashlib, hmac, json, os, time, urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import urllib.request, urllib.error


BASE_URL = f"https://{os.environ.get('MARK19_BYBIT_BASE', 'api.bybit.com')}"
RECV_WINDOW = "5000"
RAW_LOG_DIR = Path(os.environ.get("MARK19_RAW_LOG_DIR", "/Users/mark/mark19_data/bybit_raw_logs"))
RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)


class BybitError(Exception):
    pass


def _sign(secret: str, timestamp: str, api_key: str, recv_window: str, payload: str) -> str:
    """V5 spec: HMAC-SHA256 of (timestamp + apiKey + recvWindow + payload)."""
    msg = timestamp + api_key + recv_window + payload
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()


def _log_raw(endpoint: str, params: dict, response: dict, http_code: int):
    """Dump raw JSON to disk — for schema verification."""
    fname = RAW_LOG_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')}_{endpoint.replace('/','_')}.json"
    with open(fname, 'w') as f:
        json.dump({
            'ts': datetime.now(timezone.utc).isoformat(),
            'endpoint': endpoint,
            'params': params,
            'http_code': http_code,
            'response': response,
        }, f, indent=2)


def _http_get(endpoint: str, params: dict, key: str, secret: str, timeout: int = 10) -> dict:
    """Signed GET. Logs raw response. Returns parsed JSON dict."""
    qs = urllib.parse.urlencode(sorted(params.items()))
    timestamp = str(int(time.time() * 1000))
    sig = _sign(secret, timestamp, key, RECV_WINDOW, qs)
    url = f"{BASE_URL}{endpoint}?{qs}"
    req = urllib.request.Request(url, headers={
        "X-BAPI-API-KEY": key,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": RECV_WINDOW,
        "X-BAPI-SIGN": sig,
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            code = r.getcode()
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = {"error": str(e), "code": e.code, "body": e.read().decode()[:500]}
        code = e.code
    _log_raw(endpoint, params, body, code)
    if code != 200:
        raise BybitError(f"HTTP {code}: {body}")
    if body.get("retCode") not in (0, None):
        raise BybitError(f"retCode {body.get('retCode')}: {body.get('retMsg', '')}")
    return body


def _http_post(endpoint: str, payload: dict, key: str, secret: str, timeout: int = 10) -> dict:
    """Signed POST. ⚠️ WRITE operation — use with caution."""
    body_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
    timestamp = str(int(time.time() * 1000))
    sig = _sign(secret, timestamp, key, RECV_WINDOW, body_str)
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, data=body_str.encode(), headers={
        "X-BAPI-API-KEY": key,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": RECV_WINDOW,
        "X-BAPI-SIGN": sig,
        "Content-Type": "application/json",
    }, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            code = r.getcode()
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = {"error": str(e), "code": e.code, "body": e.read().decode()[:500]}
        code = e.code
    _log_raw(endpoint + "_POST", payload, body, code)
    if code != 200:
        raise BybitError(f"HTTP {code}: {body}")
    if body.get("retCode") not in (0, None):
        raise BybitError(f"retCode {body.get('retCode')}: {body.get('retMsg', '')}")
    return body


# ============== Read-only fetch ==============
def get_wallet_balance(key: str, secret: str, account_type: str = "UNIFIED") -> dict:
    """Return raw response. Mapping done by caller."""
    return _http_get("/v5/account/wallet-balance", {"accountType": account_type}, key, secret)


def get_positions(key: str, secret: str, symbol: str = "ETHUSDT", category: str = "linear") -> dict:
    return _http_get("/v5/position/list", {"category": category, "symbol": symbol}, key, secret)


def get_open_orders(key: str, secret: str, symbol: str = "ETHUSDT", category: str = "linear") -> dict:
    return _http_get("/v5/order/realtime", {"category": category, "symbol": symbol}, key, secret)


# ============== Write (DO NOT call until mapping verified) ==============
def place_limit_order(key: str, secret: str, symbol: str, side: str, qty: str, price: str,
                      category: str = "linear", time_in_force: str = "PostOnly") -> dict:
    """
    ⚠️ WRITE — only invoke after read-only mapping verified + manual confirm.

    side: "Buy" | "Sell"  (V5 convention for linear perp)
    qty: ETH amount as STRING (Bybit expects strings)
    price: USD price as STRING
    timeInForce: "PostOnly" enforces maker-only (rejects if would taker)
    """
    payload = {
        "category": category,
        "symbol": symbol,
        "side": side,
        "orderType": "Limit",
        "qty": qty,
        "price": price,
        "timeInForce": time_in_force,
    }
    return _http_post("/v5/order/create", payload, key, secret)


def cancel_order(key: str, secret: str, symbol: str, order_id: str, category: str = "linear") -> dict:
    """⚠️ WRITE."""
    payload = {"category": category, "symbol": symbol, "orderId": order_id}
    return _http_post("/v5/order/cancel", payload, key, secret)


# ============== Execution / closed-PnL history (for realized P&L) ==============
def get_executions(key: str, secret: str, symbol: str = "ETHUSDT", category: str = "linear",
                    start_ms: int = None, end_ms: int = None, limit: int = 100) -> dict:
    """GET /v5/execution/list — recent fills (max 7-day window per call)."""
    params = {"category": category, "symbol": symbol, "limit": str(limit)}
    if start_ms is not None: params["startTime"] = str(start_ms)
    if end_ms is not None: params["endTime"] = str(end_ms)
    return _http_get("/v5/execution/list", params, key, secret)


def get_closed_pnl(key: str, secret: str, symbol: str = "ETHUSDT", category: str = "linear",
                    start_ms: int = None, end_ms: int = None, limit: int = 100) -> dict:
    """GET /v5/position/closed-pnl — realized P&L per closed trade."""
    params = {"category": category, "symbol": symbol, "limit": str(limit)}
    if start_ms is not None: params["startTime"] = str(start_ms)
    if end_ms is not None: params["endTime"] = str(end_ms)
    return _http_get("/v5/position/closed-pnl", params, key, secret)


# ============== Leverage + position-mode (must call BEFORE first trade) ==============
def set_leverage(key: str, secret: str, symbol: str, buy_leverage: int, sell_leverage: int,
                 category: str = "linear") -> dict:
    """
    Force leverage. MUST call before first live trade to ensure we trade at the
    leverage we expect (default account leverage may be 10x or higher).

    For ETHUSDT linear perp:
      buyLeverage / sellLeverage as STRINGS, e.g. "1" for 1x.
      retCode 110043 = "leverage not modified" (already set) — treated as success.
    """
    payload = {
        "category": category, "symbol": symbol,
        "buyLeverage": str(buy_leverage), "sellLeverage": str(sell_leverage),
    }
    try:
        return _http_post("/v5/position/set-leverage", payload, key, secret)
    except BybitError as e:
        if "110043" in str(e) or "not modified" in str(e).lower():
            return {"retCode": 0, "retMsg": "leverage already set", "note": str(e)}
        raise


def set_position_mode(key: str, secret: str, symbol: str, mode: int,
                       category: str = "linear") -> dict:
    """
    mode: 0 = one-way (single position per symbol), 3 = hedge (separate long/short).
    Verification phase enforces one-way (mode=0) to keep position accounting simple.
    """
    payload = {"category": category, "symbol": symbol, "mode": mode}
    return _http_post("/v5/position/switch-mode", payload, key, secret)


def get_position_info(key: str, secret: str, symbol: str = "ETHUSDT",
                       category: str = "linear") -> dict:
    """Includes leverage, positionMode, marginMode, liqPrice — for pre-trade verification."""
    return get_positions(key, secret, symbol, category)


# ============== Mapping verification (read-only flow) ==============
def fetch_state_and_map(key: str, secret: str, symbol: str = "ETHUSDT") -> dict:
    """
    Fetch all 3 read endpoints, log raw, and produce CANDIDATE mapping
    in the format Reconciler expects: {balance_usdt, positions[], open_orders[]}.

    ⚠️ MAPPING ASSUMPTIONS (verify against real responses):
      - Wallet: response.result.list[0].coin[?coin=='USDT'].availableToWithdraw → balance_usdt
        (Alt fields seen: walletBalance, availableBalance, equity. Pick deliberately.)
      - Position: response.result.list[*]
        - symbol = item.symbol
        - side   = item.side  ("Buy" | "Sell" | "None" when flat)
        - size   = float(item.size)  (ETH for linear perp)
        - avgPrice = float(item.avgPrice)
        - unrealisedPnl = float(item.unrealisedPnl)
      - Orders: response.result.list[*]
        - orderId = item.orderId
        - side = item.side, qty = float(item.qty), price = float(item.price)
        - status = item.orderStatus
    """
    raw = {
        "wallet": get_wallet_balance(key, secret),
        "positions": get_positions(key, secret, symbol),
        "orders": get_open_orders(key, secret, symbol),
    }

    # ⚠️ CANDIDATE MAPPING — verify these field paths against logged raw responses
    candidate = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "balance_usdt": None,
        "total_equity": None,
        "used_margin": None,
        "positions": [],
        "open_orders": [],
        "_mapping_warnings": [],
    }

    # Wallet — VERIFIED 2026-05-25 against live UNIFIED account response:
    #   account-level totalAvailableBalance = authoritative available (free for new orders)
    #   account-level totalEquity = wallet + unrealised
    #   account-level totalInitialMargin = locked by open positions/orders
    #   USDT coin.walletBalance is the same as totalEquity when only USDT held.
    #   USDT coin.availableToWithdraw was EMPTY STRING in test — do NOT rely on it.
    try:
        wl = raw["wallet"]["result"]["list"]
        if wl:
            acct = wl[0]
            # Account-level (most reliable for trading decisions)
            candidate["balance_usdt"] = float(acct.get("totalAvailableBalance", 0) or 0)
            candidate["total_equity"] = float(acct.get("totalEquity", 0) or 0)
            candidate["used_margin"] = float(acct.get("totalInitialMargin", 0) or 0)
            # Sanity vs USDT coin record
            coins = acct.get("coin", [])
            usdt = next((c for c in coins if c.get("coin") == "USDT"), None)
            if usdt is None:
                candidate["_mapping_warnings"].append("no USDT coin record in wallet")
            else:
                usdt_eq = float(usdt.get("walletBalance", 0) or 0)
                if abs(usdt_eq - candidate["total_equity"]) > 1.0:
                    candidate["_mapping_warnings"].append(
                        f"USDT.walletBalance {usdt_eq} differs from account.totalEquity {candidate['total_equity']} (multi-coin?)")
    except Exception as e:
        candidate["_mapping_warnings"].append(f"wallet parse: {e}")

    # Positions
    try:
        for p in raw["positions"]["result"]["list"]:
            side = p.get("side", "None")
            size = float(p.get("size", 0) or 0)
            if size > 0 and side in ("Buy", "Sell"):  # actively held
                candidate["positions"].append({
                    "symbol": p.get("symbol"),
                    "side": side,
                    "size": size,
                    "avgPrice": float(p.get("avgPrice", 0) or 0),
                    "unrealisedPnl": float(p.get("unrealisedPnl", 0) or 0),
                })
    except Exception as e:
        candidate["_mapping_warnings"].append(f"positions parse: {e}")

    # Orders
    try:
        for o in raw["orders"]["result"]["list"]:
            candidate["open_orders"].append({
                "orderId": o.get("orderId"),
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "qty": float(o.get("qty", 0) or 0),
                "price": float(o.get("price", 0) or 0),
                "status": o.get("orderStatus"),
            })
    except Exception as e:
        candidate["_mapping_warnings"].append(f"orders parse: {e}")

    return {"raw": raw, "mapped": candidate}


# TODO(macbook-bugs): merge defensive patches when historical bug list arrives.
# Anticipated: rate-limit handling, partial-fill semantics, side='None' edge cases,
# clientOrderId reuse rules, recvWindow drift, leverage settings, position-mode
# (one-way vs hedge), Unified vs Contract account distinctions.


if __name__ == "__main__":
    """Quick read-only verification script. Requires MARK19_BYBIT_KEY/SECRET env."""
    key = os.environ.get("MARK19_BYBIT_KEY")
    secret = os.environ.get("MARK19_BYBIT_SECRET")
    if not key or not secret:
        print("Set MARK19_BYBIT_KEY and MARK19_BYBIT_SECRET to run verification.")
        print("Currently testable WITHOUT keys: only public endpoints (see test_public.py).")
        raise SystemExit(0)

    print(f"=== Read-only V5 fetch (ETHUSDT linear perp) ===")
    print(f"Base: {BASE_URL}")
    print(f"Raw log dir: {RAW_LOG_DIR}")
    out = fetch_state_and_map(key, secret, "ETHUSDT")
    print(f"\n--- Mapped candidate (compare to your expected ExchangeState format) ---")
    print(json.dumps(out["mapped"], indent=2, default=str))
    print(f"\n--- Raw response keys ---")
    for k, v in out["raw"].items():
        print(f"  {k}: retCode={v.get('retCode')} retMsg={v.get('retMsg','')[:50]}")
        try:
            sample = v.get("result", {}).get("list", [])
            if sample:
                print(f"    first item keys: {list(sample[0].keys())[:15]}")
        except Exception:
            pass
    if out["mapped"].get("_mapping_warnings"):
        print(f"\n⚠️  Mapping warnings: {out['mapped']['_mapping_warnings']}")
