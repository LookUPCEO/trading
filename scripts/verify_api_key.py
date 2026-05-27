"""Read-only verification of new Bybit API key.

Tests:
  1. wallet-balance (USDT equity)
  2. position/list (current positions, expected: empty)
  3. order/realtime (open orders, expected: empty)
  4. user/query-api (key permission/expiry info)

NO write/trade calls.
"""
import os, sys, time, hmac, hashlib, json
from pathlib import Path
import requests

# Load .env
env_path = Path(__file__).resolve().parent.parent / "live_bot" / ".env"
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("#") or not line or "=" not in line: continue
        k, v = line.split("=", 1)
        if k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip()

API_KEY = os.environ.get("BYBIT_API_KEY", "")
API_SECRET = os.environ.get("BYBIT_API_SECRET", "")
TESTNET = os.environ.get("BYBIT_TESTNET", "false").lower() == "true"

assert API_KEY, "BYBIT_API_KEY missing from .env"
assert API_SECRET, "BYBIT_API_SECRET missing"
print(f"API key prefix: {API_KEY[:6]}*** (len {len(API_KEY)})")
print(f"Testnet: {TESTNET}")

BASE = "https://api-testnet.bybit.com" if TESTNET else "https://api.bybit.com"


def signed_get(path, params=""):
    ts = str(int(time.time() * 1000))
    payload = ts + API_KEY + "5000" + params
    sign = hmac.new(API_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-RECV-WINDOW": "5000",
        "X-BAPI-SIGN": sign,
        "Content-Type": "application/json",
    }
    url = f"{BASE}{path}?{params}" if params else f"{BASE}{path}"
    r = requests.get(url, headers=headers, timeout=15)
    return r.status_code, r.json()


print("\n=== 1. wallet-balance (USDT) ===")
status, body = signed_get("/v5/account/wallet-balance", "accountType=UNIFIED&coin=USDT")
print(f"  HTTP {status}  retCode {body.get('retCode')}  retMsg {body.get('retMsg')}")
if body.get("retCode") == 0:
    coins = body["result"]["list"][0]["coin"]
    for c in coins:
        if c["coin"] == "USDT":
            print(f"  USDT equity: ${float(c['equity']):.4f}")
            print(f"  USDT walletBalance: ${float(c['walletBalance']):.4f}")
            print(f"  USDT availableToWithdraw: ${float(c.get('availableToWithdraw') or 0):.4f}")
else:
    print(f"  raw: {body}")

print("\n=== 2. position/list (current positions) ===")
status, body = signed_get("/v5/position/list", "category=linear&symbol=ETHUSDT")
print(f"  HTTP {status}  retCode {body.get('retCode')}  retMsg {body.get('retMsg')}")
if body.get("retCode") == 0:
    positions = body["result"]["list"]
    print(f"  N positions: {len(positions)}")
    for p in positions:
        size = float(p.get("size") or 0)
        if size > 0:
            print(f"    {p['symbol']} {p['side']} size={size} avgPrice={p.get('avgPrice')} unrealisedPnl={p.get('unrealisedPnl')}")
        else:
            print(f"    {p['symbol']}: empty (size 0)")
else:
    print(f"  raw: {body}")

print("\n=== 3. order/realtime (open orders) ===")
status, body = signed_get("/v5/order/realtime", "category=linear&symbol=ETHUSDT")
print(f"  HTTP {status}  retCode {body.get('retCode')}  retMsg {body.get('retMsg')}")
if body.get("retCode") == 0:
    orders = body["result"]["list"]
    print(f"  N open orders: {len(orders)}")
    for o in orders[:5]:
        print(f"    {o['symbol']} {o['side']} qty={o['qty']} @ {o['price']} type={o['orderType']}")
else:
    print(f"  raw: {body}")

print("\n=== 4. user/query-api (key info) ===")
status, body = signed_get("/v5/user/query-api", "")
print(f"  HTTP {status}  retCode {body.get('retCode')}  retMsg {body.get('retMsg')}")
if body.get("retCode") == 0:
    info = body.get("result", {})
    print(f"  ID: {info.get('id')}")
    print(f"  api_key prefix: {info.get('apiKey', '')[:6]}***")
    print(f"  Read-only: {info.get('readOnly')}")
    print(f"  Expired at: {info.get('expiredAt')}")
    print(f"  IPs: {info.get('ips')}")
    print(f"  Permissions: {info.get('permissions')}")
    print(f"  isMaster: {info.get('isMaster')}")
else:
    print(f"  raw: {body}")

print("\n=== Diagnosis ===")
def ok(s, b): return s == 200 and b.get("retCode") == 0

s_wallet, b_wallet = signed_get("/v5/account/wallet-balance", "accountType=UNIFIED&coin=USDT")
s_pos, b_pos = signed_get("/v5/position/list", "category=linear&symbol=ETHUSDT")
s_ord, b_ord = signed_get("/v5/order/realtime", "category=linear&symbol=ETHUSDT")
s_key, b_key = signed_get("/v5/user/query-api", "")

if all(ok(s, b) for s, b in [(s_wallet, b_wallet), (s_pos, b_pos), (s_ord, b_ord), (s_key, b_key)]):
    print("  ✅ All 4 read-only calls succeeded — API key + IP whitelist + permission OK")
else:
    failed = []
    if not ok(s_wallet, b_wallet): failed.append(f"wallet({s_wallet}/{b_wallet.get('retCode')})")
    if not ok(s_pos, b_pos): failed.append(f"position({s_pos}/{b_pos.get('retCode')})")
    if not ok(s_ord, b_ord): failed.append(f"order({s_ord}/{b_ord.get('retCode')})")
    if not ok(s_key, b_key): failed.append(f"keyinfo({s_key}/{b_key.get('retCode')})")
    print(f"  ❌ Failed: {failed}")
