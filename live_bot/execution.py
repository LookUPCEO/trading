"""Order execution: abstracts between simulated (backtest) and live (Bybit v5).

v5: two distinct exit models.
  - Strong-exit trend → split-entry (50/50) + single full TP at 2.8*ATR, SL at 1.3*ATR.
  - Weak-exit trend   → scalp model: single 100% entry, TP at 1.0*ATR, SL at 0.8*ATR.

Partial-TP and breakeven-SL migration are REMOVED.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests

from live_bot.config import CFG

log = logging.getLogger(__name__)

# ---- Entry scaling (strong trades only) ----
SPLIT_PULLBACK_ATR = 0.15
SPLIT_FIRST_PCT = 0.5
SPLIT_SECOND_PCT = 0.5

# ---- Single-entry exit parameters (used by volatility breakout) ----
WEAK_TP_ATR = 3.0
WEAK_SL_ATR = 1.5


# =========================================================
# Fill simulation primitives
# =========================================================

@dataclass
class PendingLimit:
    side: str
    price: float
    qty: float
    placed_bar: int
    expires_bar: int
    post_only: bool = True
    leg: str = "A"


def limit_fill_touched(pending: PendingLimit, bar_low: float, bar_high: float) -> bool:
    return bar_low <= pending.price <= bar_high


def apply_slippage(price: float, side: str, slippage_bp: float) -> float:
    bump = price * slippage_bp / 10_000
    if side == "long":
        return price - bump
    return price + bump


# =========================================================
# Entry-side helpers
# =========================================================

def split_leg_prices(side: str, primary: float, atr: float) -> Tuple[float, float]:
    offset = SPLIT_PULLBACK_ATR * atr
    if side == "long":
        return primary, primary - offset
    return primary, primary + offset


def split_leg_qty(full_qty: float, step: float = 0.001) -> Tuple[float, float]:
    half = full_qty * SPLIT_SECOND_PCT
    qty_b = round(half - (half % step), 3) if step else round(half, 3)
    qty_a = round(full_qty - qty_b, 3)
    return qty_a, qty_b


def build_entry_legs(side: str, primary: float, atr: float, full_qty: float,
                     placed_bar: int, timeout_bars: int,
                     strong_exit: bool) -> List[PendingLimit]:
    """Route to split (strong) or single-shot (weak) entry construction."""
    if strong_exit:
        p_a, p_b = split_leg_prices(side, primary, atr)
        q_a, q_b = split_leg_qty(full_qty)
        legs: List[PendingLimit] = []
        if q_a > 0:
            legs.append(PendingLimit(side=side, price=p_a, qty=q_a,
                                     placed_bar=placed_bar,
                                     expires_bar=placed_bar + timeout_bars, leg="A"))
        if q_b > 0:
            legs.append(PendingLimit(side=side, price=p_b, qty=q_b,
                                     placed_bar=placed_bar,
                                     expires_bar=placed_bar + timeout_bars, leg="B"))
        return legs
    # Weak: single 100% leg at the primary price.
    qty = round(full_qty, 3)
    if qty <= 0:
        return []
    return [PendingLimit(side=side, price=primary, qty=qty,
                         placed_bar=placed_bar,
                         expires_bar=placed_bar + timeout_bars, leg="A")]


def weighted_avg(legs: List[Tuple[float, float]]) -> Tuple[float, float]:
    total = sum(q for _, q in legs)
    if total <= 0:
        return 0.0, 0.0
    avg = sum(p * q for p, q in legs) / total
    return avg, total


# =========================================================
# Exit-side helpers
# =========================================================

def tp_price(side: str, avg_entry: float, atr: float, tp_atr_mult: float) -> float:
    return (avg_entry + tp_atr_mult * atr
            if side == "long"
            else avg_entry - tp_atr_mult * atr)


def sl_price(side: str, avg_entry: float, atr: float, sl_atr_mult: float) -> float:
    return (avg_entry - sl_atr_mult * atr
            if side == "long"
            else avg_entry + sl_atr_mult * atr)


def exit_multipliers(strong_exit: bool, cfg=CFG) -> Tuple[float, float]:
    """Return (tp_atr_mult, sl_atr_mult) for the chosen exit model."""
    if strong_exit:
        return cfg.tp_atr_strong, cfg.sl_atr
    return WEAK_TP_ATR, WEAK_SL_ATR


# =========================================================
# Live Bybit v5 REST client
# =========================================================

class BybitClient:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.base = "https://api-testnet.bybit.com" if cfg.testnet else cfg.base_url

    def _sign(self, params: str, ts: str) -> str:
        payload = ts + self.cfg.api_key + "5000" + params
        return hmac.new(self.cfg.api_secret.encode(),
                        payload.encode(), hashlib.sha256).hexdigest()

    def _headers(self, params: str) -> dict:
        ts = str(int(time.time() * 1000))
        return {
            "X-BAPI-API-KEY": self.cfg.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": "5000",
            "X-BAPI-SIGN": self._sign(params, ts),
            "Content-Type": "application/json",
        }

    def get_wallet_equity(self) -> float:
        url = f"{self.base}/v5/account/wallet-balance"
        params = "accountType=UNIFIED&coin=USDT"
        r = requests.get(url + "?" + params, headers=self._headers(params), timeout=10)
        r.raise_for_status()
        data = r.json()
        coins = data["result"]["list"][0]["coin"]
        for c in coins:
            if c["coin"] == "USDT":
                return float(c["equity"])
        return 0.0

    def place_limit(self, side: str, qty: float, price: float,
                    reduce_only: bool = False, post_only: bool = True,
                    order_link_id: Optional[str] = None) -> dict:
        url = f"{self.base}/v5/order/create"
        body = {
            "category": self.cfg.category,
            "symbol": self.cfg.symbol,
            "side": "Buy" if side == "long" else "Sell",
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(price),
            "timeInForce": "PostOnly" if post_only else "GTC",
            "reduceOnly": reduce_only,
        }
        if order_link_id:
            body["orderLinkId"] = order_link_id
        payload = json.dumps(body)
        r = requests.post(url, data=payload, headers=self._headers(payload), timeout=10)
        r.raise_for_status()
        return r.json()

    def place_market(self, side: str, qty: float,
                     reduce_only: bool = False,
                     order_link_id: Optional[str] = None) -> dict:
        """Place a real Market order on Bybit V5."""
        url = f"{self.base}/v5/order/create"
        body = {
            "category": self.cfg.category,
            "symbol": self.cfg.symbol,
            "side": "Buy" if side == "long" else "Sell",
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "IOC",
            "reduceOnly": reduce_only,
        }
        if order_link_id:
            body["orderLinkId"] = order_link_id
        payload = json.dumps(body)
        r = requests.post(url, data=payload, headers=self._headers(payload), timeout=10)
        r.raise_for_status()
        return r.json()

    def place_entry(self, side: str, full_qty: float, primary_price: float,
                    atr: float, strong_exit: bool) -> List[dict]:
        """Strong → two limit legs. Weak → one limit at primary."""
        resp = []
        if strong_exit:
            p_a, p_b = split_leg_prices(side, primary_price, atr)
            q_a, q_b = split_leg_qty(full_qty)
            if q_a > 0:
                resp.append(self.place_limit(side, q_a, round(p_a, 2),
                                             post_only=True, order_link_id="entryA"))
            if q_b > 0:
                resp.append(self.place_limit(side, q_b, round(p_b, 2),
                                             post_only=True, order_link_id="entryB"))
        else:
            resp.append(self.place_limit(side, round(full_qty, 3),
                                         round(primary_price, 2),
                                         post_only=True, order_link_id="entryW"))
        return resp

    def place_stop_market(self, side: str, qty: float, trigger_price: float,
                          order_link_id: Optional[str] = None) -> dict:
        url = f"{self.base}/v5/order/create"
        body = {
            "category": self.cfg.category,
            "symbol": self.cfg.symbol,
            "side": "Sell" if side == "long" else "Buy",
            "orderType": "Market",
            "qty": str(qty),
            "triggerPrice": str(trigger_price),
            "triggerBy": "LastPrice",
            "reduceOnly": True,
            "timeInForce": "IOC",
        }
        if order_link_id:
            body["orderLinkId"] = order_link_id
        payload = json.dumps(body)
        r = requests.post(url, data=payload, headers=self._headers(payload), timeout=10)
        r.raise_for_status()
        return r.json()

    def place_exit_suite(self, side: str, total_qty: float, avg_entry: float,
                         atr: float, strong_exit: bool) -> dict:
        """Single-TP exit stack for both modes.

        Strong → TP at 2.8*ATR (maker), SL at 1.3*ATR (taker stop).
        Weak   → TP at 1.0*ATR (maker), SL at 0.8*ATR (taker stop).
        """
        tp_mult, sl_mult = exit_multipliers(strong_exit, self.cfg)
        exit_side = "short" if side == "long" else "long"
        tp = tp_price(side, avg_entry, atr, tp_mult)
        sl = sl_price(side, avg_entry, atr, sl_mult)
        return {
            "tp": self.place_limit(exit_side, total_qty, round(tp, 2),
                                   reduce_only=True, post_only=True,
                                   order_link_id="tp_full"),
            "sl": self.place_stop_market(side, total_qty, round(sl, 2),
                                         order_link_id="sl_init"),
        }

    def set_trading_stop(self, sl_price: float, side: str = "long") -> dict:
        """Attach a Bybit-native stop-loss to the current position.

        Survives bot crashes — Bybit auto-closes if mark touches sl_price.
        side: "long"  → SL fires when price drops below sl_price
              "short" → SL fires when price rises above sl_price
        """
        url = f"{self.base}/v5/position/trading-stop"
        body = {
            "category": self.cfg.category,
            "symbol": self.cfg.symbol,
            "stopLoss": str(round(sl_price, 2)),
            "tpslMode": "Full",
            "slTriggerBy": "MarkPrice",
            "positionIdx": 0,  # one-way mode
        }
        payload = json.dumps(body)
        r = requests.post(url, data=payload, headers=self._headers(payload), timeout=10)
        return r.json()

    def cancel_by_link(self, order_link_id: str) -> dict:
        url = f"{self.base}/v5/order/cancel"
        body = {
            "category": self.cfg.category,
            "symbol": self.cfg.symbol,
            "orderLinkId": order_link_id,
        }
        payload = json.dumps(body)
        try:
            r = requests.post(url, data=payload, headers=self._headers(payload), timeout=10)
            return r.json()
        except Exception:
            return {}

    def cancel_all(self) -> dict:
        url = f"{self.base}/v5/order/cancel-all"
        body = {"category": self.cfg.category, "symbol": self.cfg.symbol}
        payload = json.dumps(body)
        r = requests.post(url, data=payload, headers=self._headers(payload), timeout=10)
        r.raise_for_status()
        return r.json()

    def get_position(self) -> Optional[dict]:
        url = f"{self.base}/v5/position/list"
        params = f"category={self.cfg.category}&symbol={self.cfg.symbol}"
        r = requests.get(url + "?" + params, headers=self._headers(params), timeout=10)
        r.raise_for_status()
        lst = r.json()["result"]["list"]
        return lst[0] if lst else None
