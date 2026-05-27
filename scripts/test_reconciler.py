"""
Self-tests for Reconciler / RiskRail / OrderManager.

Goal: prove that the reconciliation layer DOES catch known bad states
before they reach a live exchange. testnet 없으니 이게 유일한 검증.
"""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mark19_live as ml
from mark19_live import (RiskRail, Reconciler, ExchangeAdapter, OrderManager,
                          InternalBook, ExchangeState, Decision)


def make_log():
    log = logging.getLogger("test")
    log.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("    %(message)s"))
    log.addHandler(h); log.setLevel(logging.INFO)
    return log


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def assert_eq(name, actual, expected):
    ok = actual == expected
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name}: {actual} (expected {expected})")
    return ok


def assert_true(name, cond):
    mark = "✓" if cond else "✗"
    print(f"  {mark} {name}")
    return cond


# ============== Test 1: Reconciler catches BALANCE mismatch ==============
def test_balance_mismatch():
    section("TEST 1: Reconciler catches BALANCE mismatch")
    log = make_log()
    rail = RiskRail("shadow", log)
    rec = Reconciler(log, rail)
    internal = InternalBook(balance_usdt=1000.0)
    exchange = ExchangeState(ts="now", balance_usdt=950.0, total_equity=950, used_margin=0,
                              positions=[], open_orders=[])
    ok, diffs = rec.check(internal, exchange)
    assert_true("mismatch detected", not ok)
    assert_true("rail halted", rail.halted)
    assert_true("balance diff in message", any("BALANCE" in d for d in diffs))


# ============== Test 2: Reconciler catches POSITION SIZE drift ==============
def test_position_size_drift():
    section("TEST 2: Reconciler catches POSITION SIZE drift (silent partial fill)")
    log = make_log()
    rail = RiskRail("shadow", log)
    rec = Reconciler(log, rail)
    internal = InternalBook(balance_usdt=1000,
                             positions=[{'symbol':'ETHUSDT','side':'Buy','size':0.01,'avgPrice':2000}])
    # Exchange says we only got partial fill 0.007 ETH
    exchange = ExchangeState(ts="now", balance_usdt=1000, total_equity=1000, used_margin=14,
                              positions=[{'symbol':'ETHUSDT','side':'Buy','size':0.007,'avgPrice':2000,'unrealisedPnl':0}],
                              open_orders=[])
    ok, diffs = rec.check(internal, exchange)
    assert_true("mismatch detected", not ok)
    assert_true("size diff in message", any("SIZE" in d for d in diffs))


# ============== Test 3: Reconciler catches WRONG SIDE ==============
def test_wrong_side():
    section("TEST 3: Reconciler catches WRONG SIDE (logic bug)")
    log = make_log()
    rail = RiskRail("shadow", log)
    rec = Reconciler(log, rail)
    internal = InternalBook(balance_usdt=1000,
                             positions=[{'symbol':'ETHUSDT','side':'Buy','size':0.01,'avgPrice':2000}])
    exchange = ExchangeState(ts="now", balance_usdt=1000, total_equity=1000, used_margin=20,
                              positions=[{'symbol':'ETHUSDT','side':'Sell','size':0.01,'avgPrice':2000,'unrealisedPnl':0}],
                              open_orders=[])
    ok, diffs = rec.check(internal, exchange)
    assert_true("mismatch detected", not ok)
    assert_true("side diff in message", any("SIDE" in d for d in diffs))


# ============== Test 4: Reconciler catches UNTRACKED ORDER (external/duplicate) ==============
def test_untracked_order():
    section("TEST 4: Reconciler catches UNTRACKED ORDER (external/duplicate placement)")
    log = make_log()
    rail = RiskRail("shadow", log)
    rec = Reconciler(log, rail)
    internal = InternalBook(balance_usdt=1000, open_order_ids={"order-1"})
    exchange = ExchangeState(ts="now", balance_usdt=1000, total_equity=1000, used_margin=0,
                              positions=[],
                              open_orders=[{'orderId':'order-1','symbol':'ETHUSDT','side':'Buy','qty':0.01,'price':2000,'status':'New'},
                                            {'orderId':'order-999','symbol':'ETHUSDT','side':'Sell','qty':0.05,'price':2050,'status':'New'}])
    ok, diffs = rec.check(internal, exchange)
    assert_true("mismatch detected", not ok)
    assert_true("untracked order flagged", any("untracked" in d for d in diffs))


# ============== Test 5: Reconciler catches DISAPPEARED ORDER ==============
def test_disappeared_order():
    section("TEST 5: Reconciler catches DISAPPEARED ORDER (silent fill or cancel)")
    log = make_log()
    rail = RiskRail("shadow", log)
    rec = Reconciler(log, rail)
    internal = InternalBook(balance_usdt=1000, open_order_ids={"order-1", "order-2"})
    exchange = ExchangeState(ts="now", balance_usdt=1000, total_equity=1000, used_margin=0,
                              positions=[],
                              open_orders=[{'orderId':'order-1','symbol':'ETHUSDT','side':'Buy','qty':0.01,'price':2000,'status':'New'}])
    ok, diffs = rec.check(internal, exchange)
    assert_true("mismatch detected", not ok)
    assert_true("disappeared order flagged", any("disappeared" in d for d in diffs))


# ============== Test 6: Reconciler PASSES on aligned state ==============
def test_aligned_state():
    section("TEST 6: Reconciler PASSES when internal == exchange")
    log = make_log()
    rail = RiskRail("shadow", log)
    rec = Reconciler(log, rail)
    internal = InternalBook(balance_usdt=995.50,
                             positions=[{'symbol':'ETHUSDT','side':'Buy','size':0.01,'avgPrice':2000}],
                             open_order_ids={"order-1"})
    exchange = ExchangeState(ts="now", balance_usdt=995.50, total_equity=995.50, used_margin=20,
                              positions=[{'symbol':'ETHUSDT','side':'Buy','size':0.01,'avgPrice':2000,'unrealisedPnl':0}],
                              open_orders=[{'orderId':'order-1','symbol':'ETHUSDT','side':'Sell','qty':0.01,'price':2050,'status':'New'}])
    ok, diffs = rec.check(internal, exchange)
    assert_true("no mismatch", ok)
    assert_true("rail NOT halted", not rail.halted)


# ============== Test 7: RiskRail blocks oversize order ==============
def test_rail_oversize():
    section("TEST 7: RiskRail blocks order > MAX_POSITION_SIZE_ETH")
    log = make_log()
    rail = RiskRail("shadow", log)
    rail.update_balance(1000)
    ok, reason = rail.check_order("LONG", 0.05, 2000, 2000, 0)  # 0.05 > 0.01 max
    assert_true("blocked", not ok)
    assert_true("size in reason", "size" in reason)


# ============== Test 8: RiskRail blocks bad price ==============
def test_rail_bad_price():
    section("TEST 8: RiskRail blocks limit > mark ±0.5%")
    log = make_log()
    rail = RiskRail("shadow", log)
    rail.update_balance(1000)
    ok, reason = rail.check_order("LONG", 0.01, 2020, 2000, 0)  # 2020/2000 = +1% (>0.5%)
    assert_true("blocked", not ok)
    assert_true("price-sanity reason", "deviates" in reason or "PRICE" in reason.upper() or "limit" in reason)


# ============== Test 9: RiskRail halts on daily loss limit ==============
def test_rail_daily_loss():
    section("TEST 9: RiskRail halts when daily loss ≤ -3%")
    log = make_log()
    rail = RiskRail("shadow", log)
    rail.update_balance(1000)
    from datetime import datetime, timezone
    rail.record_fill(-31.0, datetime.now(timezone.utc))  # -3.1% on $1000
    assert_true("halted", rail.halted)
    assert_true("daily-loss reason", "daily loss" in rail.halt_reason.lower())


# ============== Test 10: RiskRail halts on max drawdown ==============
def test_rail_drawdown():
    section("TEST 10: RiskRail halts when drawdown ≤ -10%")
    log = make_log()
    rail = RiskRail("shadow", log)
    rail.update_balance(1000)
    rail.update_balance(1100)  # new peak
    rail.update_balance(989)   # -10.1% from 1100 → halt
    assert_true("halted", rail.halted)
    assert_true("drawdown reason", "drawdown" in rail.halt_reason.lower())


# ============== Test 11: OrderManager blocks place after reconcile mismatch ==============
def test_order_blocked_by_reconcile():
    section("TEST 11: OrderManager refuses to place after reconcile mismatch")
    log = make_log()
    rail = RiskRail("shadow", log)
    rec = Reconciler(log, rail)
    ex = ExchangeAdapter("shadow", log)
    # Force a mismatch: internal balance 1000 but exchange says 500
    ex._shadow_balance = 500
    internal = InternalBook(balance_usdt=1000)
    om = OrderManager(ex, rail, rec, internal, log, manual_confirm_first=0)
    from datetime import datetime, timezone
    d = Decision(ts=datetime.now(timezone.utc).isoformat(), p_up=0.6, confidence=0.1,
                 action="LONG", limit_price=2000, size=0.01, mode="shadow",
                 mid=2000, best_bid=2000, best_ask=2001)
    result = om.place(d, mark_price=2000)
    assert_true("order blocked", result is None)
    assert_true("rail halted", rail.halted)


# ============== Test 12: Shadow fill flow (end-to-end) ==============
def test_shadow_fill_flow():
    section("TEST 12: End-to-end shadow flow — place → simulate fill → reconcile")
    log = make_log()
    rail = RiskRail("shadow", log)
    rec = Reconciler(log, rail)
    ex = ExchangeAdapter("shadow", log)
    internal = InternalBook(balance_usdt=ex._shadow_balance)  # sync at start
    om = OrderManager(ex, rail, rec, internal, log, manual_confirm_first=0)
    from datetime import datetime, timezone
    d = Decision(ts=datetime.now(timezone.utc).isoformat(), p_up=0.6, confidence=0.1,
                 action="LONG", limit_price=2000, size=0.01, mode="shadow",
                 mid=2000, best_bid=2000, best_ask=2001)
    order = om.place(d, mark_price=2000)
    assert_true("order placed", order is not None and 'orderId' in order)
    assert_true("orderId tracked", order['orderId'] in internal.open_order_ids)
    # Simulate fill at maker rate
    ex._shadow_simulate_fill(order['orderId'], fill_price=2000.0, fee_bp=2.0)
    om.update_after_fill(order['orderId'], 2000.0, 0.01, "Buy")
    assert_true("orderId removed from internal after fill", order['orderId'] not in internal.open_order_ids)
    assert_true("internal position created", len(internal.positions) == 1)
    # After post-fill reconcile, balances/positions match → not halted
    assert_true("rail NOT halted after sync", not rail.halted)


def run_all():
    tests = [test_balance_mismatch, test_position_size_drift, test_wrong_side,
             test_untracked_order, test_disappeared_order, test_aligned_state,
             test_rail_oversize, test_rail_bad_price, test_rail_daily_loss,
             test_rail_drawdown, test_order_blocked_by_reconcile, test_shadow_fill_flow]
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            print(f"  ✗ {t.__name__} threw: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("\n" + "="*70 + "\nAll tests run.\n" + "="*70)


if __name__ == "__main__":
    run_all()
