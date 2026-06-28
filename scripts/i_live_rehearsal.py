#!/usr/bin/env python3
"""[I] 13단계 리허설 — 전체 신호→안전장치→주문→부분/타임아웃/거부→4h청산 루프.
Mock Bybit (실자금/실키 0). 모든 경로 결정적 검증. 실 testnet 은 키 발급 후 사용자."""
import os, sys, time, json, logging, shutil
os.environ['DRY_RUN'] = 'false'; os.environ['LIVE_ARM'] = 'yes'   # 주문경로 활성 (단 client=Mock)
D = '/Users/mark/Desktop/Mark/mark19/live_bot_state/i_rehearsal'
shutil.rmtree(D, ignore_errors=True); os.makedirs(D, exist_ok=True)
import i_live, i_live_order
i_live.LIVE_DIR = D; i_live.KILL_FILE = f'{D}/KILL'; i_live.LEDGER = f'{D}/ledger.jsonl'; i_live.STATE = f'{D}/positions.json'
i_live.CAP_TOTAL = 180; i_live.CAP_PER_TRADE = 150; i_live.MAX_CONCURRENT = 1; i_live.LOSS_HALT = 60  # one-way 넷팅: 단일 4h
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger()
P = 0; F = 0
def chk(n, c):
    global P, F; print(f"  [{'PASS' if c else 'FAIL'}] {n}"); P += c; F += (not c)

# OrderManager 를 쓰는 실행기 (i_live.LiveExecutor.enter/exit 를 OrderManager 경유로)
def make_exec(scenario, no_sleep=True):
    led_reset()
    mock = i_live_order.MockBybitClient(scenario)
    om = i_live_order.OrderManager(mock, log=log, base_backoff=0.01,
                                   sleep=(lambda s: None) if no_sleep else time.sleep)
    ex = i_live.LiveExecutor(client=mock, log=log)
    ex.om = om
    return ex, mock, om
def led_reset():
    for f in ['ledger.jsonl', 'positions.json', 'KILL']:
        p = f'{D}/{f}';
        if os.path.exists(p): os.remove(p)

# enter/exit 를 OrderManager 경유로 패치 (멱등/재시도/부분체결 반영)
def enter_om(ex, horizon, direction, price):
    key = f"reh_{horizon}_{direction}"
    notional = i_live.CAP_PER_TRADE
    qty, _r = i_live_order.quantize_qty(notional, price)   # I.29: 실 step 으로 quantize (구 round(.,3) 버그 수정)
    try: ex.check_can_enter(key, notional)
    except i_live.SafetyError as e:
        log.warning(f"진입 거부 [{key}]: {e}"); return None, None
    side = 'long' if direction > 0 else 'short'
    res = ex.om.market(side, qty, key)
    if not res.ok:
        log.error(f"진입 주문 실패 {key}: {res} — 포지션 기록 안 함 (실패 오인 방지)")
        if res.reject: open(ex.led.__class__.__module__ and i_live.KILL_FILE, 'w').write('reject halt')
        return None, res
    # 실제 체결 수량으로 기록 (부분체결 반영)
    ex.led.positions[key] = dict(horizon=horizon, side=side, dir=int(direction),
                                 qty=res.filled_qty, entry=price, notional=res.filled_qty * price,
                                 deadline=time.time() + i_live.HORIZONS[horizon] * 60)
    ex.led.save(); return key, res
def exit_om(ex, key, reason):
    p = ex.led.positions.get(key);
    if not p: return None
    side = 'short' if p['side'] == 'long' else 'long'
    res = ex.om.market(side, p['qty'], key + '_x', reduce_only=True)
    if res.ok: del ex.led.positions[key]; ex.led.save()
    return res

print("=== 작업1: 재시도/부분체결/타임아웃/거부 (Mock 시나리오) ===")
# 1) 정상
_q150 = i_live_order.quantize_qty(150, 1600.0)[0]
ex, mock, om = make_exec('ok'); k, r = enter_om(ex, '4h', +1, 1600.0)
chk("정상 진입 체결 (filled=quantized qty)", k and abs(r.filled_qty - _q150) < 1e-9)
chk("  Mock 포지션 = 장부 qty", abs(mock.position - ex.led.positions[k]['qty']) < 1e-9)

# 2) 부분체결 — 실제 체결분만 기록
ex, mock, om = make_exec('partial'); k, r = enter_om(ex, '4h', +1, 1600.0)
req = _q150
chk("부분체결: 요청>체결, 체결분(0.6×)만 기록", k and r.filled_qty < req and abs(ex.led.positions[k]['qty'] - r.filled_qty) < 1e-9)

# 3) 일시 오류 후 성공 (재시도 백오프)
ex, mock, om = make_exec('transient_then_ok'); k, r = enter_om(ex, '4h', +1, 1600.0)
chk("일시오류 2회 후 재시도 성공", k is not None and r.ok and mock.place_calls >= 3)

# 4) 거부 (잔고부족) — 재시도 안 함, 포지션 기록 안 함
ex, mock, om = make_exec('reject'); k, r = enter_om(ex, '4h', +1, 1600.0)
chk("거부 시 진입 기록 안 함 (실패 오인 방지)", k is None and r is not None and r.reject)
chk("  거부 시 재시도 안 함 (place 1회)", mock.place_calls == 1)

# 5) 타임아웃이지만 실제 체결 → 멱등 조회로 중복 방지
ex, mock, om = make_exec('timeout_filled')
k, r = enter_om(ex, '4h', +1, 1600.0)
chk("타임아웃-실체결: 멱등 조회로 체결 인식 (중복 주문 X)", k is not None and r.ok and mock.place_calls == 1)

print("\n=== 작업2: 신호→주문 루프 (진입→4h deadline→청산) ===")
ex, mock, om = make_exec('ok')
k, r = enter_om(ex, '30m', +1, 1600.0)
ex.led.positions[k]['deadline'] = time.time() - 1   # 만료 강제
# enforce_exits 를 OrderManager 경유 청산으로
for kk in list(ex.led.positions):
    if time.time() >= ex.led.positions[kk]['deadline']: exit_om(ex, kk, 'deadline')
chk("4h deadline 도달 → reduce_only 청산", len(ex.led.positions) == 0)
chk("  청산 후 Mock 포지션 ≈ 0", abs(mock.position) < 1e-6)

print("\n=== 작업3: 안전장치 testnet(mock) 실작동 + 정합 ===")
ex, mock, om = make_exec('ok')
k1, _ = enter_om(ex, '4h', +1, 1600.0)
k2, _ = enter_om(ex, '4h', +1, 1600.0)   # 단일 포지션(MAX_CONCURRENT=1) → 2번째 거부 (one-way 넷팅)
chk("단일 포지션 상한 (2번째 거부, one-way)", k1 is not None and k2 is None and len(ex.led.positions) == 1)
chk("  reconcile 정합 (mock net = 장부 qty)", ex.reconcile())
mock.position += 5.0   # 불일치 유도
ok = ex.reconcile()
chk("정합 불일치 감지 → KILL 생성", (not ok) and os.path.exists(i_live.KILL_FILE))

print("\n=== 작업4: 실 거래소 제약 검증 (I.29 — Mock 가짜통과 방지) ===")
# 4a) 인터페이스 정합: 실 BybitClient 가 주문경로의 모든 메서드를 진짜 갖는가 (get_order_by_link 누락 재발 방지)
sys.path.insert(0, '/Users/mark/Desktop/Mark/mark19')
need = ['place_market', 'get_order_by_link', 'get_position', 'get_wallet_equity',
        'cancel_all', 'get_instrument_spec']
try:
    from live_bot.execution import BybitClient as _RealClient
    missing = [m for m in need if not callable(getattr(_RealClient, m, None))]
    chk(f"실 BybitClient 인터페이스 완비 (누락={missing})", not missing)
    # Mock 이 실 클라 인터페이스와 동일 (메서드 누락 = 가짜통과 → 잡음)
    mockmiss = [m for m in need if not callable(getattr(i_live_order.MockBybitClient, m, None))]
    chk(f"Mock = 실 클라 인터페이스 동일 (누락={mockmiss})", not mockmiss)
except Exception as e:
    chk(f"BybitClient import (실패={e})", False)

# 4b) quantize_qty 정확성 (구 round(.,3) 버그 차단)
q, rs = i_live_order.quantize_qty(60, 1705.30)
chk(f"quantize 60/1705.30 = {q} (step 0.01 배수, {rs})", i_live_order.on_step(q) and q == 0.03)
chk("  명목 한도 미초과 (floor: 0.03×1705=$51 ≤ $60)", q * 1705.30 <= 60 + 1e-9)
chk("  구 버그 round(.,3)=0.035 는 step 위반 (on_step False)", not i_live_order.on_step(round(60/1705.30, 3)))

# 4c) Mock 이 구 버그 qty(0.035) 를 실제로 거부하는가 (이게 I.13 에서 안 잡혔음)
mk = i_live_order.MockBybitClient('ok', mark=1705.30)
bad = mk.place_market('long', round(60/1705.30, 3))   # 0.035 = step 위반
chk(f"Mock 이 step 위반 qty 거부 (retCode={bad.get('retCode')})", bad.get('retCode') == 10001)
good = mk.place_market('long', q)   # 0.03 = valid
chk(f"Mock 이 valid quantized qty 수락 (retCode={good.get('retCode')})", good.get('retCode') == 0)
chk("  min_notional 위반(미세 qty) 거부", i_live_order.quantize_qty(3, 1705.30)[0] == 0.0)

# 4d) 과거 실패 3건(점검 발견) 재현 — 이제 valid qty 산출되나
print("  과거 실패 3건 재현 (수정 후 valid?):")
fails = [('6/19 숏', 1705.30), ('6/25 롱 fup0.733(핵심 edge)', 1551.27), ('6/28 숏', 1583.91)]
allok = True
for nm, px in fails:
    qq, rr = i_live_order.quantize_qty(60, px)
    valid = i_live_order.on_step(qq) and qq > 0 and 5 <= qq * px <= 60 + 1e-9
    allok &= valid
    print(f"    {nm}: px={px} → qty={qq} 명목=${qq*px:.0f} {'✅valid' if valid else '❌'}")
chk("과거 실패 3건 전부 이제 valid qty (6/25 핵심 edge 포함)", allok)

print(f"\n=== 리허설: {P} PASS / {F} FAIL (실자금/실키 0, Mock — 실 제약 검증 강화) ===")
print("I.29: Mock 이 실 거래소 제약(step/min/메서드) 검증 → 가짜통과 방지. 다음 신호 시 실주문 모니터.")
sys.exit(1 if F else 0)
