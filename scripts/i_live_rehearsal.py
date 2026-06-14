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
    notional = i_live.CAP_PER_TRADE; qty = round(notional / price, 3)
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
ex, mock, om = make_exec('ok'); k, r = enter_om(ex, '4h', +1, 1600.0)
chk("정상 진입 체결 (filled=qty)", k and abs(r.filled_qty - round(150/1600, 3)) < 1e-9)
chk("  Mock 포지션 = 장부 qty", abs(mock.position - ex.led.positions[k]['qty']) < 1e-9)

# 2) 부분체결 — 실제 체결분만 기록
ex, mock, om = make_exec('partial'); k, r = enter_om(ex, '4h', +1, 1600.0)
req = round(150/1600, 3)
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

print(f"\n=== 리허설: {P} PASS / {F} FAIL (실자금/실키 0, Mock) ===")
print("실 testnet (키 발급 후): 동일 루프 1회 + 부분/타임아웃 유도. 본인 액션.")
sys.exit(1 if F else 0)
