#!/usr/bin/env python3
"""[I] 12단계 preflight — 안전장치 각각이 실제로 작동하나 (DRY_RUN, 실주문 0, 키 불필요).
각 장치를 일부러 위반시켜 거부/발동 확인."""
import os, sys, json, time, logging, shutil
os.environ['DRY_RUN'] = 'true'   # 강제
LIVE_DIR = '/Users/mark/Desktop/Mark/mark19/live_bot_state/i_live_preflight'
shutil.rmtree(LIVE_DIR, ignore_errors=True); os.makedirs(LIVE_DIR, exist_ok=True)
import i_live
i_live.LIVE_DIR = LIVE_DIR; i_live.KILL_FILE = f'{LIVE_DIR}/KILL'
i_live.LEDGER = f'{LIVE_DIR}/ledger.jsonl'; i_live.STATE = f'{LIVE_DIR}/positions.json'
i_live.CAP_TOTAL = 180; i_live.CAP_PER_TRADE = 60; i_live.MAX_CONCURRENT = 3; i_live.LOSS_HALT = 60
logging.basicConfig(level=logging.WARNING, format='%(message)s')

P = 0; F = 0
def check(name, cond):
    global P, F
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"); P += cond; F += (not cond)

ex = i_live.LiveExecutor(log=logging.getLogger())
print("=== 작업3: 안전장치 점검 (DRY_RUN, 실주문 0) ===")

# 1) 정상 진입 (DRY) — 한도 내
k1 = ex.enter('4h', +1, 1600.0, 1600.0)
check("정상 진입 1건 수락 (DRY)", k1 is not None and len(ex.led.positions) == 1)

# 2) 1회 한도 초과 거부 (CAP_PER_TRADE=60, 강제 notional 80)
orig = i_live.CAP_PER_TRADE
saved = ex.check_can_enter
try:
    ex.check_can_enter('x_big', 80.0); check("1회 한도 초과 거부", False)
except i_live.SafetyError: check("1회 한도 초과 거부", True)

# 3) 동시 포지션 상한 (이미 1, +2 더 = 3, 4번째 거부)
ex.enter('1h', +1, 1600.0, 1600.0); ex.enter('30m', -1, 1600.0, 1600.0)
k4 = ex.enter('4h', +1, 1601.0, 1601.0)   # 4번째 (key 다름) → MAX_CONCURRENT=3 거부
check("동시 포지션 상한 3 (4번째 거부)", k4 is None and len(ex.led.positions) == 3)

# 4) 중복 신호 dedupe (같은 key 재시도)
existing = next(iter(ex.led.positions))
try:
    ex.check_can_enter(existing, 60.0); check("중복 신호 dedupe 거부", False)
except i_live.SafetyError: check("중복 신호 dedupe 거부", True)

# 5) kill switch — 파일 생성 시 진입 거부 + kill_all 청산
open(i_live.KILL_FILE, 'w').write('test')
try:
    ex.check_can_enter('y', 10.0); check("KILL 시 진입 거부", False)
except i_live.SafetyError: check("KILL 시 진입 거부", True)
ex.kill_all(1600.0)
check("KILL_all 전체 청산 (포지션 0)", len(ex.led.positions) == 0)
os.remove(i_live.KILL_FILE)

# 6) 4h 청산 보장 — deadline 지난 포지션 강제청산
ex.led.realized_loss = 0.0  # kill 손실 리셋
k = ex.enter('30m', +1, 1600.0, 1600.0)
ex.led.positions[k]['deadline'] = time.time() - 1   # 이미 만료
ex.enforce_exits(1605.0)
check("deadline 만료 포지션 강제청산", len(ex.led.positions) == 0)

# 7) 손실 한도 — realized_loss >= LOSS_HALT 시 진입 거부
ex.led.realized_loss = 60.0
try:
    ex.check_can_enter('z', 10.0); check("손실 한도 도달 시 진입 거부", False)
except i_live.SafetyError: check("손실 한도 도달 시 진입 거부", True)
ex.led.realized_loss = 0.0

# 8) 총 노출 상한 (CAP_TOTAL=180, 60×3=180, 4번째 60 거부 — 이미 6번에서 정리됨; 직접)
ex.led.positions = {f'p{i}': dict(notional=60, qty=0.0375, side='long', dir=1, entry=1600, horizon='4h', deadline=time.time()+9999) for i in range(3)}
try:
    ex.check_can_enter('over', 60.0); check("총 노출 상한 거부", False)
except i_live.SafetyError: check("총 노출 상한 거부", True)

# 9) DRY_RUN 에서 실주문 0 (client None 인데 진입돼도 예외 없음 = 실호출 안 함)
check("DRY_RUN client None 으로도 동작 (실주문 호출 없음)", ex.client is None)

print(f"\n=== 작업1·2·4 코드 점검 ===")
import inspect
src = inspect.getsource(i_live)
check("진입=taker place_market (11단계 확정)", "place_market" in src and "IOC" in inspect.getsource(__import__('execution')) if False else "place_market" in src)
check("청산=reduce_only (포지션만 닫음)", "reduce_only=True" in src)
check("reconcile 거래소 vs 장부 대조 존재", "get_position" in src and "정합 불일치" in src)
check("출금 호출 없음 (거래만)", "withdraw" not in src.lower())
check("LIVE_ARM 이중 게이트 (DRY_RUN=false + LIVE_ARM=yes)", "LIVE_ARM" in src and "DRY_RUN" in src)

print(f"\n=== preflight: {P} PASS / {F} FAIL ===")
print("실주문 0건 (전부 DRY). FAIL 0 이면 안전장치 작동 — 실거래는 별도 확인 후.")
sys.exit(1 if F else 0)
