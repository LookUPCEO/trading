#!/usr/bin/env python3
"""[I] 12단계 — 소액 라이브 실행기 (4h thr0.70 결합). ⚠️ DRY_RUN 기본 (실주문 X).

안전 최우선 (돈 잃는 버그가 edge 보다 위험):
  - DRY_RUN=True 기본: 의도 주문 로그만, 실주문 0. 실거래는 LIVE_ARM=yes + DRY_RUN=false 둘 다.
  - kill switch: KILL 파일 존재 시 즉시 전체청산+중단.
  - 자본 한도: 1회 주문 명목 ≤ CAP_PER_TRADE, 총 노출 ≤ CAP_TOTAL. 초과 시 거부.
  - 과다주문 방지: 같은 (day,min,horizon) 신호 1회만 (dedupe), 동시 포지션 수 상한.
  - 4h 청산 보장: 각 포지션 exit_deadline 기록, 누락 감지 시 강제 시장청산.
  - 포지션 정합: 시스템 장부 vs 거래소 get_position 주기 대조, 불일치 시 알림+중단.
  - API: 거래 권한만 (출금 X — 키 발급 시 보장, 코드는 출금 호출 자체 없음).

신호 = i_shadow.Engine (동결 artifact, replay bit-identical 검증). 실행만 추가.
이 stage = 인프라/안전 점검 (preflight). 실주문은 별도 확인 후."""
import os, sys, json, time, threading, logging
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LIVE_DIR = '/Users/mark/Desktop/Mark/mark19/live_bot_state/i_live'
os.makedirs(LIVE_DIR, exist_ok=True)
KILL_FILE = f'{LIVE_DIR}/KILL'
LEDGER = f'{LIVE_DIR}/ledger.jsonl'
STATE = f'{LIVE_DIR}/positions.json'

# ---- 안전 파라미터 (소액) ----
CAP_TOTAL = float(os.environ.get('CAP_TOTAL', '180'))      # 총 자본 USD
CAP_PER_TRADE = float(os.environ.get('CAP_PER_TRADE', '150')) # 1회 명목 USD (단일 4h, 총 자본 내)
MAX_CONCURRENT = int(os.environ.get('MAX_CONCURRENT', '1'))   # one-way 퍼프 넷팅 → 단일 포지션(4h). 동시 다방향 불가
LEVERAGE = float(os.environ.get('LEVERAGE', '1'))             # 레버리지 1 (감쇠 시 손실 확대 방지)
LOSS_HALT = float(os.environ.get('LOSS_HALT', '60'))         # 누적손실 이 USD 도달 시 중단 (총의 1/3)
DRY_RUN = os.environ.get('DRY_RUN', 'true').lower() != 'false'
LIVE_ARM = os.environ.get('LIVE_ARM', '') == 'yes'

HORIZONS = {'30m': 30, '1h': 60, '4h': 240}
THR = 0.70

def now(): return datetime.now(timezone.utc)

class SafetyError(Exception): pass

class Ledger:
    """시스템 장부 (포지션 = 진입가/수량/방향/청산기한)."""
    def __init__(self):
        self.positions = {}   # key -> dict
        self.realized_loss = 0.0
        if os.path.exists(STATE):
            d = json.load(open(STATE)); self.positions = d.get('positions', {})
            self.realized_loss = d.get('realized_loss', 0.0)
    def save(self):
        json.dump({'positions': self.positions, 'realized_loss': self.realized_loss}, open(STATE, 'w'))
    def open_notional(self):
        return sum(p['notional'] for p in self.positions.values())
    def log(self, ev):
        ev['ts'] = now().isoformat()
        with open(LEDGER, 'a') as f: f.write(json.dumps(ev) + '\n')

class LiveExecutor:
    def __init__(self, engine=None, client=None, log=None):
        self.eng = engine; self.client = client; self.log = log or logging.getLogger()
        self.led = Ledger()

    # ---- 안전 체크 (주문 전 전부 통과해야) ----
    def kill_active(self):
        return os.path.exists(KILL_FILE)

    def check_can_enter(self, key, notional):
        if self.kill_active(): raise SafetyError('KILL switch active')
        if self.led.realized_loss >= LOSS_HALT: raise SafetyError(f'손실한도 도달 {self.led.realized_loss:.1f}>={LOSS_HALT}')
        if key in self.led.positions: raise SafetyError(f'중복 신호 {key} (dedupe)')
        if len(self.led.positions) >= MAX_CONCURRENT: raise SafetyError(f'동시포지션 상한 {MAX_CONCURRENT}')
        if notional > CAP_PER_TRADE + 1e-6: raise SafetyError(f'1회 한도 초과 {notional:.1f}>{CAP_PER_TRADE}')
        if self.led.open_notional() + notional > CAP_TOTAL + 1e-6: raise SafetyError(f'총노출 초과')
        return True

    def enter(self, horizon, direction, price, mid):
        key = f"{now().strftime('%Y%m%d')}_{horizon}_{now().strftime('%H%M')}"
        notional = CAP_PER_TRADE
        qty = round(notional / price, 3)
        try:
            self.check_can_enter(key, notional)
        except SafetyError as e:
            self.log.warning(f"진입 거부 [{key}]: {e}"); return None
        deadline = time.time() + HORIZONS[horizon] * 60
        side = 'long' if direction > 0 else 'short'
        if DRY_RUN or not LIVE_ARM:
            self.log.info(f"[DRY] 진입 {side} {horizon} qty={qty} @~{price:.2f} (실주문 X)")
            resp = {'dry': True}
        else:
            resp = self.client.place_market(side, qty, order_link_id=key)   # taker
            self.log.info(f"[LIVE] 진입 {side} {horizon} qty={qty}: {resp.get('retCode')}")
        self.led.positions[key] = dict(horizon=horizon, side=side, dir=int(direction),
                                       qty=qty, entry=price, notional=notional,
                                       deadline=deadline, opened=now().isoformat())
        self.led.save(); self.led.log({'ev': 'enter', 'key': key, **self.led.positions[key], 'resp': resp})
        return key

    def exit_position(self, key, reason, price):
        p = self.led.positions.get(key)
        if not p: return
        side = 'short' if p['side'] == 'long' else 'long'   # 반대로 청산
        if DRY_RUN or not LIVE_ARM:
            self.log.info(f"[DRY] 청산 {key} ({reason}) @~{price:.2f} (실주문 X)")
            resp = {'dry': True}
        else:
            resp = self.client.place_market(side, p['qty'], reduce_only=True, order_link_id=key + '_x')
            self.log.info(f"[LIVE] 청산 {key} ({reason}): {resp.get('retCode')}")
        pnl_usd = p['dir'] * (price / p['entry'] - 1) * p['notional']
        if pnl_usd < 0: self.led.realized_loss += -pnl_usd
        self.led.log({'ev': 'exit', 'key': key, 'reason': reason, 'exit': price, 'pnl_usd': pnl_usd, 'resp': resp})
        del self.led.positions[key]; self.led.save()

    # ---- 4h 청산 보장 + 정합성 (주기 호출) ----
    def enforce_exits(self, mid):
        for key in list(self.led.positions):
            p = self.led.positions[key]
            if time.time() >= p['deadline']:
                self.exit_position(key, 'deadline', mid)

    def reconcile(self):
        """시스템 장부 명목 vs 거래소 실제 포지션 대조."""
        if DRY_RUN or self.client is None: return True
        try:
            pos = self.client.get_position()
            exch_size = abs(float(pos['size'])) if pos and pos.get('size') else 0.0
        except Exception as e:
            self.log.error(f"reconcile 조회 실패: {e}"); return False
        book_qty = sum(p['qty'] for p in self.led.positions.values())
        if abs(exch_size - book_qty) > 0.01:
            self.log.error(f"⚠️ 정합 불일치: 거래소 {exch_size} vs 장부 {book_qty} → KILL")
            open(KILL_FILE, 'w').write(f"reconcile mismatch {now()}")
            return False
        return True

    def kill_all(self, mid):
        self.log.warning("KILL: 전체청산+중단")
        for key in list(self.led.positions):
            self.exit_position(key, 'kill', mid)
        if not DRY_RUN and LIVE_ARM and self.client:
            try: self.client.cancel_all()
            except Exception: pass

if __name__ == '__main__':
    # preflight 은 i_live_preflight.py 로 분리. 여기 직접 실행은 데몬(별도 확인 후).
    print("i_live: DRY_RUN=%s LIVE_ARM=%s CAP_TOTAL=%s CAP_PER_TRADE=%s LEV=%s" % (
        DRY_RUN, LIVE_ARM, CAP_TOTAL, CAP_PER_TRADE, LEVERAGE))
    print("실거래는 preflight 통과 + 별도 확인 후. 이 stage 는 점검만.")
