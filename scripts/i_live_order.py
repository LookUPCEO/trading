#!/usr/bin/env python3
"""[I] 13단계 — 견고한 주문 레이어: 재시도/부분체결/타임아웃/거부 처리 + 멱등성.
client-agnostic (실 BybitClient 또는 MockClient). 실패를 성공으로 오인 = 추적 깨짐 방지.

멱등성: 주문마다 order_link_id. 타임아웃/네트워크 오류 시 재전송 전 link 로 상태조회
  → 이미 존재/체결이면 재전송 안 함 (중복 주문 방지).
부분체결: 체결 후 cumExecQty 조회 → ledger 에 '실제 체결 수량' 기록 (요청 수량 가정 X).
거부(잔고/한도): 재시도 X, 안전 중단 신호.
"""
import time, logging, math

# Bybit ETHUSDT linear perp 실 명세 (instruments-info, 2026-06-28 확인) — fetch 실패 시 fallback
SPEC_FALLBACK = dict(qty_step=0.01, min_qty=0.01, min_notional=5.0, tick=0.01)

def quantize_qty(notional, price, qty_step=0.01, min_qty=0.01, min_notional=5.0, **_):
    """명목(notional)·가격 → 거래소 qtyStep 배수로 floor (한도 초과 방지).
    반환 (qty, reason). qty<=0 = 거래 불가(min 미달). round(.,3) 버그(step 위반) 차단.
    floor 이유: round 는 명목 한도($60)를 초과할 수 있음 → 항상 내림."""
    if price <= 0 or notional <= 0:
        return 0.0, 'bad_input'
    raw = notional / price
    steps = math.floor(raw / qty_step + 1e-9)
    qty = round(steps * qty_step, 8)
    if qty < min_qty - 1e-12:
        return 0.0, f'below_min_qty({qty:.4f}<{min_qty})'
    if qty * price < min_notional - 1e-9:
        return 0.0, f'below_min_notional({qty*price:.2f}<{min_notional})'
    # step 정합 재확인 (부동소수 안전)
    if abs(qty / qty_step - round(qty / qty_step)) > 1e-6:
        return 0.0, f'step_misalign({qty})'
    return qty, 'ok'

def on_step(qty, qty_step=0.01):
    """qty 가 qtyStep 배수인가 (검증/리허설용)."""
    return qty > 0 and abs(qty / qty_step - round(qty / qty_step)) <= 1e-6

class OrderResult:
    def __init__(self, ok, filled_qty=0.0, avg_price=0.0, status='', reject=False, raw=None):
        self.ok = ok; self.filled_qty = filled_qty; self.avg_price = avg_price
        self.status = status; self.reject = reject; self.raw = raw
    def __repr__(self):
        return f"OrderResult(ok={self.ok} filled={self.filled_qty} @{self.avg_price} {self.status} reject={self.reject})"

# 거래소 거부 = 재시도 무의미 (안전 중단). 일시 오류 = 재시도.
REJECT_CODES = {110007, 110004, 110012, 110045, 30031, 110017}  # 잔고부족/한도/리스크 류 (예시)

class OrderManager:
    def __init__(self, client, log=None, max_retries=3, base_backoff=0.5, sleep=time.sleep):
        self.c = client; self.log = log or logging.getLogger()
        self.max_retries = max_retries; self.base = base_backoff; self.sleep = sleep

    def _query(self, link_id):
        """order_link_id 로 상태 조회 (멱등성). (status, cumExecQty, avgPrice) 또는 None."""
        try:
            o = self.c.get_order_by_link(link_id)
        except Exception as e:
            self.log.warning(f"상태조회 실패 {link_id}: {e}"); return None
        if not o: return None
        return (o.get('orderStatus', ''), float(o.get('cumExecQty', 0) or 0),
                float(o.get('avgPrice', 0) or 0))

    def market(self, side, qty, link_id, reduce_only=False):
        """멱등 재시도 시장주문. 부분체결/타임아웃/거부 처리."""
        for attempt in range(self.max_retries + 1):
            # 재전송 전 기존 주문 확인 (멱등)
            q = self._query(link_id)
            if q is not None:
                status, cum, avg = q
                if status in ('Filled', 'PartiallyFilled') or cum > 0:
                    self.log.info(f"[idem] {link_id} 이미 체결 {cum} ({status}) — 재전송 안 함")
                    return OrderResult(True, cum, avg, status)
                if status in ('Rejected', 'Cancelled'):
                    self.log.error(f"[reject] {link_id} {status}")
                    return OrderResult(False, 0, 0, status, reject=True)
            try:
                resp = self.c.place_market(side, qty, reduce_only=reduce_only, order_link_id=link_id)
                code = resp.get('retCode', -1)
                if code == 0:
                    # 체결 수량 확인 (부분체결 대비)
                    self.sleep(0.3)
                    q2 = self._query(link_id)
                    if q2 is not None:
                        status, cum, avg = q2
                        if cum + 1e-9 < qty:
                            self.log.warning(f"[partial] {link_id} 요청 {qty} 체결 {cum} ({status})")
                        return OrderResult(cum > 0, cum, avg, status)
                    return OrderResult(True, qty, 0.0, 'Filled?')   # 조회 안되면 보수적 성공(다음 reconcile 이 잡음)
                if code in REJECT_CODES:
                    self.log.error(f"[reject] {link_id} code={code} {resp.get('retMsg')}")
                    return OrderResult(False, 0, 0, str(code), reject=True)
                self.log.warning(f"[retry {attempt}] {link_id} code={code} {resp.get('retMsg')}")
            except Exception as e:
                self.log.warning(f"[retry {attempt}] {link_id} 예외: {e}")
            if attempt < self.max_retries:
                self.sleep(self.base * (2 ** attempt))   # 지수 백오프
        self.log.error(f"[fail] {link_id} 최대 재시도 초과 — 안전 중단 필요")
        return OrderResult(False, 0, 0, 'max_retries', reject=False)


# ───────── Mock Bybit V5 (testnet 리허설용: 성공/부분/타임아웃/거부 결정적 시뮬) ─────────
class MockBybitClient:
    """결정적 시나리오로 주문 생명주기 시뮬. 실키/실자금 불필요.
    ⚠️ 실 거래소 제약(qtyStep/minQty/minNotional)을 실제로 검증 — Mock 가짜통과 방지(I.13 교훈).
    실 BybitClient 와 동일 인터페이스(메서드 누락 = 리허설 실패로 잡힘)."""
    def __init__(self, scenario='ok', spec=None, mark=1600.0):
        self.scenario = scenario; self.orders = {}; self.position = 0.0; self.calls = 0
        self.place_calls = 0
        self.spec = spec or dict(SPEC_FALLBACK); self.mark = mark
    def get_instrument_spec(self):
        return dict(self.spec)
    def place_market(self, side, qty, reduce_only=False, order_link_id=None):
        self.place_calls += 1
        sgn = 1 if side == 'long' else -1
        sc = self.scenario
        # 실 거래소 제약 검증 (실제 Bybit 거부 재현 — round(.,3)=0.035 같은 step 위반 잡음)
        st = self.spec['qty_step']
        if qty < self.spec['min_qty'] - 1e-12 or abs(qty/st - round(qty/st)) > 1e-6:
            return {'retCode': 10001, 'retMsg': 'Qty invalid'}
        if qty * self.mark < self.spec['min_notional'] - 1e-9:
            return {'retCode': 10001, 'retMsg': 'The order value is lower than the minimum'}
        if sc == 'reject':
            return {'retCode': 110007, 'retMsg': 'insufficient balance'}
        if sc == 'transient_then_ok':
            # 처음 2회 일시 오류 후 성공
            if self.place_calls <= 2: return {'retCode': 10002, 'retMsg': 'timeout'}
        if sc == 'timeout_filled':
            # place 는 예외(타임아웃)지만 실제론 체결됨 → 다음 조회가 잡아야 (멱등)
            self.orders[order_link_id] = dict(orderStatus='Filled', cumExecQty=qty, avgPrice=1600.0)
            self.position += sgn * qty   # 실제론 체결됨 (one-way 넷팅)
            raise TimeoutError('network timeout (실제론 체결)')
        # 정상 / 부분
        fill = qty if sc != 'partial' else round(qty * 0.6, 3)
        self.orders[order_link_id] = dict(orderStatus='Filled' if fill >= qty else 'PartiallyFilled',
                                          cumExecQty=fill, avgPrice=1600.0)
        self.position += sgn * fill   # side 가 방향 인코딩 (reduce_only 도 동일; one-way 넷팅)
        return {'retCode': 0, 'retMsg': 'OK', 'result': {'orderLinkId': order_link_id}}
    def get_order_by_link(self, link_id):
        return self.orders.get(link_id)
    def get_position(self):
        return {'size': str(abs(self.position)), 'side': 'Buy' if self.position >= 0 else 'Sell'}
    def get_wallet_equity(self):
        return 188.32   # 리허설용 고정 (실 클라 인터페이스 정합)
    def cancel_all(self): return {'retCode': 0}
