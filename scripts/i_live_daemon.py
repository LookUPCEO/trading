#!/usr/bin/env python3
"""[I] 22단계 — 라이브 주문 데몬 (실제 돈). 검증 부품 결합:
  신호 = i_shadow_daemon.Shadow (bit-identical 신호경로, replay 0.00e+00 검증)
  실행 = i_live.LiveExecutor (안전장치 preflight 15/15)
  주문 = i_live_order.OrderManager (재시도/멱등/부분체결, 리허설 12/12)
⚠️ DRY_RUN=true 기본 (실주문 X). 실거래 = DRY_RUN=false + LIVE_ARM=yes 둘 다.
shadow 수집 데몬(PID 별도)과 충돌 안 함 — 라이브는 persist 안 함, 자체 WS, 자체 ledger.
"""
import os, sys, json, time, threading, logging
from datetime import datetime, timezone
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 실제 .env 키 로드 (live_bot/.env) — 값 출력 안 함
ENVP='/Users/mark/Desktop/Mark/mark19/live_bot/.env'
for line in open(ENVP):
    line=line.strip()
    if '=' in line and not line.startswith('#'):
        k,v=line.split('=',1); os.environ.setdefault(k, v.strip().strip('"').strip("'"))
sys.path.insert(0,'/Users/mark/Desktop/Mark/mark19/live_bot')
sys.path.insert(0,'/Users/mark/Desktop/Mark/mark19')  # live_bot 패키지 import

import i_shadow, i_shadow_daemon as SD, i_live, i_live_order

CAP_PER_TRADE=float(os.environ.get('CAP_PER_TRADE','60'))
DRY=os.environ.get('DRY_RUN','true').lower()!='false'
ARM=os.environ.get('LIVE_ARM','')=='yes'

class LiveShadow(SD.Shadow):
    def __init__(self, eng, log, client, om, ex, spec):
        super().__init__(eng, log)
        self.client=client; self.om=om; self.ex=ex; self.spec=spec
    def rollover(self, newday):
        # 라이브는 persist 안 함 (shadow 수집 데몬이 1Hz 저장). 버퍼만 교체.
        self.log.info(f"[live rollover] {self.buf.day} -> {newday} (persist skip)")
        self.buf=SD.DayBuf(newday); self.save_pending()
    def minute_close(self, mo):
        before=set(id(p) for p in self.pending)
        super().minute_close(mo)   # 신호 계산 + shadow pending/outcome 로깅 (bit-identical)
        # 새 신호 → 실제 진입 주문 (i_live 안전장치 + OrderManager)
        for p in self.pending:
            if id(p) in before: continue
            self.place_entry(p)
        # 4h deadline 도달 포지션 청산 (i_live ledger 기준)
        mid=self.mid_at(mo) or p.get('entry') if self.pending else None
        cur_mid=self.mid_at(mo)
        if cur_mid: self.enforce_live_exits(cur_mid)
    def place_entry(self, p):
        price=p['entry']; key=f"{p['day']}_{p['min']}_4h"
        notional=CAP_PER_TRADE
        qty,reason=i_live_order.quantize_qty(notional, price, **self.spec)   # I.29: step floor (구 round(.,3) 버그 수정)
        if qty<=0:
            self.log.error(f"[live] qty 산출 불가 {key}: {reason} (notional={notional} price={price:.2f}) — 진입 안 함"); return
        try:
            self.ex.check_can_enter(key, notional)
        except i_live.SafetyError as e:
            self.log.warning(f"[live] 진입 거부 {key}: {e}"); return
        side='long' if p['dir']>0 else 'short'
        if DRY or not ARM:
            self.log.info(f"[live DRY] 진입 {side} qty={qty} @~{price:.2f} (실주문 X)")
            res=type('R',(),{'ok':True,'filled_qty':qty,'avg_price':price,'reject':False})()
        else:
            res=self.om.market(side, qty, key)
        if not res.ok:
            self.log.error(f"[live] 진입 실패 {key}: reject={res.reject} — 기록 안 함")
            if res.reject: open(i_live.KILL_FILE,'w').write('reject halt')
            return
        self.ex.led.positions[key]=dict(horizon='4h',side=side,dir=int(p['dir']),
            qty=res.filled_qty, entry=res.avg_price or price, notional=res.filled_qty*price,
            deadline=time.time()+240*60, opened=datetime.now(timezone.utc).isoformat())
        self.ex.led.save(); self.ex.led.log({'ev':'live_enter','key':key,**self.ex.led.positions[key]})
        self.log.info(f"🟢 [live ENTER] {key} {side} qty={res.filled_qty} @{res.avg_price or price:.2f}")
    def enforce_live_exits(self, mid):
        for key in list(self.ex.led.positions):
            pos=self.ex.led.positions[key]
            if time.time()>=pos['deadline']:
                side='short' if pos['side']=='long' else 'long'
                if DRY or not ARM:
                    self.log.info(f"[live DRY] 청산 {key} @~{mid:.2f}"); res=type('R',(),{'ok':True})()
                else:
                    res=self.om.market(side,pos['qty'],key+'_x',reduce_only=True)
                pnl=pos['dir']*(mid/pos['entry']-1)*pos['notional']
                if pnl<0: self.ex.led.realized_loss+=-pnl
                self.ex.led.log({'ev':'live_exit','key':key,'exit':mid,'pnl_usd':round(pnl,3)})
                self.log.info(f"🔴 [live EXIT] {key} @{mid:.2f} pnl=${pnl:+.2f}")
                del self.ex.led.positions[key]; self.ex.led.save()
    def reconcile(self):
        if DRY or not ARM: return True
        try:
            pos=self.client.get_position(); exch=abs(float(pos['size'])) if pos and pos.get('size') else 0.0
        except Exception as e:
            self.log.error(f"reconcile 실패 {e}"); return False
        book=sum(p['qty'] for p in self.ex.led.positions.values())
        if abs(exch-book)>0.01:
            self.log.error(f"⚠️ 정합 불일치 거래소 {exch} vs 장부 {book} → KILL")
            open(i_live.KILL_FILE,'w').write(f"reconcile {datetime.now(timezone.utc)}"); return False
        return True

def main():
    LD=f'{i_shadow.SHD}/live'; os.makedirs(LD,exist_ok=True)
    i_live.LIVE_DIR=LD; i_live.KILL_FILE=f'{LD}/KILL'; i_live.LEDGER=f'{LD}/ledger.jsonl'; i_live.STATE=f'{LD}/positions.json'
    i_live.CAP_PER_TRADE=CAP_PER_TRADE; i_live.MAX_CONCURRENT=1; i_live.DRY_RUN=DRY; i_live.LIVE_ARM=ARM
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(f'{LD}/live_daemon.log'), logging.StreamHandler(sys.stdout)])
    log=logging.getLogger()
    from execution import BybitClient
    client=BybitClient()
    eq=client.get_wallet_equity()
    try:
        spec=client.get_instrument_spec()
        log.info(f"   instrument spec: {spec}")
    except Exception as e:
        spec=dict(i_live_order.SPEC_FALLBACK); log.warning(f"   spec fetch 실패 {e} → fallback {spec}")
    log.info(f"=== [I] LIVE daemon — DRY={DRY} ARM={ARM} equity=${eq:.2f} CAP/trade=${CAP_PER_TRADE} ===")
    log.info(f"   ⚠️ {'DRY (실주문 X)' if (DRY or not ARM) else '★ 실거래 ARMED ★'}")
    om=i_live_order.OrderManager(client, log=log)
    ex=i_live.LiveExecutor(client=client, log=log)
    eng=i_shadow.Engine()
    sh=LiveShadow(eng, log, client, om, ex, spec)
    def minute_loop():
        last=-1
        while True:
            time.sleep(1.0); now=datetime.now(timezone.utc); mo=now.hour*60+now.minute
            if mo!=last and now.second>=2:
                if last!=-1 and mo>0:
                    try: sh.minute_close(mo-1);
                    except Exception as e: log.error(f"minute_close 실패: {e}")
                    if not DRY and ARM: sh.reconcile()
                last=mo
    threading.Thread(target=minute_loop, daemon=True).start()
    import websocket
    while True:
        try:
            ws=websocket.WebSocketApp(SD.WS_URL,
                on_open=lambda w:(w.send(json.dumps({'op':'subscribe','args':[f'orderbook.{SD.LV}.{SD.SYM}',f'publicTrade.{SD.SYM}']})), log.info('live WS connected'))[1],
                on_message=sh.on_message, on_error=lambda w,e:log.error(f'WS err {e}'), on_close=lambda w,c,m:log.info('WS closed'))
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e: log.error(f'WS loop {e}')
        time.sleep(5)

if __name__=='__main__':
    main()
