#!/usr/bin/env python3
"""
[I] shadow 데몬 — Bybit WS 라이브, 매분 fup240 기록 + thr70 신호/결과 로그 (돈 X).
i_shadow.Engine (동결 artifact) + labels_from_seconds (replay 일치 0.0 검증된 경로) 사용.

- 당일 1Hz OB(50레벨)/trades 를 기존 스키마로 persist → 수집 재개 겸용.
- 신호 조건 = 백테스트와 동일: min_of_day ∈ [480, 1199], fup>=0.70 or <=0.30, votes>=70.
- 결과: 신호 분 +240분의 mid (자체 로그에서) — outcome CSV.
- 시작일은 warm-up (day 처음부터 못 봄) → 다음 UTC 자정부터 유효. big_thr 는 전일 q95 자동갱신.
"""
import os, sys, json, time, threading, logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import websocket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i_shadow import Engine, labels_from_seconds, SHD
import i_labeling as IL

WS_URL = "wss://stream.bybit.com/v5/public/linear"
SYM = "ETHUSDT"
LV = 50
OB_OUT = '/Users/mark/mark19_data/ETHUSDT'
TR_OUT = '/Users/mark/mark19_data/trades_perp/ETHUSDT'
LOGD = f'{SHD}/logs'
os.makedirs(LOGD, exist_ok=True)

class DayBuf:
    """UTC 하루치 초 그리드 버퍼."""
    def __init__(self, day):
        self.day = day
        n = 86400
        self.have = np.zeros(n, bool)
        self.bidp = np.full((n, LV), np.nan, np.float32); self.bids = np.zeros((n, LV), np.float32)
        self.askp = np.full((n, LV), np.nan, np.float32); self.asks = np.zeros((n, LV), np.float32)
        self.buyv = np.zeros(n); self.sellv = np.zeros(n); self.vols = np.zeros(n); self.bigv = np.zeros(n)
        self.trades = []   # (epoch_s, side, size) persist 용
        self.last_sec = -1

class Shadow:
    def __init__(self, eng, log):
        self.eng = eng; self.log = log
        self.book = {'b': {}, 'a': {}}
        self.buf = DayBuf(datetime.now(timezone.utc).strftime('%Y-%m-%d'))
        self.warm_day = self.buf.day   # 이 날은 불완전 (mid-day 시작)
        st = {}
        stf = f'{SHD}/state.json'
        if os.path.exists(stf): st = json.load(open(stf))
        self.big_thr = st.get('big_thr', eng.meta['big_thr'])
        self.pending = []   # (day, minute, dir, fup, mid)
        self.lock = threading.Lock()
        self.connected = False; self.last_msg = time.time()

    # ---- WS ----
    def on_message(self, ws, raw):
        self.last_msg = time.time()
        m = json.loads(raw)
        topic = m.get('topic', '')
        now = time.time()
        sec_utc = datetime.now(timezone.utc)
        day = sec_utc.strftime('%Y-%m-%d')
        s = sec_utc.hour * 3600 + sec_utc.minute * 60 + sec_utc.second
        with self.lock:
            if day != self.buf.day:
                self.rollover(day)
            if topic.startswith('orderbook'):
                d = m['data']
                if m['type'] == 'snapshot':
                    self.book = {'b': {float(p): float(q) for p, q in d['b']},
                                 'a': {float(p): float(q) for p, q in d['a']}}
                else:
                    for p, q in d['b']:
                        p = float(p); q = float(q)
                        if q == 0: self.book['b'].pop(p, None)
                        else: self.book['b'][p] = q
                    for p, q in d['a']:
                        p = float(p); q = float(q)
                        if q == 0: self.book['a'].pop(p, None)
                        else: self.book['a'][p] = q
                self.snap(s)
            elif topic.startswith('publicTrade'):
                for t in m['data']:
                    ts_ms = int(t['T']); sz = float(t['v']); side = t['S']
                    es = ts_ms / 1000.0
                    tsec = int(np.round(es - (datetime(sec_utc.year, sec_utc.month, sec_utc.day,
                                                       tzinfo=timezone.utc).timestamp())))
                    if 0 <= tsec < 86400:
                        b = self.buf
                        if side == 'Buy': b.buyv[tsec] += sz
                        else: b.sellv[tsec] += sz
                        b.vols[tsec] += sz
                        if sz >= self.big_thr:
                            b.bigv[tsec] += sz if side == 'Buy' else -sz
                        b.trades.append((es, side, sz))

    def snap(self, s):
        if not self.book['b'] or not self.book['a']: return
        b = self.buf
        bids = sorted(self.book['b'].items(), key=lambda x: -x[0])[:LV]
        asks = sorted(self.book['a'].items(), key=lambda x: x[0])[:LV]
        if len(bids) < 5 or len(asks) < 5: return
        for i, (p, q) in enumerate(bids): b.bidp[s, i] = p; b.bids[s, i] = q
        for i, (p, q) in enumerate(asks): b.askp[s, i] = p; b.asks[s, i] = q
        b.have[s] = True; b.last_sec = max(b.last_sec, s)

    # ---- 분 처리 ----
    def minute_close(self, mo):
        """분 mo 의 마지막 초까지 들어온 후 호출 (다음 분 시작 시)."""
        b = self.buf
        T = mo * 60 + 59
        if T > b.last_sec or not b.have[:T + 1].any(): return
        hv = np.where(b.have[:T + 1])[0]
        if len(hv) < 300: return
        # ffill 초 그리드 (배치 ffill_idx 동일 의미)
        idx = np.searchsorted(hv, np.arange(T + 1), side='right') - 1
        idx = hv[np.maximum(idx, 0)]
        bidp = b.bidp[idx]; bids = b.bids[idx]; askp = b.askp[idx]; asks = b.asks[idx]
        mid = (bidp[:, 0] + askp[:, 0]) / 2.0
        def obid(d):
            x = bids[:, :d].sum(1); y = asks[:, :d].sum(1); return (x - y) / (x + y + 1e-9)
        o1, o5, o20, o50 = obid(1), obid(5), obid(20), obid(50)
        mv = mid[:, None]
        wb = 1.0 / (1.0 + np.abs(bidp - mv) / 0.01); wa = 1.0 / (1.0 + np.abs(askp - mv) / 0.01)
        owt = ((bids * wb).sum(1) - (asks * wa).sum(1)) / ((bids * wb).sum(1) + (asks * wa).sum(1) + 1e-9)
        spr = (askp[:, 0] - bidp[:, 0]) / mid * 1e4
        cb = np.cumsum(b.buyv[:T + 1]); cs_ = np.cumsum(b.sellv[:T + 1])
        cg = np.cumsum(b.bigv[:T + 1]); cv = np.cumsum(b.vols[:T + 1])
        df = labels_from_seconds(mid, o1, o5, o20, o50, owt, spr, cb, cs_, cg, cv, T + 1, self.big_thr)
        row = df.iloc[-1]
        if int(row['min_of_day']) != mo: return
        z = self.eng.normalize(row)
        nan47 = np.isnan(z).sum()
        fup, nv = (np.nan, 0)
        if nan47 == 0 and mo >= 480 and self.buf.day != self.warm_day:
            fup, nv = self.eng.fup240(z)
        sig = 0
        if not np.isnan(fup) and 480 <= mo <= 1199:
            if fup >= 0.70: sig = 1
            elif fup <= 0.30: sig = -1
        line = dict(ts=datetime.now(timezone.utc).isoformat(), day=b.day, min_of_day=mo,
                    mid=float(row['mid']), fup240=None if np.isnan(fup) else round(float(fup), 4),
                    votes=nv, signal=sig, warm=self.buf.day == self.warm_day)
        with open(f'{LOGD}/minutes_{b.day}.jsonl', 'a') as f:
            f.write(json.dumps(line) + '\n')
        if sig != 0:
            self.pending.append(dict(day=b.day, min=mo, dir=sig, fup=float(fup), entry=float(row['mid'])))
            self.log.info(f"🔔 SIGNAL {b.day} min{mo} dir={sig} fup={fup:.3f} entry={row['mid']:.2f}")
            self.save_pending()
        # outcome: +240분 지난 pending
        done = []
        for p in self.pending:
            if p['day'] == b.day and mo >= p['min'] + 240:
                outc = (float(row['mid']) if mo == p['min'] + 240 else self.mid_at(p['min'] + 240))
                if outc is None: continue
                gross = (outc / p['entry'] - 1) * 1e4 * p['dir']
                rec = dict(**p, exit_mid=outc, gross_bp=round(gross, 2), net_TT=round(gross - 11, 2),
                           closed=datetime.now(timezone.utc).isoformat())
                with open(f'{SHD}/outcomes.jsonl', 'a') as f:
                    f.write(json.dumps(rec) + '\n')
                self.log.info(f"✅ OUTCOME {rec}")
                done.append(p)
        for p in done: self.pending.remove(p)
        if done: self.save_pending()

    def mid_at(self, mo):
        b = self.buf
        T = mo * 60 + 59
        hv = np.where(b.have[:T + 1])[0]
        if len(hv) == 0: return None
        s = hv[-1]
        return float((b.bidp[s, 0] + b.askp[s, 0]) / 2.0)

    def save_pending(self):
        json.dump(dict(big_thr=self.big_thr, pending=self.pending),
                  open(f'{SHD}/state.json', 'w'))

    def rollover(self, newday):
        b = self.buf
        self.log.info(f"[rollover] {b.day} -> {newday}, persist 1Hz...")
        try:
            hv = np.where(b.have)[0]
            if len(hv) > 1000:
                cols = {'timestamp': pd.to_datetime(b.day) + pd.to_timedelta(hv, unit='s')}
                for i in range(LV):
                    cols[f'bid_{i}_price'] = b.bidp[hv, i]; cols[f'bid_{i}_size'] = b.bids[hv, i]
                    cols[f'ask_{i}_price'] = b.askp[hv, i]; cols[f'ask_{i}_size'] = b.asks[hv, i]
                pd.DataFrame(cols).to_parquet(f'{OB_OUT}/{b.day}.parquet')
                if b.trades:
                    tdf = pd.DataFrame(b.trades, columns=['timestamp', 'side', 'size'])
                    os.makedirs(TR_OUT, exist_ok=True)
                    tdf.to_parquet(f'{TR_OUT}/{b.day}.parquet')
                    self.big_thr = float(np.quantile(tdf['size'].values, 0.95))  # 내일용 (causal)
                self.log.info(f"[rollover] persisted {len(hv)} secs, new big_thr={self.big_thr:.3f}")
        except Exception as e:
            self.log.error(f"persist 실패: {e}")
        self.buf = DayBuf(newday)
        self.save_pending()

def run_daemon(eng):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(f'{LOGD}/daemon.log'),
                                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger()
    log.info(f"=== [I] shadow daemon — artifact asof={eng.meta['asof']}, "
             f"norm window {eng.meta['norm_window']} (stale 주의), DB {eng.X.shape} ===")
    sh = Shadow(eng, log)

    def minute_loop():
        last_min = -1
        while True:
            time.sleep(1.0)
            now = datetime.now(timezone.utc)
            mo = now.hour * 60 + now.minute
            if mo != last_min and now.second >= 2:   # 분 시작 +2s: 직전 분 마감 처리
                prev = mo - 1 if mo > 0 else None
                if prev is not None and last_min != -1:
                    try:
                        with sh.lock:
                            pass
                        sh.minute_close(prev)
                    except Exception as e:
                        log.error(f"minute_close({prev}) 실패: {e}")
                last_min = mo
    threading.Thread(target=minute_loop, daemon=True).start()

    while True:
        try:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_open=lambda w: (w.send(json.dumps({'op': 'subscribe', 'args':
                    [f'orderbook.{LV}.{SYM}', f'publicTrade.{SYM}']})),
                    log.info('WS connected+subscribed'))[1],
                on_message=sh.on_message,
                on_error=lambda w, e: log.error(f'WS error: {e}'),
                on_close=lambda w, c, m: log.info('WS closed'))
            sh.connected = True
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            log.error(f'WS loop exception: {e}')
        sh.connected = False
        log.info('reconnect in 5s...')
        time.sleep(5)
