#!/usr/bin/env python3
"""
H-1: Binance(spot) -> Bybit(perp) 선행 관계 (lead-lag), lookahead 엄격.

정렬/지연 (핵심):
  Binance 1s kline: open_time t(초경계), close = [t, t+1s) 마지막 체결가 (정보 ~t+0.999s 까지).
  Bybit OB snapshot: 초 t 의 스냅 ≈ t+0.975s.
  → lookahead 방지: Binance 신호는 close[t] (정보 t+0.999 까지),
    Bybit 진입은 그 다음 초부터 = mid[t+1+δ] (δ=latency 초). 신호(t+0.999) < 진입(t+1.975+δ).
  → 자연히 ~1s+ latency 내장. δ sweep 으로 latency haircut.

작업1: lead-lag cross-corr (누가 먼저). 작업2: 방향 gross+fee+시기. 작업3: latency+fill haircut.
"""
import os, glob
import numpy as np
import pandas as pd
import json

BNB='/tmp/bnb'
OB='/Users/mark/mark19_data/ETHUSDT'
OUT='/Users/mark/Desktop/Mark/mark19/research/h_binance'
os.makedirs(OUT,exist_ok=True)

days=sorted([os.path.basename(f).replace('ETHUSDT-1s-','').replace('.csv','')
             for f in glob.glob(f'{BNB}/ETHUSDT-1s-*.csv')])
days=[d for d in days if d!='2024-06-01']  # 테스트파일 제외(없는 Bybit일 가능)
print(f'[setup] pilot days {days}')

def load_bnb(day):
    df=pd.read_csv(f'{BNB}/ETHUSDT-1s-{day}.csv',header=None,
        names=['ot','o','h','l','c','v','ct','qv','n','tbb','tbq','ig'])
    ot=df['ot'].values.astype('int64')
    # Binance 단위 가변: ms(13자리)/µs(16자리, 2025+). epoch sec(~1.7e9) 로 정규화
    o0=int(ot[0])
    div=1_000_000 if o0>=10**15 else (1000 if o0>=10**12 else 1)
    sec=(ot//div)
    return sec, df['c'].values.astype(float), df['tbb'].values.astype(float), df['v'].values.astype(float)

def load_bybit(day):
    ob=pd.read_parquet(f'{OB}/{day}.parquet',columns=['timestamp','bid_0_price','ask_0_price'])
    ts=pd.to_datetime(ob['timestamp'],utc=True); keep=ts.dt.date==pd.Timestamp(day).date()
    ob=ob[keep]; ts=ts[keep]
    epoch=pd.Timestamp('1970-01-01',tz='UTC')
    sec=np.floor((ts-epoch).dt.total_seconds().values).astype('int64')  # epoch sec (floor); 스냅은 .975
    mid=((ob['bid_0_price']+ob['ask_0_price'])/2).values
    return sec, mid

# ── lead-lag cross-corr + 방향 gross ──
LAGS=range(-5,6)
corr_acc={l:[] for l in LAGS}
rows=[]
for day in days:
    try:
        bsec,bc,btbb,bv=load_bnb(day)
        ysec,ymid=load_bybit(day)
    except Exception as e:
        print('skip',day,e); continue
    # 공통 epoch second grid
    lo=max(bsec.min(),ysec.min()); hi=min(bsec.max(),ysec.max())
    n=hi-lo+1
    gb=np.full(n,np.nan); gy=np.full(n,np.nan)
    bi=bsec-lo; m=(bi>=0)&(bi<n); gb[bi[m]]=bc[m]
    yi=ysec-lo; m=(yi>=0)&(yi<n); gy[yi[m]]=ymid[m]
    def ff(a):
        mk=~np.isnan(a);
        if not mk.any(): return a
        pos=np.where(mk,np.arange(len(a)),0); np.maximum.accumulate(pos,out=pos)
        a=a[pos]; a[:np.argmax(mk)]=a[np.argmax(mk)]; return a
    gb=ff(gb); gy=ff(gy)
    # 1s 로그수익
    rb=np.diff(np.log(gb)); ry=np.diff(np.log(gy))
    # cross-corr: corr(rb[t], ry[t+lag]). lag>0 = Binance 가 먼저(Bybit 미래와 상관)
    for l in LAGS:
        if l>=0: a=rb[:len(rb)-l]; b=ry[l:len(ry)] if l>0 else ry[:len(ry)]
        else: a=rb[-l:]; b=ry[:len(ry)+l]
        L=min(len(a),len(b)); a=a[:L]; b=b[:L]
        if L>100 and a.std()>0 and b.std()>0:
            corr_acc[l].append(np.corrcoef(a,b)[0,1])
    # 방향 gross: Binance close[t] 대비 close[t-k] 모멘텀 -> Bybit (t+1+δ, t+1+δ+H]
    yr=day[:4]
    for k in [3,5,10]:        # Binance lookback k초
      for H in [10,30,60]:    # Bybit hold H초
        for d in [1,2,5]:     # latency δ초 (진입 지연)
            t=np.arange(k, n-1-d-H)
            bnb_mom=np.sign(gb[t]-gb[t-k])              # 신호: 정보 t 까지
            entry=gy[t+1+d]                              # Bybit 진입 t+1+δ (신호 후)
            exitp=gy[t+1+d+H]
            ret=(exitp-entry)/entry*1e4
            gross=np.nanmean(bnb_mom*ret)
            rows.append(dict(day=day,yr=yr,k=k,H=H,delta=d,gross=float(gross),
                             win=float(np.nanmean(bnb_mom*ret>0)),n=len(t)))

# cross-corr 요약
print('\n=== lead-lag cross-corr (corr(BinanceRet[t], BybitRet[t+lag])) ===')
print('  lag>0 = Binance 선행. lag(s): corr')
cc={}
for l in LAGS:
    v=np.mean(corr_acc[l]) if corr_acc[l] else np.nan; cc[l]=float(v)
    star=' <== peak?' if l>0 else (' (동시)' if l==0 else '')
    print(f"   {l:+d}s: {v:+.4f}{star}")
peak=max(cc,key=lambda l:cc[l])
print(f"  peak lag = {peak:+d}s (corr {cc[peak]:+.4f})  -> {'Binance 선행' if peak>0 else '동시/Bybit선행' }")

D=pd.DataFrame(rows)
D.to_parquet(f'{OUT}/leadlag.parquet')
print('\n=== 방향 gross (Binance 모멘텀 -> Bybit), bp. δ=latency초 ===')
piv=D.groupby(['delta','H'])['gross'].mean().unstack()
print(piv.round(3).to_string())
print('\n=== δ=1s, k/H 별 (latency 1초 가정) ===')
for _,r in D[D.delta==1].groupby(['k','H'])['gross'].mean().reset_index().iterrows():
    print(f"   k={int(r.k)}s H={int(r.H)}s: gross {r.gross:+.3f}bp")
print('\n=== best 설정 시기분해 ===')
best=D.groupby(['k','H','delta'])['gross'].mean().idxmax()
print(f"  best (k,H,δ)={best} gross {D.groupby(['k','H','delta'])['gross'].mean().max():+.3f}bp")
bb=D[(D.k==best[0])&(D.H==best[1])&(D.delta==best[2])]
for yr,g in bb.groupby('yr'):
    print(f"   {yr}: gross {g.gross.mean():+.3f}bp  win {g.win.mean():.3f}")

json.dump({'crosscorr':cc,'peak_lag':int(peak),
           'best_setting':[int(x) for x in best],
           'best_gross':float(D.groupby(['k','H','delta'])['gross'].mean().max()),
           'gross_by_delta_H':{f'd{d}_H{h}':float(piv.loc[d,h]) for d in piv.index for h in piv.columns}},
          open(f'{OUT}/leadlag.json','w'),indent=2)
print('\nsaved leadlag.json + leadlag.parquet')
