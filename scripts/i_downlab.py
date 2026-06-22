#!/usr/bin/env python3
"""[I] 28단계 — 하락 특화 라벨 신설 (1초 OB/체결 파생, 청산 미수집).
27단계 한계: "있는 47라벨" 안에서만 봄(거울). 하락 전조를 처음부터 라벨링 →
  "하락 예측불가"가 라벨 부족인지 시장 본질인지 가름.
신규 라벨 (전부 causal t≤0, 분말 초 e 까지만):
  지지붕괴: bid_dep_chg30/60 (매수벽 두께 변화), bid_conc (받침 집중도)
  비대칭호가: ask_dep_chg30, book_thin_asym (매수 두께가 매도보다 빨리 마르나)
  하방변동성: dn_rv_60/300 (하락 반값분산), rv_skew_300 (하방 우세도)
  매도공격성: sell_accel (매도비율 가속), sell_spike (집중 매도 버스트)
i_labeling.py 와 동일 데이터/그리드(ffill) — 라이브 파이프라인 미변경(별도 스크립트).
출력: research/i_similarity/downlab.parquet (day,min_of_day,신규라벨)."""
import os, sys, numpy as np, pandas as pd
SYMBOL='ETHUSDT'
OB=f'/Users/mark/mark19_data/{SYMBOL}'; TR=f'/Users/mark/mark19_data/trades_perp/{SYMBOL}'
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LV=20; BURN_MIN=240
obcols=['timestamp','bid_0_price','ask_0_price']
for i in range(LV): obcols+=[f'bid_{i}_size',f'ask_{i}_size']

def ffill_idx(so,n):
    idx=np.full(n,-1,dtype=int); idx[so]=np.arange(len(so)); mask=idx>=0
    pos=np.where(mask,np.arange(n),0); np.maximum.accumulate(pos,out=pos)
    idx=idx[pos]; first=np.argmax(mask); idx[:first]=idx[first]; return idx

def process_day(day):
    try: ob=pd.read_parquet(f'{OB}/{day}.parquet',columns=obcols)
    except Exception: return None
    ts=pd.to_datetime(ob['timestamp'],utc=True)
    keep=(ts.dt.date==pd.Timestamp(day).date()).values
    ob=ob[keep].reset_index(drop=True); ts=ts[keep]
    if len(ob)<5000: return None
    so=np.round((ts-ts.iloc[0]).dt.total_seconds().values).astype(int)
    t0=ts.iloc[0]; n=int(so[-1])+1
    if n<BURN_MIN*60+120: return None
    bp=ob['bid_0_price'].values; ap=ob['ask_0_price'].values; mid=(bp+ap)/2.0
    BS=ob[[f'bid_{i}_size' for i in range(LV)]].values
    AS=ob[[f'ask_{i}_size' for i in range(LV)]].values
    bdep20=BS.sum(1); adep20=AS.sum(1); bdep5=BS[:,:5].sum(1)
    idx=ffill_idx(so,n)
    g_mid=mid[idx]; g_bd20=bdep20[idx]; g_ad20=adep20[idx]; g_bd5=bdep5[idx]
    # trades per-second
    try: tr=pd.read_parquet(f'{TR}/{day}.parquet',columns=['timestamp','side','size'])
    except Exception: return None
    tts=pd.to_datetime(tr['timestamp'],unit='s',utc=True)
    tsec=np.round((tts-t0).dt.total_seconds().values).astype(int)
    m=(tsec>=0)&(tsec<n); tsec=tsec[m]; tside=tr['side'].values[m]; tsz=tr['size'].values[m]
    buyv=np.zeros(n); sellv=np.zeros(n); isbuy=(tside=='Buy')
    np.add.at(buyv,tsec[isbuy],tsz[isbuy]); np.add.at(sellv,tsec[~isbuy],tsz[~isbuy])
    cbuy=np.cumsum(buyv); csell=np.cumsum(sellv)
    # downside/upside rv (per-sec log ret)
    lr=np.zeros(n); lr[1:]=np.log(g_mid[1:]/g_mid[:-1]+1e-12)
    dn=np.minimum(lr,0.0); up=np.maximum(lr,0.0)
    cdn2=np.cumsum(dn*dn); cup2=np.cumsum(up*up)
    def semivol(c2,s,w):
        a=max(s-w,0); return np.sqrt((c2[s]-c2[a])/max(s-a,1))*1e4
    # minute-end seconds
    nmin=n//60; e=np.arange(1,nmin+1)*60-1; e=e[e<n]; nmin=len(e)
    def rel(g,w):
        prev=g[np.maximum(e-w,0)]; return (g[e]-prev)/(np.abs(prev)+1e-9)
    feat={}
    feat['bid_dep_chg30']=rel(g_bd20,30); feat['bid_dep_chg60']=rel(g_bd20,60)
    feat['ask_dep_chg30']=rel(g_ad20,30)
    feat['book_thin_asym']=feat['bid_dep_chg30']-feat['ask_dep_chg30']
    feat['bid_conc']=g_bd5[e]/(g_bd20[e]+1e-9)
    feat['dn_rv_60']=np.array([semivol(cdn2,s,60) for s in e])
    feat['dn_rv_300']=np.array([semivol(cdn2,s,300) for s in e])
    up300=np.array([semivol(cup2,s,300) for s in e])
    feat['rv_skew_300']=(feat['dn_rv_300']-up300)/(feat['dn_rv_300']+up300+1e-9)
    def sratio(s,w):
        a=max(s-w,0); bb=cbuy[s]-cbuy[a]; ss=csell[s]-csell[a]; return ss/(bb+ss+1e-9)
    sr60=np.array([sratio(s,60) for s in e]); sr300=np.array([sratio(s,300) for s in e])
    feat['sell_accel']=sr60-sr300
    def sspike(s):
        a=max(s-60,0); seg=sellv[a:s+1]
        if len(seg)<5: return 0.0
        med=np.median(seg[seg>0]) if (seg>0).any() else 0.0
        return seg.max()/(med+1e-9) if med>0 else 0.0
    feat['sell_spike']=np.array([sspike(s) for s in e])
    df=pd.DataFrame(feat); df.insert(0,'min_of_day',np.arange(nmin)); df.insert(0,'day',day)
    df['mid']=g_mid[e]
    return df.iloc[BURN_MIN:].reset_index(drop=True)

def main():
    days=sorted(d[:-8] for d in os.listdir(OB) if d.endswith('.parquet'))
    STEP=int(os.environ.get('STEP','3'))
    days=days[::STEP]
    LIMIT=int(os.environ.get('LIMIT','0'))
    if LIMIT: days=days[:LIMIT]
    print(f"[downlab] {len(days)} days (STEP={STEP})",flush=True)
    from time import time as _t; t0=_t(); out=[]
    for i,d in enumerate(days):
        r=process_day(d)
        if r is not None: out.append(r)
        if i%30==0: print(f"  {i}/{len(days)} {_t()-t0:.0f}s rows={sum(len(x) for x in out)}",flush=True)
    R=pd.concat(out,ignore_index=True)
    R.to_parquet(f'{OUT}/downlab.parquet')
    print(f"[done] {len(R)} rows, {R.day.nunique()} days, {_t()-t0:.0f}s -> downlab.parquet")

if __name__=='__main__':
    main()
