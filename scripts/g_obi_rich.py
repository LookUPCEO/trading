#!/usr/bin/env python3
"""
A-1: OBI child — 풍부한 호가/체결로 재계산. 있는 데이터만 (multiple testing 경계).
모든 feature t<=0 (진입 t=0 시점), target = 5m fwd ret. lookahead 엄격.

feature (전부 causal):
  깊이 OBI: obi1/obi5/obi20/obi50 (size 불균형), obi_wtd (거리가중)
  trades flow: flow_30s/1m/5m (aggressor buy-sell vol 불균형), bigflow_5m (큰체결 net)
  OB 동역학: dobi5_30/60 (직전 30/60s OBI 변화)
출력: 각 feature 단독 causal gross + |feature|분위 gradient + 깊이/trades/동역학 비교.
"""
import os, json, glob
import numpy as np
import pandas as pd

OB='/Users/mark/mark19_data/ETHUSDT'
TR='/Users/mark/mark19_data/trades_perp/ETHUSDT'
OUT='/Users/mark/Desktop/Mark/mark19/research/g_sibling'
os.makedirs(OUT,exist_ok=True)

all_days=sorted([d[:-8] for d in os.listdir(OB) if d.endswith('.parquet')])
STEP=int(os.environ.get('STEP','24'))
days=all_days[::STEP]
print(f'[setup] {len(days)} days, {days[0]}~{days[-1]}')

LV=50
obcols=['timestamp']
for i in range(LV):
    obcols+=[f'bid_{i}_price',f'bid_{i}_size',f'ask_{i}_price',f'ask_{i}_size']
WIN=300; HB=900

def ffill_idx(so, n):
    """second -> source row index (last snapshot <= sec)."""
    idx=np.full(n,-1,dtype=int); idx[so]=np.arange(len(so))
    mask=idx>=0
    pos=np.where(mask,np.arange(n),0); np.maximum.accumulate(pos,out=pos)
    idx=idx[pos]; first=np.argmax(mask); idx[:first]=idx[first]
    return idx

rows=[]
for di,day in enumerate(days):
    try:
        ob=pd.read_parquet(f'{OB}/{day}.parquet',columns=obcols)
    except Exception as e:
        print('skip ob',day,e); continue
    ts=pd.to_datetime(ob['timestamp'],utc=True)
    keep=ts.dt.date==pd.Timestamp(day).date()
    ob=ob[keep]; ts=ts[keep]
    if len(ob)<1000: continue
    so=np.round((ts-ts.iloc[0]).dt.total_seconds().values).astype(int)
    t0=ts.iloc[0]
    n=int(so[-1])+1
    bp=ob['bid_0_price'].values; ap=ob['ask_0_price'].values
    mid_row=(bp+ap)/2.0
    # depth sizes
    BS=ob[[f'bid_{i}_size' for i in range(LV)]].values
    AS=ob[[f'ask_{i}_size' for i in range(LV)]].values
    BP=ob[[f'bid_{i}_price' for i in range(LV)]].values
    AP=ob[[f'ask_{i}_price' for i in range(LV)]].values
    def obi(d):
        b=BS[:,:d].sum(1); a=AS[:,:d].sum(1); return (b-a)/(b+a+1e-9)
    obi1=obi(1); obi5=obi(5); obi20=obi(20); obi50=obi(50)
    # 거리가중 OBI (가까운 레벨 가중↑: w=1/(1+|price-mid|/tick), tick≈0.01)
    midv=mid_row[:,None]
    wb=1.0/(1.0+np.abs(BP-midv)/0.01); wa=1.0/(1.0+np.abs(AP-midv)/0.01)
    bw=(BS*wb).sum(1); aw=(AS*wa).sum(1); obi_wtd=(bw-aw)/(bw+aw+1e-9)
    # second grid via idx
    idx=ffill_idx(so,n)
    g_mid=mid_row[idx]; g_obi5=obi5[idx]
    g={'obi1':obi1[idx],'obi5':g_obi5,'obi20':obi20[idx],'obi50':obi50[idx],'obi_wtd':obi_wtd[idx]}

    # trades -> per-second buy/sell vol
    try:
        tr=pd.read_parquet(f'{TR}/{day}.parquet',columns=['timestamp','side','size','price'])
    except Exception as e:
        print('skip tr',day,e); continue
    tts=pd.to_datetime(tr['timestamp'],unit='s',utc=True)
    tsec=np.round((tts-t0).dt.total_seconds().values).astype(int)
    m=(tsec>=0)&(tsec<n)
    tsec=tsec[m]; tside=tr['side'].values[m]; tsz=tr['size'].values[m]
    buyv=np.zeros(n); sellv=np.zeros(n); bign=np.zeros(n)
    isbuy=(tside=='Buy')
    np.add.at(buyv,tsec[isbuy],tsz[isbuy])
    np.add.at(sellv,tsec[~isbuy],tsz[~isbuy])
    # 큰체결: 그 날 size p95 이상
    if len(tsz)>100:
        big=tsz>=np.quantile(tsz,0.95)
        np.add.at(bign,tsec[big], np.where(isbuy[big],tsz[big],-tsz[big]))
    cbuy=np.cumsum(buyv); csell=np.cumsum(sellv); cbig=np.cumsum(bign)
    def flow(a,b):  # net aggressor imbalance over (a,b]
        bb=cbuy[b]-cbuy[a]; ss=csell[b]-csell[a]; return (bb-ss)/(bb+ss+1e-9)

    yr=day[:4]
    for s in range(HB,n-WIN,WIN):
        p0=g_mid[s]; p1=g_mid[s+WIN]
        if not np.isfinite([p0,p1]).all() or p0<=0: continue
        ret=(p1-p0)/p0*1e4
        seg=g_mid[s-HB:s+1]; rr=np.diff(seg)/seg[:-1]
        vol=np.std(rr)*1e4 if len(rr)>5 else np.nan
        rows.append(dict(day=day,yr=yr,s=s,ret=ret,oracle=abs(ret),vol=vol,
            obi1=g['obi1'][s],obi5=g['obi5'][s],obi20=g['obi20'][s],obi50=g['obi50'][s],obi_wtd=g['obi_wtd'][s],
            flow30=flow(s-30,s),flow1m=flow(s-60,s),flow5m=flow(s-300,s),
            bigflow5m=(cbig[s]-cbig[s-300]),
            dobi5_30=g['obi5'][s]-g['obi5'][s-30],
            dobi5_60=g['obi5'][s]-g['obi5'][s-60]))
    if di%10==0: print(f'  [{di}/{len(days)}] {day} rows={len(rows)}')

R=pd.DataFrame(rows)
R.to_parquet(f'{OUT}/windows_rich.parquet')
print(f'\n[done] {len(R)} windows')

FEATS=['obi1','obi5','obi20','obi50','obi_wtd','flow30','flow1m','flow5m','bigflow5m','dobi5_30','dobi5_60']
print('\n=== 각 feature 단독 causal gross (side=sign(feat)*ret), bp ===')
res={}
for f in FEATS:
    side=np.sign(R[f].values); net=side*R.ret.values
    g=np.nanmean(net); win=np.nanmean(net>0)
    # top-quintile (|feat| 강할 때)
    fa=np.abs(R[f].values)
    q=fa>=np.nanquantile(fa,0.8)
    gq=np.nanmean(net[q])
    res[f]=dict(gross=float(g),win=float(win),gross_q5=float(gq))
    print(f"  {f:10s} gross {g:+.3f}  win {win:.3f}  |q5 strong| {gq:+.3f}bp")

print('\n=== 깊이 비교 (obi1<5<20<50<wtd 어디가 최선) ===')
for f in ['obi1','obi5','obi20','obi50','obi_wtd']:
    print(f"  {f:8s}: gross {res[f]['gross']:+.3f}  q5 {res[f]['gross_q5']:+.3f}")
print('\n=== trades vs OB 상관 (다른 정보인가) ===')
print(f"  corr(obi5, flow5m) = {R[['obi5','flow5m']].corr().iloc[0,1]:+.3f}")
print(f"  corr(obi5, flow30) = {R[['obi5','flow30']].corr().iloc[0,1]:+.3f}")
print(f"  corr(obi5, dobi5_60)= {R[['obi5','dobi5_60']].corr().iloc[0,1]:+.3f}")

json.dump(res,open(f'{OUT}/obi_rich_single.json','w'),indent=2)
print('\nsaved windows_rich.parquet + obi_rich_single.json')
