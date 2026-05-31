#!/usr/bin/env python3
"""
A-1b 갈래4: OOS 생존 결합규칙(obi5×vol AND, strong) 에 fill adverse selection + 시기.
OOS test set 윈도우만 (선택=train direction, 평가=test). 실제 trades 체결판정.
"""
import os, json
import numpy as np
import pandas as pd

OB='/Users/mark/mark19_data/ETHUSDT'
TR='/Users/mark/mark19_data/trades_perp/ETHUSDT'
OUT='/Users/mark/Desktop/Mark/mark19/research/g_sibling'
R=pd.read_parquet(f'{OUT}/windows_rich.parquet').dropna().reset_index(drop=True)
days=sorted(R.day.unique()); cut=days[int(len(days)*0.6)]
tr=R[R.day<cut]; te=R[R.day>=cut].copy()
# obi5×vol 방향: train 에서 obi5 부호방향 (vol 은 게이트라 방향성 없음 → obi5 sign)
dir_obi5=np.sign(np.mean(np.sign(tr.obi5.values)*tr.ret.values))
thr_obi=np.quantile(np.abs(tr.obi5.values),0.8); thr_vol=np.quantile(tr.vol.values,0.8)
fire=(np.abs(te.obi5.values)>=thr_obi)&(te.vol.values>=thr_vol)
sub=te[fire].copy(); sub['side']=dir_obi5*np.sign(sub.obi5.values)
print(f'[setup] OOS test obi5×vol fire: {len(sub)} (dir_obi5={dir_obi5:+.0f})')
print(f"  naive(mid) gross {np.mean(sub.side.values*sub.ret.values):+.3f}bp")
print('  시기:')
for yr,g in sub.groupby('yr'):
    print(f"    {yr}: {np.mean(g.side.values*g.ret.values):+.3f}bp n={len(g)}")

FILLW=60
rows=[]
for day in sorted(sub.day.unique()):
    s_=sub[sub.day==day]
    ob=pd.read_parquet(f'{OB}/{day}.parquet',columns=['timestamp','bid_0_price','ask_0_price'])
    ts=pd.to_datetime(ob['timestamp'],utc=True); keep=ts.dt.date==pd.Timestamp(day).date()
    ob=ob[keep]; ts=ts[keep]
    so=np.round((ts-ts.iloc[0]).dt.total_seconds().values).astype(int); t0=ts.iloc[0]; n=int(so[-1])+1
    bid=ob['bid_0_price'].values; ask=ob['ask_0_price'].values; mid=(bid+ask)/2
    def ff(a):
        m=~np.isnan(a); pos=np.where(m,np.arange(len(a)),0); np.maximum.accumulate(pos,out=pos)
        r=a[pos]; r[:np.argmax(m)]=a[np.argmax(m)]; return r
    gb=ff(np.where(np.isin(np.arange(n),so),0,np.nan)*0+np.nan) # placeholder
    gb=np.full(n,np.nan);gb[so]=bid;gb=ff(gb)
    ga=np.full(n,np.nan);ga[so]=ask;ga=ff(ga)
    gm=np.full(n,np.nan);gm[so]=mid;gm=ff(gm)
    trd=pd.read_parquet(f'{TR}/{day}.parquet',columns=['timestamp','side','size','price'])
    tts=pd.to_datetime(trd['timestamp'],unit='s',utc=True)
    tsec=np.round((tts-t0).dt.total_seconds().values).astype(int)
    m=(tsec>=0)&(tsec<n); tsec=tsec[m]; tp=trd['price'].values[m]; tsd=trd['side'].values[m]
    o=np.argsort(tsec); tsec=tsec[o]; tp=tp[o]; tsd=tsd[o]
    for _,r in s_.iterrows():
        s=int(r.s); sd=r.side
        if s+300>=n: continue
        j0=np.searchsorted(tsec,s); j1=np.searchsorted(tsec,s+FILLW)
        sp=tp[j0:j1]; ss=tsd[j0:j1]
        if sd>0:
            L=gb[s]; hit=((ss=='Sell')&(sp<=L)).any()
            pm=(gm[s+300]-L)/L*1e4 if hit else np.nan
            pt=(gm[s+300]-ga[s])/ga[s]*1e4
        else:
            L=ga[s]; hit=((ss=='Buy')&(sp>=L)).any()
            pm=(L-gm[s+300])/L*1e4 if hit else np.nan
            pt=(gb[s]-gm[s+300])/gb[s]*1e4
        rows.append(dict(filled=bool(hit),pm=pm,pt=pt,naive=sd*(gm[s+300]-gm[s])/gm[s]*1e4))
F=pd.DataFrame(rows); ff_=F[F.filled]
print(f"\n=== fill adverse selection (obi5×vol OOS 생존규칙) ===")
print(f"  n={len(F)} fill_rate {F.filled.mean():.3f}")
print(f"  naive(mid) {F.naive.mean():+.3f}bp")
print(f"  maker 체결분 gross {ff_.pm.mean():+.3f}  -> net(fee 7.5) {ff_.pm.mean()-7.5:+.3f}bp")
print(f"  maker 체결분 net(fee 4 M+M 낙관) {ff_.pm.mean()-4:+.3f}bp")
print(f"  taker gross {F.pt.mean():+.3f}  -> net(fee 11) {F.pt.mean()-11:+.3f}bp")
json.dump(dict(n=len(F),fill_rate=float(F.filled.mean()),naive=float(F.naive.mean()),
    maker_gross=float(ff_.pm.mean()),maker_net75=float(ff_.pm.mean()-7.5),maker_net4=float(ff_.pm.mean()-4),
    taker_net=float(F.pt.mean()-11)),open(f'{OUT}/obi_combo_fill.json','w'),indent=2)
print('saved obi_combo_fill.json')
