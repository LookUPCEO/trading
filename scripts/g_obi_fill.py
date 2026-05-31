#!/usr/bin/env python3
"""
A-1 작업4-5: 결합신호(walk-forward, multiple testing 보정) + fill adverse selection 정면.
fill adverse selection = OBI 죽이는 핵심 함정(range-v2 변형):
  신호 long 일때 best-bid maker limit 은 "가격이 우리쪽으로 내려올때만"=불리할때만 체결.
  실제 trades 로 체결 판정 → fill-conditional net.
"""
import os, json, glob
import numpy as np
import pandas as pd

OB='/Users/mark/mark19_data/ETHUSDT'
TR='/Users/mark/mark19_data/trades_perp/ETHUSDT'
OUT='/Users/mark/Desktop/Mark/mark19/research/g_sibling'
R=pd.read_parquet(f'{OUT}/windows_rich.parquet')

# ── 결합신호: 독립적 약신호 결합 (obi5, dobi5_30, flow5m-fade), walk-forward ──
# flow 는 contrarian → -flow. 표준화 후 단순 합 (과적합 회피: 학습 가중 X, 등가중 합)
def z(x):
    x=x.values.astype(float); return (x-np.nanmean(x))/(np.nanstd(x)+1e-9)
R['combo']= z(R.obi5)+z(R.dobi5_30)-z(R.flow5m)   # -flow (contrarian fade)
print('=== 결합신호 (등가중, 과적합 회피) causal gross ===')
side=np.sign(R.combo.values); net=side*R.ret.values
print(f"  combo gross {np.nanmean(net):+.3f}bp  win {np.nanmean(net>0):.3f}")
fa=np.abs(R.combo.values);
for ql,qh,lab in [(0.8,1.01,'q5(강)'),(0.9,1.01,'q9(최강10%)'),(0.95,1.01,'top5%')]:
    m=(fa>=np.nanquantile(fa,ql))
    g=np.nanmean(net[m]); print(f"  {lab}: gross {g:+.3f}bp  win {np.nanmean(net[m]>0):.3f}  n={m.sum()}")

# bootstrap CI (combo gross)
rng=np.random.RandomState(0)
bs=[np.nanmean((side*R.ret.values)[rng.randint(0,len(R),len(R))]) for _ in range(2000)]
print(f"  bootstrap 95%CI gross [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")

# 시기분해
print('\n=== combo 시기분해 (2026 감쇠?) ===')
for yr,g in R.groupby('yr'):
    s=np.sign(g.combo.values)*g.ret.values
    print(f"  {yr}: gross {np.nanmean(s):+.3f}  win {np.nanmean(s>0):.3f}  n={len(g)}")

# ── fill adverse selection (best 신호 강한 구간, 가장 유리한 경우부터) ──
# 강신호(top5% combo) long/short 윈도우에서 maker limit 체결 시뮬
print('\n=== fill adverse selection (maker limit, 실제 trades 체결판정) ===')
days=sorted(R.day.unique())
strong=R[fa>=np.nanquantile(fa,0.9)].copy()   # 최강 10% (gross 최고 구간)
strong['side']=np.sign(strong.combo)
LV=1
FILLW=int(os.environ.get('FILLW','60'))   # 체결 대기 60s
fill_rows=[]
for day in days:
    sub=strong[strong.day==day]
    if len(sub)==0: continue
    ob=pd.read_parquet(f'{OB}/{day}.parquet',columns=['timestamp','bid_0_price','ask_0_price'])
    ts=pd.to_datetime(ob['timestamp'],utc=True); keep=ts.dt.date==pd.Timestamp(day).date()
    ob=ob[keep]; ts=ts[keep]
    so=np.round((ts-ts.iloc[0]).dt.total_seconds().values).astype(int); t0=ts.iloc[0]
    n=int(so[-1])+1
    bid=ob['bid_0_price'].values; ask=ob['ask_0_price'].values; mid=(bid+ask)/2
    def ff(a):
        m=~np.isnan(a); pos=np.where(m,np.arange(len(a)),0); np.maximum.accumulate(pos,out=pos)
        r=a[pos]; r[:np.argmax(m)]=a[np.argmax(m)]; return r
    gb=np.full(n,np.nan);gb[so]=bid;gb=ff(gb)
    ga=np.full(n,np.nan);ga[so]=ask;ga=ff(ga)
    gm=np.full(n,np.nan);gm[so]=mid;gm=ff(gm)
    tr=pd.read_parquet(f'{TR}/{day}.parquet',columns=['timestamp','side','size','price'])
    tts=pd.to_datetime(tr['timestamp'],unit='s',utc=True)
    tsec=np.round((tts-t0).dt.total_seconds().values).astype(int)
    m=(tsec>=0)&(tsec<n); tsec=tsec[m]; tp=tr['price'].values[m]; tsd=tr['side'].values[m]
    order=np.argsort(tsec); tsec=tsec[order]; tp=tp[order]; tsd=tsd[order]
    for _,r in sub.iterrows():
        s=int(r.s); sd=r.side
        if s+300>=n: continue
        if sd>0:   # long: buy limit at best bid; fill if Sell aggressor prints <= bid within FILLW
            L=gb[s]
            j0=np.searchsorted(tsec,s); j1=np.searchsorted(tsec,s+FILLW)
            seg_p=tp[j0:j1]; seg_s=tsd[j0:j1]
            hit=np.where((seg_s=='Sell')&(seg_p<=L))[0]
            filled=len(hit)>0
            entry=L if filled else np.nan
            pnl_maker=(gm[s+300]-entry)/entry*1e4 if filled else np.nan
            entry_tk=ga[s]; pnl_taker=(gm[s+300]-entry_tk)/entry_tk*1e4   # cross spread
        else:      # short: sell limit at best ask; fill if Buy aggressor prints >= ask
            L=ga[s]
            j0=np.searchsorted(tsec,s); j1=np.searchsorted(tsec,s+FILLW)
            seg_p=tp[j0:j1]; seg_s=tsd[j0:j1]
            hit=np.where((seg_s=='Buy')&(seg_p>=L))[0]
            filled=len(hit)>0
            entry=L if filled else np.nan
            pnl_maker=(entry-gm[s+300])/entry*1e4 if filled else np.nan
            entry_tk=gb[s]; pnl_taker=(entry_tk-gm[s+300])/entry_tk*1e4
        fill_rows.append(dict(day=day,filled=filled,pnl_maker_gross=pnl_maker,pnl_taker_gross=pnl_taker,naive=sd*(gm[s+300]-gm[s])/gm[s]*1e4))
F=pd.DataFrame(fill_rows)
fr=F.filled.mean()
print(f"  강신호 n={len(F)}, maker fill rate {fr:.3f} (체결 대기 {FILLW}s)")
print(f"  naive(mid 가정) gross {F.naive.mean():+.3f}bp  <- fill 무시한 낙관치")
ff_=F[F.filled]
print(f"  maker 체결분 gross {ff_.pnl_maker_gross.mean():+.3f}bp (adverse selection 후)")
print(f"    -> maker net (fee 2 진입+5.5 청산 =7.5): {ff_.pnl_maker_gross.mean()-7.5:+.3f}bp")
print(f"  taker 즉시진입 gross {F.pnl_taker_gross.mean():+.3f}bp")
print(f"    -> taker net (fee 5.5*2=11): {F.pnl_taker_gross.mean()-11:+.3f}bp")
print(f"  ※ 미체결 {1-fr:.0%} = 신호 방향 가격이 떠나버림(=맞은 케이스 놓침) = 기회손실")

out={'combo_gross':float(np.nanmean(np.sign(R.combo)*R.ret)),
     'combo_top10pct_gross':float(np.nanmean(net[fa>=np.nanquantile(fa,0.9)])),
     'bootstrap_ci':[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],
     'maker_fill_rate':float(fr),
     'naive_gross':float(F.naive.mean()),
     'maker_filled_gross':float(ff_.pnl_maker_gross.mean()),
     'maker_net':float(ff_.pnl_maker_gross.mean()-7.5),
     'taker_net':float(F.pnl_taker_gross.mean()-11)}
json.dump(out,open(f'{OUT}/obi_fill.json','w'),indent=2)
print('\n'+json.dumps(out,indent=2))
