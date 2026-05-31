#!/usr/bin/env python3
"""
G 형제 2 (실현가능 oracle = hedge/straddle+stop) + 형제3 심화 (OBI×vol) + 시각화.

형제2: t=0 에 long+short 동시진입(delta-neutral). loser 를 stop -X 에 cut(taker),
  winner 는 t=300 hold. = "방향 결정 안 하고 oracle |move| 잡기" 시도.
  실제론 breakout/stop 과 동형 → 이미 죽은 가지인지 1Hz 로 확인.
형제3: 강한 OBI × 저vol 에서 방향 net 이 fee 넘나 (유일하게 기울기 보인 곳).
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/Users/mark/Desktop/Mark/mark19/research/g_sibling'
RAW = '/Users/mark/mark19_data/ETHUSDT'
R = pd.read_parquet(f'{OUT}/windows.parquet')

# ── 형제3 심화: OBI × vol 2D (강한 imbalance + 저vol 조합) ──
R['volq'] = pd.qcut(R.vol_prior.rank(method='first'), 5, labels=[1,2,3,4,5])
R['obiq'] = pd.qcut(R.obi0.abs().rank(method='first'), 5, labels=[1,2,3,4,5])
print('=== 형제3 심화: net_obi (bp) by [|OBI|q × volq] ===')
piv = R.pivot_table(values='net_obi', index='obiq', columns='volq', aggfunc='mean', observed=True)
print(piv.round(3).to_string())
print('\n=== 같은 셀 표본수 ===')
print(R.pivot_table(values='net_obi', index='obiq', columns='volq', aggfunc='size', observed=True).to_string())
print('\n=== net_obi by |OBI|q (전체 + winrate + p10/p90) ===')
for oq, g in R.groupby('obiq', observed=True):
    print(f"  obiq{oq}: mean {g.net_obi.mean():+.3f}  win {(g.net_obi>0).mean():.3f}  "
          f"p10 {g.net_obi.quantile(.1):+.2f}  p90 {g.net_obi.quantile(.9):+.2f}  n={len(g)}")

# obiq5 시기분해 (시기 안정성 — 흥분 금지 체크)
print('\n=== obiq5 (강한 imbalance) 시기분해 ===')
g5 = R[R.obiq==5]
for yr, g in g5.groupby('yr'):
    print(f"  {yr}: net_obi {g.net_obi.mean():+.3f}bp  win {(g.net_obi>0).mean():.3f}  n={len(g)}")

# ── 형제2: straddle + stop (1Hz path) ──
# 재로딩 필요 (path) → windows.parquet 엔 s,day 있음. 일부 날만 path 시뮬.
print('\n=== 형제2: straddle+stop (대표 8일 1Hz path) ===')
days = sorted(R.day.unique())[::6][:8]
LV=5
cols=['timestamp']
for i in range(LV):
    cols+=[f'bid_{i}_price',f'bid_{i}_size',f'ask_{i}_price',f'ask_{i}_size']
WIN=300; HB=900
straddle_rows=[]
for day in days:
    df=pd.read_parquet(f'{RAW}/{day}.parquet', columns=['timestamp','bid_0_price','ask_0_price'])
    ts=pd.to_datetime(df['timestamp'],utc=True)
    df=df[ts.dt.date==pd.Timestamp(day).date()]
    ts=pd.to_datetime(df['timestamp'],utc=True)
    so=np.round((ts-ts.iloc[0]).dt.total_seconds().values).astype(int)
    mid=((df['bid_0_price']+df['ask_0_price'])/2.0).values
    maxs=int(so[-1])
    grid=np.full(maxs+1,np.nan); grid[so]=mid
    mask=~np.isnan(grid); idx=np.where(mask,np.arange(len(grid)),0)
    np.maximum.accumulate(idx,out=idx); grid=grid[idx]
    first=np.argmax(mask); grid[:first]=grid[first]
    for s in range(HB, maxs-WIN, WIN):
        p0=grid[s]; path=grid[s:s+WIN+1]
        if not np.isfinite(p0) or p0<=0: continue
        r=(path-p0)/p0*1e4   # bp path (long pov)
        # straddle: long & short. stop loser at -STOP (taker). winner hold to end.
        for STOP in [10,20,40]:
            # long leg PnL: stopped if r hits -STOP first, else r[-1]
            hit_dn = np.where(r<=-STOP)[0]
            hit_up = np.where(r>= STOP)[0]
            t_dn = hit_dn[0] if len(hit_dn) else 10**9
            t_up = hit_up[0] if len(hit_up) else 10**9
            # long leg: stop at -STOP if t_dn first; else end r[-1]
            long_pnl  = -STOP if t_dn<t_up and t_dn<10**9 else r[-1]
            # short leg (pnl = -r): stop at -STOP if up move (t_up) first
            short_pnl = -STOP if t_up<t_dn and t_up<10**9 else -r[-1]
            # one leg stopped(taker 5.5*2=11 RT), other held(maker entry+taker exit ~7.5)
            # 보수: 양 leg 진입 maker(2)*2=4, 청산: stop leg taker5.5, hold leg taker5.5 → +11
            gross=long_pnl+short_pnl
            net=gross-(4+11)  # 진입 maker2x4 + 청산 taker2x ~11 (보수)
            straddle_rows.append(dict(day=day,STOP=STOP,gross=gross,net=net))
SD=pd.DataFrame(straddle_rows)
for STOP,g in SD.groupby('STOP'):
    print(f"  stop {STOP}bp: gross {g.gross.mean():+.2f}bp  net(fee~15) {g.net.mean():+.2f}bp  "
          f"win_gross {(g.gross>0).mean():.3f}  n={len(g)}")
print("  주: straddle gross = |move| 잡되 loser stop 비용 차감. 방향 안 정해도 oracle 일부?")

# ── 시각화 ──
fig,ax=plt.subplots(2,2,figsize=(13,9))
# 1) oracle vs causal gross 분포
ax[0,0].hist(R.oracle,bins=80,alpha=.6,label='oracle |move|',color='steelblue',density=True)
ax[0,0].hist(R.net_obi,bins=80,alpha=.5,label='causal OBI net',color='orange',density=True,range=(-40,40))
ax[0,0].axvline(4,color='r',ls='--',label='fee M+M 4bp')
ax[0,0].set_title(f'oracle(11.8bp) vs causal-OBI(0.37bp) gross\n간극={R.oracle.mean()-R.net_obi.mean():.1f}bp 실현불가')
ax[0,0].legend(); ax[0,0].set_xlim(-40,50)
# 2) OBI gate gradient
obig=[R[R.obiq==q].net_obi.mean() for q in [1,2,3,4,5]]
ax[0,1].bar([1,2,3,4,5],obig,color='teal')
ax[0,1].axhline(4,color='r',ls='--',label='fee 4bp')
ax[0,1].set_title('net_obi by |OBI| quintile\n(유일하게 기울기 — 단 q5도 fee 미만)')
ax[0,1].set_xlabel('|OBI| quintile'); ax[0,1].set_ylabel('net bp'); ax[0,1].legend()
# 3) gap by year
yrs=sorted(R.yr.unique())
orc=[R[R.yr==y].oracle.mean() for y in yrs]
cau=[max(R[R.yr==y][c].mean() for c in ['net_mom5','net_rev5','net_mom15','net_obi']) for y in yrs]
x=np.arange(len(yrs))
ax[1,0].bar(x-.2,orc,.4,label='oracle',color='steelblue')
ax[1,0].bar(x+.2,cau,.4,label='best causal',color='orange')
ax[1,0].axhline(4,color='r',ls='--')
ax[1,0].set_xticks(x); ax[1,0].set_xticklabels(yrs)
ax[1,0].set_title('시기별: oracle vs best causal (간극 항상 큼)'); ax[1,0].legend()
# 4) OBI×vol heatmap
im=ax[1,1].imshow(piv.values,aspect='auto',cmap='RdYlGn',vmin=-2,vmax=2)
ax[1,1].set_xticks(range(5)); ax[1,1].set_xticklabels([1,2,3,4,5])
ax[1,1].set_yticks(range(5)); ax[1,1].set_yticklabels([1,2,3,4,5])
ax[1,1].set_xlabel('vol quintile'); ax[1,1].set_ylabel('|OBI| quintile')
ax[1,1].set_title('net_obi by |OBI|×vol (bp)\n적색=음, 녹색=양')
plt.colorbar(im,ax=ax[1,1])
for i in range(5):
    for j in range(5):
        ax[1,1].text(j,i,f'{piv.values[i,j]:.1f}',ha='center',va='center',fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/g_sibling.png',dpi=110)
print(f'\n[viz] {OUT}/g_sibling.png')

# 저장
out={'obi_gate_gradient_bp':[float(v) for v in obig],
     'obiq5_by_year':{y:float(g5[g5.yr==y].net_obi.mean()) for y in sorted(g5.yr.unique())},
     'straddle_net_by_stop':{int(s):float(g.net.mean()) for s,g in SD.groupby('STOP')},
     'gap_bp':float(R.oracle.mean()-R.net_obi.mean())}
json.dump(out,open(f'{OUT}/sibling23.json','w'),indent=2)
print(json.dumps(out,indent=2))
