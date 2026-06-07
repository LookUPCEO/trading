#!/usr/bin/env python3
"""[I] 5단계 작업6: why — net(h) 를 hit-edge × |move| − fee 로 분해."""
import numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
R=pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
Rh=pd.read_parquet(f'{OUT}/lean70_v2_per_query_hfine.parquet')
rows=[]
for src,h in [(R,'5m'),(R,'10m'),(Rh,'15m'),(Rh,'20m'),(R,'30m'),(Rh,'45m'),(R,'1h'),(Rh,'2h'),(Rh,'3h'),(R,'4h'),(Rh,'6h')]:
    ok=(src[f'{h}_n']>=70)&~src[f'{h}_frq'].isna()&(src[f'{h}_frq']!=0)
    s=src[ok]
    allmove=np.abs(s[f'{h}_frq'])*1e4   # 전체 쿼리의 |move| (시장 스케일)
    lean=(s[f'{h}_fup']>=.7)|(s[f'{h}_fup']<=.3)
    L=s[lean]
    if len(L)<15: continue
    sgn=np.where(L[f'{h}_fup']>=.5,1.,-1.)
    ret=sgn*L[f'{h}_frq'].to_numpy()*1e4
    hit=(ret>0).mean(); edge=2*hit-1
    mv=np.abs(ret)
    # 근사: gross ≈ edge × E|move| (win/loss 같은 크기 가정 시) — 실제와 대조
    rows.append(dict(h=h, n=len(L), hit=round(hit,3), edge=round(edge,3),
                     mkt_move=round(allmove.median(),1), lean_move=round(np.median(mv),1),
                     approx=round(edge*np.median(mv),1), gross=round(ret.mean(),1),
                     fee_pct_of_move=round(11/np.median(mv)*100)))
T=pd.DataFrame(rows)
print(T.to_string(index=False))
print()
print("해석: fee 11bp 가 median |move| 의 몇 %인가 → 5m 은 fee 가 이동폭 초과 (구조적 불가)")
T.to_csv(f'{OUT}/why_horizon_decomp.csv',index=False)
# viz
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
mins=[5,10,15,20,30,45,60,120,180,240,360]
fig,axes=plt.subplots(1,3,figsize=(17,4.3))
axes[0].semilogx(mins[:len(T)],T.hit,'o-'); axes[0].axhline(.5,color='k',lw=.5)
axes[0].set_title('hit rate vs horizon (thr70)'); axes[0].set_xlabel('min')
axes[1].semilogx(mins[:len(T)],T.gross,'o-',label='gross')
axes[1].semilogx(mins[:len(T)],T.gross-11,'s--',label='net (T+T)')
axes[1].axhline(0,color='k',lw=.5); axes[1].legend(); axes[1].set_title('gross/net vs horizon')
axes[2].semilogx(mins[:len(T)],T.lean_move,'o-',label='median |move| (lean)')
axes[2].axhline(11,color='r',ls='--',label='fee 11bp')
axes[2].legend(); axes[2].set_title('|move| scale vs fee')
for ax in axes: ax.set_xticks([5,15,30,60,120,240,360]); ax.set_xticklabels(['5m','15m','30m','1h','2h','4h','6h'])
plt.tight_layout(); plt.savefig(f'{OUT}/why_horizon.png',dpi=110)
print('saved png')
