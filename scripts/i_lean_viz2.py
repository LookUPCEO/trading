import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
R=pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
fig,axes=plt.subplots(1,3,figsize=(18,4.6))
days=sorted(R.qday.unique())
for h,col in [('30m','tab:green'),('1h','tab:red'),('4h','tab:purple')]:
    ok=(R[f'{h}_n']>=70)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
    s=R[ok]; lean=(s[f'{h}_fup']>=.7)|(s[f'{h}_fup']<=.3); L=s[lean].sort_values('q')
    sgn=np.where(L[f'{h}_fup']>=.5,1.,-1.)
    ret=sgn*L[f'{h}_frq'].to_numpy()*1e4
    axes[0].plot(L.qday.to_numpy(),np.cumsum(ret-11),label=f'{h} (n={len(L)})',color=col)
    axes[1].scatter(L.qday,ret,s=8,alpha=.5,color=col,label=h)
axes[0].axhline(0,color='k',lw=.5); axes[0].legend(); axes[0].set_title('누적 net bp (T+T 11bp 차감, thr70) — qday 순')
axes[0].set_xlabel('day index (2024~2026)')
axes[1].axhline(0,color='k',lw=.5); axes[1].legend(); axes[1].set_title('이벤트별 signed gross bp')
h='1h'
ok=(R[f'{h}_n']>=70)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
s=R[ok]; lean=(s[f'{h}_fup']>=.7)|(s[f'{h}_fup']<=.3); L=s[lean]
sgn=np.where(L[f'{h}_fup']>=.5,1.,-1.)
ret=sgn*L[f'{h}_frq'].to_numpy()*1e4
axes[2].hist(ret,bins=40)
axes[2].axvline(0,color='k',lw=1); axes[2].axvline(np.median(ret),color='r',ls='--',label=f'med {np.median(ret):+.0f}')
axes[2].set_title('1h thr70 signed gross 분포'); axes[2].legend()
plt.tight_layout(); plt.savefig(f'{OUT}/lean70_v2_net.png',dpi=110)
print('saved')
