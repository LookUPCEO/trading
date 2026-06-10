import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
H=['30m','1h','4h']; FEE=11.0
R=pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
evs=[]
for h in H:
    ok=(R[f'{h}_n']>=70)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
    s=R[ok]; st=np.maximum(s[f'{h}_fup'].to_numpy(),1-s[f'{h}_fup'].to_numpy())
    sgn=np.where(s[f'{h}_fup']>=.5,1,-1)
    evs.append(pd.DataFrame(dict(strength=st,net=sgn*s[f'{h}_frq'].to_numpy()*1e4-FEE)))
E=pd.concat(evs,ignore_index=True).sort_values('strength').reset_index(drop=True)
w=150
roll_win=E.net.gt(0).rolling(2*w,center=True,min_periods=w).mean()*100
fig,ax=plt.subplots(1,2,figsize=(13,4.5))
ax[0].plot(E.strength,roll_win,lw=1)
ax[0].axvline(0.70,color='r',ls='--',label='thr 0.70'); ax[0].axhline(50,color='k',lw=.5)
ax[0].set_xlim(0.60,0.78); ax[0].set_ylim(30,75); ax[0].legend()
ax[0].set_xlabel('strength'); ax[0].set_ylabel('rolling win% (net>0)')
ax[0].set_title('rolling 승률 vs strength — 매끄러운 점프 아님 (노이즈)')
# 0.6975 부근 fine bins win + Wilson
edges=np.arange(0.66,0.7325,0.005); cen=[]; win=[]; err=[]; ns=[]
for i in range(len(edges)-1):
    s=E[(E.strength>=edges[i])&(E.strength<edges[i+1])]
    if len(s)<8: continue
    k=(s.net>0).sum(); p=k/len(s); cen.append((edges[i]+edges[i+1])/2); win.append(p*100)
    err.append(1.96*np.sqrt(p*(1-p)/len(s))*100); ns.append(len(s))
ax[1].errorbar(cen,win,yerr=err,fmt='o-',capsize=3)
ax[1].axvline(0.70,color='r',ls='--'); ax[1].axhline(50,color='k',lw=.5)
ax[1].set_xlabel('strength bin'); ax[1].set_ylabel('win% ±95%'); ax[1].set_title('fine bins (n 작음 — CI 큼)')
for x,y,nn in zip(cen,win,ns): ax[1].annotate(str(nn),(x,y),fontsize=6)
plt.tight_layout(); plt.savefig(f'{OUT}/thr_fine.png',dpi=110); print('saved')
