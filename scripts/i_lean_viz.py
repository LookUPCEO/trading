import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
R=pd.read_parquet(f'{OUT}/lean70_per_query.parquet')
H=['5m','10m','30m','1h','4h']
fig,axes=plt.subplots(1,2,figsize=(15,4.5))
qtrs=sorted(R.quarter.unique())
for h in H:
    ok=R[f'{h}_n']>=70
    y=[]
    for qt in qtrs:
        s=R[ok&(R.quarter==qt)]
        f=s[f'{h}_fup']
        y.append(((f>=.7)|(f<=.3)).mean()*100 if len(s)>200 else np.nan)
    axes[0].plot(range(len(qtrs)),y,marker='o',label=h)
ok=R['5m_n']>=70
null_rate=[]
for qt in qtrs:
    s=R[ok&(R.quarter==qt)]
    f=s['5m_rnd_fup']
    null_rate.append(((f>=.7)|(f<=.3)).mean()*100 if len(s)>200 else np.nan)
axes[0].plot(range(len(qtrs)),null_rate,'k--',label='null(random) 5m')
axes[0].set_xticks(range(len(qtrs))); axes[0].set_xticklabels(qtrs,rotation=45,fontsize=8)
axes[0].set_ylabel('thr70 lean rate (%)'); axes[0].legend(fontsize=8)
axes[0].set_title('70% lean rate by quarter (decay) vs null')
bins=np.linspace(0,1,51)
axes[1].hist(R['5m_fup'].dropna(),bins=bins,density=True,alpha=.55,label='real matches (5m)')
axes[1].hist(R['5m_rnd_fup'].dropna(),bins=bins,density=True,alpha=.55,label='null random')
for x in (.3,.7): axes[1].axvline(x,color='r',ls='--',lw=1)
axes[1].set_xlabel('frac_up of 100 independent matches'); axes[1].legend(fontsize=8)
axes[1].set_title('vote distribution: real fat tails vs null binomial')
plt.tight_layout(); plt.savefig(f'{OUT}/lean70_overview.png',dpi=110)
print('saved')
