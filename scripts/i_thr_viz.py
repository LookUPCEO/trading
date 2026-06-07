import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
T=pd.read_csv(f'{OUT}/thr_curve.csv')
fig,axes=plt.subplots(1,3,figsize=(16,4.4))
for h,c in [('30m','tab:green'),('1h','tab:red'),('4h','tab:purple'),('COMB','k')]:
    s=T[T.h==h]
    axes[0].plot(s.thr,s.net_tr,'o-',color=c,label=f'{h} train',alpha=.8)
    axes[0].plot(s.thr,s.net_te,'s--',color=c,alpha=.4,label=f'{h} test')
    axes[1].plot(s.thr,s.daily_tr,'o-',color=c,label=h,alpha=.8)
    axes[1].plot(s.thr,s.daily_te,'s--',color=c,alpha=.4)
    axes[2].semilogy(s.thr,s.n_tr+s.n_te,'o-',color=c,label=h)
axes[0].axhline(0,color='r',lw=.6); axes[0].set_title('net/trade (bp) vs thr — solid train, dashed test'); axes[0].legend(fontsize=6,ncol=2)
axes[1].axhline(0,color='r',lw=.6); axes[1].set_ylim(-60,25); axes[1].set_title('daily yield (bp/day) vs thr'); axes[1].legend(fontsize=7)
axes[2].set_title('event count (log)'); axes[2].legend(fontsize=7)
for ax in axes: ax.set_xlabel('thr')
plt.tight_layout(); plt.savefig(f'{OUT}/thr_curve.png',dpi=110); print('saved')
