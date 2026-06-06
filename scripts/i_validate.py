#!/usr/bin/env python3
"""
[I] 1단계 검증: 라벨이 제대로 계산·분포 정상인지만 (거래/예측 X).
  작업2 분포/정상성 + 시각화(지표 의도대로 찍히나)
  작업3 라벨 간 중복/독립 (상관)
  작업4 시기별 분포 차이 (정규화 필요 여부)
출력: CSV 표 + PNG (사람이 직접 볼 수 있게).
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D='/Users/mark/Desktop/Mark/mark19/research/i_labeling'
R=pd.read_parquet(f'{D}/labels.parquet')
META=['yr','day','sec','min_of_day','mid']
LABS=[c for c in R.columns if c not in META]
BOUNDED={'rsi_14':(0,100),'rsi_30':(0,100),'stoch_k':(0,100),'stoch_d':(0,100),'adx_14':(0,100),
 'obi1':(-1,1),'obi5':(-1,1),'obi20':(-1,1),'obi50':(-1,1),'obi_wtd':(-1,1),
 'flow_30':(-1,1),'flow_1m':(-1,1),'flow_5m':(-1,1),'body_ratio':(-1,1),
 'upper_wick':(0,1),'lower_wick':(0,1)}
print(f'[load] {len(R)} rows, {R.day.nunique()} days, {len(LABS)} labels, {R.yr.min()}~{R.yr.max()}')

# ===== 작업2: 분포/정상성 =====
rows=[]
for c in LABS:
    s=R[c].replace([np.inf,-np.inf],np.nan)
    d=s.dropna()
    viol=''
    if c in BOUNDED:
        lo,hi=BOUNDED[c]
        nv=((d<lo-1e-6)|(d>hi+1e-6)).sum()
        viol=f'{nv} out-of-[{lo},{hi}]' if nv else 'ok'
    rows.append(dict(label=c,nan_pct=round(s.isna().mean()*100,2),
        std=d.std(), mean=d.mean(),
        p1=d.quantile(.01),p10=d.quantile(.10),p50=d.quantile(.50),
        p90=d.quantile(.90),p99=d.quantile(.99),min=d.min(),max=d.max(),
        constant=(d.std()==0),range_check=viol))
dist=pd.DataFrame(rows)
dist.to_csv(f'{D}/labels_distribution.csv',index=False)
print('\n=== 작업2 분포/정상성 (요약) ===')
print(dist[['label','nan_pct','mean','std','p1','p99','range_check']].to_string(index=False))
bad=dist[(dist.nan_pct>5)|(dist.constant)|(dist.range_check.str.contains('out',na=False))]
print('\n⚠️ 주의 라벨:', bad.label.tolist() if len(bad) else '없음 (전부 정상범위)')

# 히스토그램 그리드
nl=len(LABS); ncol=6; nrow=(nl+ncol-1)//ncol
fig,ax=plt.subplots(nrow,ncol,figsize=(ncol*3,nrow*2.2))
for i,c in enumerate(LABS):
    a=ax.flat[i]; d=R[c].replace([np.inf,-np.inf],np.nan).dropna()
    a.hist(d,bins=60,color='steelblue'); a.set_title(c,fontsize=8)
    a.tick_params(labelsize=6)
for j in range(nl,nrow*ncol): ax.flat[j].axis('off')
plt.tight_layout(); plt.savefig(f'{D}/hist_grid.png',dpi=85); plt.close()
print('saved hist_grid.png')

# ===== 작업2 시각화: 지표가 의도대로 찍히나 (sample day overlay) =====
import glob
for vf in sorted(glob.glob(f'{D}/vizday_*.parquet')):
    day=vf.split('vizday_')[1][:-8]
    v=pd.read_parquet(vf)
    if len(v)<100: continue
    x=v.min_of_day.values
    fig,ax=plt.subplots(5,1,figsize=(14,13),sharex=True)
    # 1 price + MA + Bollinger
    ax[0].plot(x,v.mid,'k',lw=.8,label='mid')
    for k,col in [(30,'tab:blue'),(120,'tab:orange')]:
        ax[0].plot(x,v.mid/(1+v[f'ma_dev_{k}']),col,lw=.7,alpha=.8,label=f'MA{k}')
    bm=v.mid/(1+0); # boll band 재구성: mid - boll_pos*std, std=boll_width*ma20
    ma20=v.mid/(1+v['ma_dev_15'])  # 근사용 표시 생략
    ax[0].set_title(f'{day}  price + MA30/120 (지표가 가격 따라가나)'); ax[0].legend(fontsize=7)
    # 2 RSI + Stoch
    ax[1].plot(x,v.rsi_14,'purple',lw=.7,label='RSI14'); ax[1].plot(x,v.stoch_k,'green',lw=.5,alpha=.6,label='StochK')
    ax[1].axhline(70,color='r',ls=':',lw=.5); ax[1].axhline(30,color='b',ls=':',lw=.5)
    ax[1].set_ylim(0,100); ax[1].legend(fontsize=7); ax[1].set_ylabel('RSI/Stoch')
    # 3 MACD hist + ADX
    ax[2].bar(x,v.macd_hist,color='gray',width=1.0,label='MACD hist(bp)')
    a2=ax[2].twinx(); a2.plot(x,v.adx_14,'brown',lw=.7,label='ADX'); a2.set_ylabel('ADX')
    ax[2].legend(fontsize=7,loc='upper left'); a2.legend(fontsize=7,loc='upper right'); ax[2].set_ylabel('MACD hist')
    # 4 OBI + flow
    ax[3].plot(x,v.obi5,'tab:red',lw=.5,label='OBI5'); ax[3].plot(x,v.flow_5m,'tab:blue',lw=.5,alpha=.6,label='flow5m')
    ax[3].axhline(0,color='k',lw=.3); ax[3].set_ylim(-1,1); ax[3].legend(fontsize=7); ax[3].set_ylabel('OBI/flow')
    # 5 rv + boll_width
    ax[4].plot(x,v.rv_300,'tab:green',lw=.6,label='rv_300(bp)'); ax[4].plot(x,v.rv_3600,'tab:orange',lw=.6,label='rv_3600')
    ax[4].legend(fontsize=7); ax[4].set_ylabel('rv(bp)'); ax[4].set_xlabel('minute of day')
    plt.tight_layout(); plt.savefig(f'{D}/viz_{day}.png',dpi=90); plt.close()
    print(f'saved viz_{day}.png')

# ===== 작업3: 중복/독립 (Spearman, 비선형/스케일 robust) =====
X=R[LABS].replace([np.inf,-np.inf],np.nan)
corr=X.corr(method='spearman')
corr.to_csv(f'{D}/corr_spearman.csv')
fig,ax=plt.subplots(figsize=(16,14))
im=ax.imshow(corr.values,cmap='RdBu_r',vmin=-1,vmax=1)
ax.set_xticks(range(len(LABS))); ax.set_xticklabels(LABS,rotation=90,fontsize=6)
ax.set_yticks(range(len(LABS))); ax.set_yticklabels(LABS,fontsize=6)
plt.colorbar(im,fraction=0.04); ax.set_title('Spearman corr (|r|>0.9 = 중복 의심)')
plt.tight_layout(); plt.savefig(f'{D}/corr_heatmap.png',dpi=95); plt.close()
print('saved corr_heatmap.png')
# 중복쌍
pairs=[]
for i in range(len(LABS)):
    for j in range(i+1,len(LABS)):
        r=corr.iloc[i,j]
        if abs(r)>=0.9: pairs.append((LABS[i],LABS[j],round(r,3)))
pairs.sort(key=lambda x:-abs(x[2]))
pd.DataFrame(pairs,columns=['a','b','spearman']).to_csv(f'{D}/redundant_pairs.csv',index=False)
print('\n=== 작업3 중복쌍 |r|>=0.9 ===')
for a,b,r in pairs: print(f'  {a:12s} ~ {b:12s} {r:+.3f}')
# 독립 그룹: |corr| 거리로 계층군집
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
dmat=1-corr.abs().values; np.fill_diagonal(dmat,0); dmat=(dmat+dmat.T)/2
Z=linkage(squareform(dmat,checks=False),method='average')
cl=fcluster(Z,t=0.2,criterion='distance')  # 거리<0.2 (|corr|>0.8) 묶음
grp={}
for lab,c in zip(LABS,cl): grp.setdefault(c,[]).append(lab)
print(f'\n=== 작업3 독립 그룹 ({len(grp)}개; |corr|>0.8 끼리 묶음) ===')
for c,g in sorted(grp.items(),key=lambda x:-len(x[1])):
    tag='[중복군]' if len(g)>1 else ''
    print(f'  G{c}{tag}: {g}')
json.dump({str(c):g for c,g in grp.items()},open(f'{D}/independent_groups.json','w'),ensure_ascii=False,indent=1)

# ===== 작업4: 시기별 분포 차이 (정규화 필요?) =====
yrs=sorted(R.yr.unique())
print(f'\n=== 작업4 시기별(연도) 분포 — 정규화 필요 판단 ===')
tstats=[]
for c in LABS:
    by=R.groupby('yr')[c].agg(['median',lambda s:s.quantile(.1),lambda s:s.quantile(.9),'std'])
    by.columns=['median','p10','p90','std']
    by['iqr']=by.p90-by.p10
    # 스케일 드리프트: 연도별 중앙값 변동폭 / 전체 IQR, 연도별 IQR 비율
    med_drift=(by['median'].max()-by['median'].min())
    overall_iqr=R[c].quantile(.9)-R[c].quantile(.1)+1e-12
    iqr_ratio=by['iqr'].max()/(by['iqr'].min()+1e-12)
    drift_norm=med_drift/overall_iqr
    tstats.append(dict(label=c,median_drift_norm=round(drift_norm,3),iqr_max_min_ratio=round(iqr_ratio,2),
        **{f'med_{y}':round(by.loc[y,'median'],4) for y in yrs if y in by.index}))
T=pd.DataFrame(tstats)
T.to_csv(f'{D}/temporal_stats.csv',index=False)
need=T[(T.median_drift_norm>0.5)|(T.iqr_max_min_ratio>2.0)].sort_values('iqr_max_min_ratio',ascending=False)
print('정규화 필요 의심 (시기간 중앙값 이동>0.5 IQR 또는 IQR비>2):')
print(need[['label','median_drift_norm','iqr_max_min_ratio']].to_string(index=False) if len(need) else '  없음')
print('\n안정 라벨 (시기 무관):', [l for l in LABS if l not in need.label.values][:20],'...')

# 시기 분포 시각화 (대표 몇개: rv, spread, obi5, rsi, boll_width, vol_z)
viz_lab=['rv_300','rv_3600','spread_bp','atr_14','boll_width','vol_z','obi5','rsi_14','ma_dev_120','range_bp']
fig,ax=plt.subplots(2,5,figsize=(20,7))
for i,c in enumerate(viz_lab):
    a=ax.flat[i]
    data=[R[R.yr==y][c].replace([np.inf,-np.inf],np.nan).dropna().values for y in yrs]
    a.boxplot(data,labels=yrs,showfliers=False)
    a.set_title(c,fontsize=10); a.tick_params(labelsize=7)
plt.suptitle('작업4: 시기별 분포 (박스가 연도마다 이동/스케일변하면 정규화 필요)')
plt.tight_layout(); plt.savefig(f'{D}/temporal_boxplots.png',dpi=90); plt.close()
print('saved temporal_boxplots.png')
print('\n[validate done] 산출물: labels_distribution.csv, hist_grid.png, viz_*.png, corr_heatmap.png, redundant_pairs.csv, independent_groups.json, temporal_stats.csv, temporal_boxplots.png')
