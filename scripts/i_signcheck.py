import numpy as np, pandas as pd, json
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
nrm = pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
meta = json.load(open(f'{OUT}/reduce_norm_meta.json'))
reps = [r for r in meta['reps'] if r!='spread_bp']
signed = [c for c in reps if c in meta['signed']]
yr = nrm['yr'].astype(int).to_numpy()
days = sorted(nrm['day'].unique()); day_ix={d:i for i,d in enumerate(days)}
drow = nrm['day'].map(day_ix).to_numpy()
C = nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32)
sidx = [reps.index(c) for c in signed]
rng = np.random.default_rng(11)
elig = np.where((nrm['min_of_day']>=330)&(yr>=2024)&(drow>=60))[0]
qs=[]
for y,t in [(2024,120),(2025,120),(2026,60)]:
    cand=elig[yr[elig]==y]; qs+=list(rng.choice(cand,min(t,len(cand)),replace=False))
qs=np.array(sorted(qs))
agree_top=np.zeros((len(qs),len(sidx))); agree_rnd=np.zeros_like(agree_top)
for i,q in enumerate(qs):
    pool=np.where(drow<drow[q])[0]
    d2=((C[pool]-C[q])**2).sum(1)
    top=pool[np.argsort(d2)[:100]]
    rnd=rng.choice(pool,100,replace=False)
    qsig=np.sign(C[q,sidx])
    agree_top[i]=(np.sign(C[np.ix_(top,sidx)])==qsig).mean(0)
    agree_rnd[i]=(np.sign(C[np.ix_(rnd,sidx)])==qsig).mean(0)
df=pd.DataFrame({'label':signed,
  'top_med':np.median(agree_top,0),'top_p10':np.quantile(agree_top,.1,0),
  'rnd_med':np.median(agree_rnd,0)})
df['lift']=df.top_med-df.rnd_med
print(df.round(3).sort_values('lift',ascending=False).to_string(index=False))
df.to_csv(f'{OUT}/sign_agreement.csv',index=False)
print('\n전체 부호일치 (모든 signed dim 평균): top', round(float(np.median(agree_top.mean(1))),3),
      '| random', round(float(np.median(agree_rnd.mean(1))),3))
