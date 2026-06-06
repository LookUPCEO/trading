import numpy as np, pandas as pd, json
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
nrm = pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
meta = json.load(open(f'{OUT}/reduce_norm_meta.json'))
reps=[r for r in meta['reps'] if r!='spread_bp']
yr=nrm['yr'].astype(int).to_numpy()
days=sorted(nrm['day'].unique()); day_ix={d:i for i,d in enumerate(days)}
drow=nrm['day'].map(day_ix).to_numpy()
dayord=pd.to_datetime(nrm['day']).map(pd.Timestamp.toordinal).to_numpy()
C=nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32)
m23=yr==2023; mu=C[m23].mean(0); S=np.cov((C[m23]-mu).T)
w,V=np.linalg.eigh(S); W=V@np.diag(1/np.sqrt(np.maximum(w,1e-6)))@V.T
Cw=((C-mu)@W).astype(np.float32)
rng=np.random.default_rng(11)
elig=np.where((nrm['min_of_day']>=330)&(yr>=2024)&(drow>=60))[0]
qs=[]
for y,t in [(2024,120),(2025,120),(2026,60)]:
    cand=elig[yr[elig]==y]; qs+=list(rng.choice(cand,min(t,len(cand)),replace=False))
qs=np.array(sorted(qs))
res={}
for nameM,M in [('C',C),('C_wh',Cw)]:
    f7=[];f30=[];f90=[]
    for q in qs:
        pool=np.where(drow<drow[q])[0]
        d2=((M[pool]-M[q])**2).sum(1)
        top=pool[np.argpartition(d2,100)[:100]]
        dd=dayord[q]-dayord[top]
        pdd=dayord[q]-dayord[pool]
        f7.append(((dd<=7).mean(), (pdd<=7).mean()))
        f30.append(((dd<=30).mean(),(pdd<=30).mean()))
        f90.append(((dd<=90).mean(),(pdd<=90).mean()))
    for nm,arr in [('7d',f7),('30d',f30),('90d',f90)]:
        a=np.array(arr)
        res[(nameM,nm)]=(np.median(a[:,0]), np.median(a[:,1]), np.median(a[:,0]/np.maximum(a[:,1],1e-9)))
for k,v in res.items():
    print(f"{k[0]:5s} ≤{k[1]:>3s}: top100 비중 med {v[0]:.3f} | pool 비중 {v[1]:.3f} | lift {v[2]:.2f}")
