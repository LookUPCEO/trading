#!/usr/bin/env python3
"""[I] 27단계 — 하락 전용 시스템 (처음부터 독립 최적화, 상승 거울 X).
본인 지적: 26단계는 down 을 '4h 상승용 21차원/thr0.70' 의 거울로만 봄. 파동이면
하락도 예측 가능해야 자연. → 하락 최적 horizon(긴 쪽 6h~2d 포함)/표현/임계 자유 탐색.
방법: 21차원 whiten(2023) kNN, pool=prefix(구조적 OOS). 미래수익 = cross-day 글로벌
  가격 인덱스(긴 horizon 가능, lookahead 0). 이웃 day-dedupe top-N → down 합의 fup.
  horizon {30m,1h,2h,4h,6h,12h,1d,2d}. down-lean = fup<=thr. 시간순(causal) 독립.
  ⚠️ 1d+ 자본중첩(20단계) — 독립=비겹침 강제. multiple testing → Bonferroni.
라이브 검증만(4h 유지). 흥분 X."""
import numpy as np, pandas as pd, json, os
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB='/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
K_CAND=1500; N_NB=100; EXCL_DAYS=3; MINV=70; Q_STRIDE=10
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
HZ={'30m':30,'1h':60,'2h':120,'4h':240,'6h':360,'12h':720,'1d':1440,'2d':2880}

def build():
    nrm=pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    meta=json.load(open(f'{OUT}/reduce_norm_meta.json'));reps=[r for r in meta['reps'] if r!='spread_bp']
    yr=nrm['yr'].astype(int).to_numpy();mod=nrm['min_of_day'].to_numpy()
    days=sorted(nrm['day'].unique());dix={d:i for i,d in enumerate(days)}
    drow=nrm['day'].map(dix).to_numpy();n=len(nrm)
    month=nrm['day'].str[5:7].astype(int).to_numpy()
    starts=np.searchsorted(drow,np.arange(len(days)))
    # cross-day 글로벌 가격 인덱스
    full=pd.read_parquet(LAB,columns=['day','min_of_day','mid']);full=full[full.day.isin(days)]
    fdays=sorted(full['day'].unique());dixf={d:i for i,d in enumerate(fdays)}
    Pf=np.full(len(fdays)*1440,np.nan,np.float32)
    Pf[full['day'].map(dixf).to_numpy()*1440+full['min_of_day'].to_numpy()]=full['mid'].to_numpy(np.float32)
    NG=len(Pf);gf=nrm['day'].map(dixf).to_numpy()*1440+mod
    FR={}
    for hn,H in HZ.items():
        fr=np.full(n,np.nan,np.float32);tgt=gf+H;ok=tgt<NG
        fr[ok]=Pf[tgt[ok]]/Pf[gf[ok]]-1;FR[hn]=fr
    C=nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32);m23=yr==2023
    mu=C[m23].mean(0);S=np.cov((C[m23]-mu).T);w,V=np.linalg.eigh(S)
    W=(V@np.diag(1/np.sqrt(np.maximum(w,1e-6)))@V.T).astype(np.float32)
    X=((C-mu)@W).astype(np.float32);xsq=(X*X).sum(1)
    qtr=np.array([f"{yr[i]}Q{(month[i]-1)//3+1}" for i in range(n)])
    return dict(nrm=nrm,X=X,xsq=xsq,drow=drow,mod=mod,yr=yr,qtr=qtr,starts=starts,FR=FR,n=n,reps=reps,
                C=C,mu=mu,W=W)

def search(D, qs, FRsel=None):
    """qs 쿼리들에 대해 horizon별 이웃 down 합의 fup + 쿼리 자기 미래 반환."""
    X,xsq,drow,mod,starts,FR=D['X'],D['xsq'],D['drow'],D['mod'],D['starts'],(FRsel or D['FR'])
    recs=[];BLK=128
    from time import time as _t;t0=_t()
    for bi in range(0,len(qs),BLK):
        qb=qs[bi:bi+BLK]
        ends=starts[np.maximum(drow[qb]-EXCL_DAYS,0)];emax=int(ends.max())
        if emax<50000: continue
        d2=xsq[None,:emax]-2.0*(X[qb]@X[:emax].T)
        for j,q in enumerate(qb):
            e=int(ends[j])
            if e<50000: continue
            row=d2[j,:e];kc=min(K_CAND,e-1)
            cand=np.argpartition(row,kc)[:kc];order=cand[np.argsort(row[cand])]
            # day-dedupe top-N 이웃 (한 day 1표 = 미래 독립)
            seen=set();picks=[]
            for i in order:
                dd=drow[i]
                if dd in seen: continue
                seen.add(dd);picks.append(i)
                if len(picks)>=N_NB: break
            picks=np.array(picks)
            rec=dict(q=int(q),qday=int(drow[q]),mod=int(mod[q]),qtr=D['qtr'][q])
            for hn in HZ:
                v=FR[hn][picks];v=v[~np.isnan(v)];v=v[v!=0]
                rec[f'{hn}_n']=len(v)
                rec[f'{hn}_fup']=(v>0).mean() if len(v) else np.nan
                rec[f'{hn}_frq']=float(D['FR'][hn][q])
            recs.append(rec)
        if bi%(BLK*40)==0: print(f"  q {bi}/{len(qs)} {_t()-t0:.0f}s",flush=True)
    return pd.DataFrame(recs)

def greedy_causal(mod,qday,h):
    o=np.lexsort((mod,qday));acc={};keep=np.zeros(len(mod),bool)
    for idx in o:
        d=qday[idx];m=mod[idx];lst=acc.get(d)
        if lst is None: acc[d]=[m];keep[idx]=True;continue
        if all(abs(m-mm)>=min(h,1440) for mm in lst): lst.append(m);keep[idx]=True
    return keep

def dayci(qd,net,seed=7):
    if len(net)<5: return np.nan,np.nan,np.nan
    dm=pd.Series(net).groupby(qd).mean().to_numpy()
    bs=np.random.default_rng(seed).choice(dm,(4000,len(dm)),replace=True).mean(axis=1)
    return dm.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)

def analyze(R, FEE=11.0):
    print(f"\n[analyze] {len(R)} 쿼리 (stride{Q_STRIDE})")
    print("\n===== 작업1+3: 하락 전용 horizon×임계 (cross-day, causal 독립) =====")
    print(f"{'hz':>4} {'thr':>5} | {'allN':>5} {'indN':>5} {'hit':>5} {'gross':>6} {'net':>6} | {'teN':>4} {'teHit':>5} dayCI(test)")
    cells=[]
    for hn in HZ:
        for thr in [.60,.65,.70]:
            ok=(R[f'{hn}_n']>=MINV)&~R[f'{hn}_frq'].isna()&(R[f'{hn}_frq']!=0)
            s=R[ok];L=s[s[f'{hn}_fup']<=1-thr].copy()
            if len(L)<5: continue
            frq=L[f'{hn}_frq'].to_numpy();L=L.assign(net=(-1)*frq*1e4-FEE,hit=(frq<0))
            keep=greedy_causal(L['mod'].to_numpy(),L.qday.to_numpy(),HZ[hn])
            Li=L[keep];te=~Li.qtr.isin(TRAIN_Q);Lte=Li[te]
            dm,lo,hi=dayci(Lte.qday.to_numpy(),Lte.net.to_numpy())
            g=(Li.net+FEE).mean()
            cells.append((hn,thr,len(L),len(Li),Li.hit.mean(),g,Li.net.mean(),len(Lte),
                          Lte.hit.mean() if len(Lte) else np.nan,dm,lo,hi))
            print(f"{hn:>4} ≤{1-thr:.2f} | {len(L):>5} {len(Li):>5} {Li.hit.mean():>5.3f} {g:>+6.1f} {Li.net.mean():>+6.1f} | "
                  f"{len(Lte):>4} {Lte.hit.mean() if len(Lte) else 0:>5.2f} [{lo:+.0f},{hi:+.0f}]")
    C=pd.DataFrame(cells,columns=['hz','thr','allN','indN','hit','gross','net','teN','teHit','dm','lo','hi'])
    print("\n  하락 hit>0.5 & gross>0 인 horizon (있긴 한가):")
    pos=C[(C.hit>0.5)&(C.gross>0)]
    print("   "+(", ".join(f"{r.hz}≤{1-r.thr:.2f}(hit{r.hit:.2f},g{r.gross:+.0f},te{r.teHit:.2f})" for _,r in pos.iterrows()) if len(pos) else "없음"))
    nb=len(C)
    surv=C[(C.gross>FEE)&(C.lo>0)&(C.teN>=5)]
    print(f"\n  Bonferroni 분모 {nb}. gross>fee & test dayCI>0 & teN>=5 생존: {len(surv)}")
    if len(surv): print(surv.to_string(index=False))
    C.to_csv(f'{OUT}/downsys_cells.csv',index=False)
    return C

def main():
    D=build()
    yr=D['yr'];mod=D['mod']
    qs=np.where((yr>=2024)&(mod%Q_STRIDE==5))[0]
    print(f"[setup] DB {D['n']} rows, queries {len(qs)} (stride{Q_STRIDE})",flush=True)
    R=search(D,qs)
    R.to_parquet(f'{OUT}/downsys_per_query.parquet')
    analyze(R)
    print("\n현행 4h 상승 hit 0.68, 0.084%/day. 하락 전용이 OOS+fee 넘으면 후보(라이브 즉시교체 X).")

if __name__=='__main__':
    if os.environ.get('ANALYZE_ONLY'):
        analyze(pd.read_parquet(f'{OUT}/downsys_per_query.parquet'))
    else: main()
