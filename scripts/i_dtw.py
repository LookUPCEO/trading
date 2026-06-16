#!/usr/bin/env python3
"""[I] 21단계 — DTW 파동 모양 유사도. 과거 60분 정규화 log-return 경로, Sakoe-Chiba band DTW.
비용: 21차원 유클리드 top-300 후보 → 그 안 DTW 재순위(근사). 방향 정렬/redundancy/hit 4h."""
import numpy as np, pandas as pd, json
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB='/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
NWIN=60; BAND=8; K_EUC=300; N_IND=100; EXCL_DAYS=3; MIN_VOTES=70; FEE=11.0; THR=0.70
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']

def batch_dtw(q, Cand, band=BAND):
    """q: (N,), Cand: (M,N). Sakoe-Chiba band DTW 거리 (M,). 배치(M축 벡터화)."""
    M,N=Cand.shape
    INF=1e18
    prev=np.full(M, INF); prev0=np.zeros(M)   # DP 행
    # D[i,j]; band: |i-j|<=band. 행 i 순회, 열 j in [i-band,i+band]
    dp_prev=np.full((M,N), INF)   # 이전 행
    # i=0
    cur=np.full((M,N), INF)
    for j in range(0, min(band,N-1)+1):
        c=(q[0]-Cand[:,j])**2
        cur[:,j]=c if j==0 else cur[:,j-1]+c
    dp_prev=cur
    for i in range(1,N):
        cur=np.full((M,N), INF)
        j0=max(0,i-band); j1=min(N-1,i+band)
        for j in range(j0,j1+1):
            c=(q[i]-Cand[:,j])**2
            best=dp_prev[:,j].copy()   # (i-1,j)
            if j>0:
                best=np.minimum(best, cur[:,j-1])        # (i,j-1)
                best=np.minimum(best, dp_prev[:,j-1])    # (i-1,j-1)
            cur[:,j]=c+best
        dp_prev=cur
    return dp_prev[:,N-1]

def main():
    nrm=pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    meta=json.load(open(f'{OUT}/reduce_norm_meta.json')); reps=[r for r in meta['reps'] if r!='spread_bp']
    yr=nrm['yr'].astype(int).to_numpy(); mod=nrm['min_of_day'].to_numpy()
    days=sorted(nrm['day'].unique()); dix={d:i for i,d in enumerate(days)}
    drow=nrm['day'].map(dix).to_numpy(); n=len(nrm); starts=np.searchsorted(drow,np.arange(len(days)))
    # 가격 경로 (과거 60분 log-return, 정규화) — labels mid grid
    lab=pd.read_parquet(LAB,columns=['day','min_of_day','mid']); lab=lab[lab.day.isin(days)]
    mid=np.full((len(days),1440),np.nan,np.float32); mid[lab['day'].map(dix).to_numpy(),lab['min_of_day'].to_numpy()]=lab['mid'].to_numpy(np.float32)
    logm=np.log(mid)
    # path[row] = 과거 60분 log-return (60,), vol 정규화 (크기제거)
    PATH=np.full((n,NWIN),np.nan,np.float32)
    for r in range(n):
        d=drow[r]; m=mod[r]
        if m-NWIN<0: continue
        seg=logm[d,m-NWIN:m+1]
        if np.isnan(seg).any(): continue
        dif=np.diff(seg); s=dif.std()+1e-9
        PATH[r]=(dif/s).astype(np.float32)
    valid=~np.isnan(PATH).any(1)
    # 21차원 whitened (후보 선정용)
    C=nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32); m23=yr==2023
    mu=C[m23].mean(0); S=np.cov((C[m23]-mu).T); w,V=np.linalg.eigh(S)
    W=(V@np.diag(1/np.sqrt(np.maximum(w,1e-6)))@V.T).astype(np.float32)
    X=((C-mu)@W).astype(np.float32); xsq=(X*X).sum(1)
    fr=np.full(n,np.nan,np.float32); ok=mod+240<=1439; fr[ok]=mid[drow[ok],mod[ok]+240]/mid[drow[ok],mod[ok]]-1
    qtr=np.char.add(yr.astype(str),np.char.add('Q',(((nrm['day'].str[5:7].astype(int)-1)//3+1)).astype(str).to_numpy()))
    qs=np.where((yr>=2024)&(mod%10==5)&valid)[0]
    qmax=int(__import__('os').environ.get('QMAX','0'))
    if qmax: qs=qs[::max(1,len(qs)//qmax)]
    nd=len(set(drow[qs])); ndte=len(set(drow[qs][~np.isin(qtr[qs],TRAIN_Q)]))
    print(f"[setup] valid path {valid.mean():.2f}, queries {len(qs)}",flush=True)

    def dedupe_day(idxs, nt):
        seen=set(); out=[]
        for i in idxs:
            d=drow[i]
            if d in seen: continue
            seen.add(d); out.append(i)
            if len(out)>=nt: break
        return out

    rec_dtw=[]; rec_euc=[]; overlaps=[]; aligns=[]
    from time import time as _t; t0=_t()
    for qi,q in enumerate(qs):
        e=starts[max(drow[q]-EXCL_DAYS,0)]
        if e<50000: continue
        # 21차원 유클리드 top-K_EUC 후보 (valid path 만)
        d2=xsq[:e]-2.0*(X[:e]@X[q]);
        vv=valid[:e]
        cand=np.argpartition(np.where(vv,d2,1e18), K_EUC)[:K_EUC]
        cand=cand[vv[cand]]
        # 유클리드 top100 (day-dedupe) — 비교군
        euc_order=cand[np.argsort(d2[cand])]
        euc100=dedupe_day(euc_order,100)
        # DTW 재순위 (후보 내)
        dd=batch_dtw(PATH[q], PATH[cand])
        dtw_order=cand[np.argsort(dd)]
        dtw100=dedupe_day(dtw_order,100)
        overlaps.append(len(set(euc100)&set(dtw100))/max(len(euc100),1))
        # vote
        for picks,store in [(euc100,rec_euc),(dtw100,rec_dtw)]:
            v=fr[picks]; v=v[~np.isnan(v)]; v=v[v!=0]
            if len(v)<MIN_VOTES: continue
            fup=(v>0).mean()
            if fup>=THR or fup<=1-THR:
                frq=fr[q]
                if np.isnan(frq) or frq==0: continue
                store.append((int(drow[q]),qtr[q],(1 if fup>=.5 else -1)*frq*1e4-FEE))
        if qi%2000==0: print(f"  q {qi}/{len(qs)} {_t()-t0:.0f}s",flush=True)
    print(f"\n[done] {_t()-t0:.0f}s")
    print(f"\n===== 작업2: DTW top100 vs 유클리드 top100 겹침률 = {np.mean(overlaps):.2f} (낮으면 새 정보) =====")
    def rep(recs,tag):
        R=pd.DataFrame(recs,columns=['qday','qtr','net'])
        if len(R)<10: print(f"{tag}: n={len(R)}"); return
        te=~R.qtr.isin(TRAIN_Q)
        dm=R.groupby('qday')['net'].mean().to_numpy()
        bs=np.random.default_rng(7).choice(dm,(4000,len(dm)),replace=True).mean(axis=1)
        print(f"{tag}: n={len(R)} hit{(R.net+FEE>0).mean():.3f} pt{R.net.mean():+.1f} "
              f"일수익 full{R.net.sum()/nd:+.2f}/test{R[te].net.sum()/ndte if te.sum() else float('nan'):+.2f} "
              f"CI[{np.percentile(bs,2.5):+.0f},{np.percentile(bs,97.5):+.0f}]")
    print("\n===== 작업3: DTW kNN vs 유클리드 kNN (4h thr0.70, 같은 쿼리) =====")
    rep(rec_euc,'유클리드(베이스 재현)')
    rep(rec_dtw,'DTW')
    print(f"\n현행 4h 유클리드 hit 0.684, +90.5, 0.084%/day. DTW 가 방향 정렬+넘으면 새 길.")

if __name__=='__main__':
    main()
