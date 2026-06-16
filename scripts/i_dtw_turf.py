#!/usr/bin/env python3
"""[I] 21-1 — DTW 자기 운동장 검증. 4h 방향에 안 묶고: 여러 horizon + 변동성/추세지속 예측 +
강한 모양일치 + 거래화. DTW(과거 60분 경로) 후보=21차원 top-300, DTW 재순위. 21단계와 동일 엔진."""
import numpy as np, pandas as pd, json
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB='/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
NWIN=60; BAND=8; K_EUC=300; N_IND=100; EXCL_DAYS=3; MIN_VOTES=70; FEE=11.0; THR=0.70
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
HZ={'5m':5,'30m':30,'1h':60,'4h':240,'1d':1440}

def batch_dtw(q,Cand,band=BAND):
    M,N=Cand.shape; INF=1e18
    dp=np.full((M,N),INF)
    for j in range(0,min(band,N-1)+1):
        c=(q[0]-Cand[:,j])**2; dp[:,j]=c if j==0 else dp[:,j-1]+c
    for i in range(1,N):
        cur=np.full((M,N),INF); j0=max(0,i-band); j1=min(N-1,i+band)
        for j in range(j0,j1+1):
            c=(q[i]-Cand[:,j])**2; best=dp[:,j].copy()
            if j>0: best=np.minimum(best,cur[:,j-1]); best=np.minimum(best,dp[:,j-1])
            cur[:,j]=c+best
        dp=cur
    return dp[:,N-1]

def main():
    nrm=pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    meta=json.load(open(f'{OUT}/reduce_norm_meta.json')); reps=[r for r in meta['reps'] if r!='spread_bp']
    yr=nrm['yr'].astype(int).to_numpy(); mod=nrm['min_of_day'].to_numpy()
    days=sorted(nrm['day'].unique()); dix={d:i for i,d in enumerate(days)}
    drow=nrm['day'].map(dix).to_numpy(); n=len(nrm); starts=np.searchsorted(drow,np.arange(len(days)))
    full=pd.read_parquet(LAB,columns=['day','min_of_day','mid']); full=full[full.day.isin(days)]
    fdays=sorted(full['day'].unique()); dixf={d:i for i,d in enumerate(fdays)}
    Pf=np.full(len(fdays)*1440,np.nan,np.float32); Pf[full['day'].map(dixf).to_numpy()*1440+full['min_of_day'].to_numpy()]=full['mid'].to_numpy(np.float32)
    NG=len(Pf); gf=nrm['day'].map(dixf).to_numpy()*1440+mod
    mid=np.full((len(days),1440),np.nan,np.float32); mid[drow,mod]=full.set_index(['day','min_of_day']).reindex(list(zip(nrm['day'],nrm['min_of_day'])))['mid'].to_numpy(np.float32) if False else 0
    # mid grid 직접
    mid=np.full((len(days),1440),np.nan,np.float32); mid[full['day'].map(dix).to_numpy(),full['min_of_day'].to_numpy()]=full['mid'].to_numpy(np.float32)
    logm=np.log(mid)
    PATH=np.full((n,NWIN),np.nan,np.float32)
    for r in range(n):
        d=drow[r]; m=mod[r]
        if m-NWIN<0: continue
        seg=logm[d,m-NWIN:m+1]
        if np.isnan(seg).any(): continue
        dif=np.diff(seg); PATH[r]=(dif/(dif.std()+1e-9)).astype(np.float32)
    valid=~np.isnan(PATH).any(1)
    C=nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32); m23=yr==2023
    mu=C[m23].mean(0); S=np.cov((C[m23]-mu).T); w,V=np.linalg.eigh(S)
    W=(V@np.diag(1/np.sqrt(np.maximum(w,1e-6)))@V.T).astype(np.float32); X=((C-mu)@W).astype(np.float32); xsq=(X*X).sum(1)
    # 미래 (cross-day for 1d)
    FR={}
    for hn,H in HZ.items():
        fr=np.full(n,np.nan,np.float32); tgt=gf+H; ok=tgt<NG; fr[ok]=Pf[tgt[ok]]/Pf[gf[ok]]-1; FR[hn]=fr
    qtr=np.char.add(yr.astype(str),np.char.add('Q',(((nrm['day'].str[5:7].astype(int)-1)//3+1)).astype(str).to_numpy()))
    qs=np.where((yr>=2024)&(mod%10==5)&valid)[0]
    qmax=int(__import__('os').environ.get('QMAX','0'))
    if qmax: qs=qs[::max(1,len(qs)//qmax)]
    print(f"[setup] queries {len(qs)}",flush=True)

    def dedupe(idxs,nt):
        seen=set();out=[]
        for i in idxs:
            if drow[i] in seen: continue
            seen.add(drow[i]);out.append(i)
            if len(out)>=nt: break
        return out

    # 각 쿼리: DTW top100, euclid top100, 그리고 DTW 거리(강일치용)
    # 예측 대상별 수집: 방향(각 horizon), 변동성(|4h|), 추세지속(부호 일치 4h vs 과거경로 부호)
    res={'dtw':{hn:[] for hn in HZ}, 'euc':{hn:[] for hn in HZ}}
    vol_dtw=[]; vol_euc=[]   # DTW/euc 이웃의 |4h| 평균 vs 쿼리 실제 |4h| (변동성 예측)
    strong=[]                # 강한 모양일치(DTW 거리 하위) 쿼리의 4h 방향 hit
    from time import time as _t; t0=_t()
    for qi,q in enumerate(qs):
        e=starts[max(drow[q]-EXCL_DAYS,0)]
        if e<50000: continue
        d2=xsq[:e]-2.0*(X[:e]@X[q]); vv=valid[:e]
        cand=np.argpartition(np.where(vv,d2,1e18),K_EUC)[:K_EUC]; cand=cand[vv[cand]]
        euc_order=cand[np.argsort(d2[cand])]; euc100=dedupe(euc_order,100)
        dd=batch_dtw(PATH[q],PATH[cand]); dtw_order=cand[np.argsort(dd)]; dtw100=dedupe(dtw_order,100)
        dtw_mindist=np.sort(dd)[:10].mean()   # 강일치 척도
        for nm,picks in [('dtw',dtw100),('euc',euc100)]:
            for hn in HZ:
                v=FR[hn][picks]; v=v[~np.isnan(v)]; v=v[v!=0]
                if len(v)<MIN_VOTES: continue
                fup=(v>0).mean()
                if fup>=THR or fup<=1-THR:
                    frq=FR[hn][q]
                    if not (np.isnan(frq) or frq==0):
                        res[nm][hn].append((int(drow[q]),qtr[q],(1 if fup>=.5 else -1)*frq*1e4-FEE))
        # 변동성 예측: 이웃의 |4h| 평균 vs 쿼리 |4h|
        nb_dtw=np.abs(FR['4h'][dtw100]); nb_dtw=nb_dtw[~np.isnan(nb_dtw)]
        nb_euc=np.abs(FR['4h'][euc100]); nb_euc=nb_euc[~np.isnan(nb_euc)]
        qv=abs(FR['4h'][q])
        if not np.isnan(qv) and len(nb_dtw)>50 and len(nb_euc)>50:
            vol_dtw.append((nb_dtw.mean(),qv)); vol_euc.append((nb_euc.mean(),qv))
        strong.append((dtw_mindist, FR['4h'][q]))
        if qi%2000==0: print(f"  q {qi}/{len(qs)} {_t()-t0:.0f}s",flush=True)
    nd=len(set(drow[qs])); ndte=len(set(drow[qs][~np.isin(qtr[qs],TRAIN_Q)]))

    print(f"\n===== 작업1: DTW vs 유클리드 — horizon별 방향 hit/net (같은 쿼리/엔진) =====")
    print(f"{'hz':>4} | DTW n/hit/net/test | EUC n/hit/net/test")
    for hn in HZ:
        out={}
        for nm in ['dtw','euc']:
            R=pd.DataFrame(res[nm][hn],columns=['qday','qtr','net'])
            if len(R)<10: out[nm]=f"n={len(R)}"; continue
            te=~R.qtr.isin(TRAIN_Q)
            out[nm]=f"{len(R):4d} {(R.net+FEE>0).mean():.3f} {R.net.mean():+6.1f} t{R[te].net.sum()/ndte if te.sum() else float('nan'):+.1f}"
        print(f"{hn:>4} | {out['dtw']:30s} | {out['euc']}")

    print(f"\n===== 작업2: DTW 변동성 예측 (이웃 |4h| → 쿼리 |4h| 상관) =====")
    vd=np.array(vol_dtw); ve=np.array(vol_euc)
    if len(vd)>10:
        cd=np.corrcoef(vd[:,0],vd[:,1])[0,1]; ce=np.corrcoef(ve[:,0],ve[:,1])[0,1]
        print(f"  DTW 이웃|4h| vs 실제|4h| corr {cd:.3f} | 유클리드 {ce:.3f} (DTW 변동성 더 예측?)")

    print(f"\n===== 작업3: 강한 모양일치(DTW 거리 하위 decile)만 4h 방향 =====")
    st=np.array(strong); st=st[~np.isnan(st[:,1])]
    thr_d=np.percentile(st[:,0],10)
    sm=st[st[:,0]<=thr_d]
    print(f"  강일치(하위10%, n={len(sm)}): 4h 방향 hit(>0) {(sm[:,1]>0).mean():.3f} | 전체 {(st[:,1]>0).mean():.3f} (강일치가 방향↑?)")
    print(f"  강일치 |4h| 평균 {np.abs(sm[:,1]).mean()*1e4:.0f}bp vs 전체 {np.abs(st[:,1]).mean()*1e4:.0f}bp (변동성↑?)")
    print(f"\n현행 4h 유클리드 hit 0.684(실 baseline). DTW 자기 운동장(다른 horizon/변동성/강일치) 강점 있나.")

if __name__=='__main__':
    main()
