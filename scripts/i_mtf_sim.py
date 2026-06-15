#!/usr/bin/env python3
"""[I] 18단계 작업3+4 — 독립 MTF 봉 지표 추가 → kNN → thr 낮춤 hit/net OOS.
redundancy 게이트 통과한 독립 지표(30m/60m 봉 rsi/macd/stoch + 15m rsi/stoch). base21 대비."""
import numpy as np, pandas as pd, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from i_labeling import rsi, macd, stoch, sma

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB = '/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
K_CAND=1000; N_IND=100; EXCL_DAYS=3; MIN_VOTES=70; FEE=11.0
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
MTF_BARS={'15m':15,'30m':30,'60m':60}   # 독립 지표 나온 봉
MTF_IND=['rsi','macd','stoch']           # 15m 은 rsi/stoch, 30m/60m 은 rsi/macd/stoch

def greedy_h(od,om,h,nt):
    acc={};out=[]
    for i in range(len(od)):
        d=od[i];m=om[i];lst=acc.get(d)
        if lst is not None:
            if any(abs(m-mm)<h for mm in lst): continue
            lst.append(m)
        else: acc[d]=[m]
        out.append(i)
        if len(out)>=nt: break
    return out

def main():
    nrm=pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    meta=json.load(open(f'{OUT}/reduce_norm_meta.json')); reps=[r for r in meta['reps'] if r!='spread_bp']
    yr=nrm['yr'].astype(int).to_numpy(); mod=nrm['min_of_day'].to_numpy()
    days=sorted(nrm['day'].unique()); dix={d:i for i,d in enumerate(days)}
    drow=nrm['day'].map(dix).to_numpy(); n=len(nrm)
    starts=np.searchsorted(drow,np.arange(len(days)))

    # MTF 지표 계산 (전 nrm day, 분→봉→last 완성봉, causal)
    lab=pd.read_parquet(LAB,columns=['day','min_of_day','mid']); lab=lab[lab.day.isin(days)]
    midgrid=np.full((len(days),1440),np.nan,np.float32)
    midgrid[lab['day'].map(dix).to_numpy(),lab['min_of_day'].to_numpy()]=lab['mid'].to_numpy(np.float32)
    feat_names=[]
    MTF=np.full((n, 8), np.nan, np.float32)   # 15m rsi/stoch + 30m rsi/macd/stoch + 60m rsi/macd/stoch
    cols=[('15m','rsi'),('15m','stoch'),('30m','rsi'),('30m','macd'),('30m','stoch'),('60m','rsi'),('60m','macd'),('60m','stoch')]
    feat_names=[f'{i}_{b}' for b,i in cols]
    # day별 분 라벨 채우기 위해 nrm 행 인덱스 맵
    row_of=np.full((len(days),1440),-1,dtype=np.int64)
    row_of[drow,mod]=np.arange(n)
    from time import time as _t; t0=_t()
    for di in range(len(days)):
        full=pd.Series(midgrid[di], index=np.arange(1440))
        per_bar={}
        for B in set(MTF_BARS[b] for b,_ in cols):
            bclose=full.groupby(full.index//B).last().reset_index(drop=True)
            r=rsi(bclose,14); mc,_,_=macd(bclose)
            sk,_=stoch(bclose,bclose,bclose,14,3)
            bar_of_min=np.arange(1440)//B
            per_bar[(B,'rsi')]=r.shift(1).reindex(bar_of_min).to_numpy()
            per_bar[(B,'macd')]=mc.shift(1).reindex(bar_of_min).to_numpy()
            per_bar[(B,'stoch')]=sk.shift(1).reindex(bar_of_min).to_numpy()
        for ci,(b,ind) in enumerate(cols):
            vals=per_bar[(MTF_BARS[b],ind)]
            rr=row_of[di]; ok=rr>=0
            MTF[rr[ok], ci]=vals[ok]
        if di%300==0: print(f"  MTF day {di}/{len(days)} {_t()-t0:.0f}s",flush=True)

    # base21 whitened
    C=nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32); m23=yr==2023
    def whiten(M,fm):
        mu=M[fm].mean(0); S=np.atleast_2d(np.cov((M[fm]-mu).T)); w,V=np.linalg.eigh(S)
        W=(V@np.diag(1/np.sqrt(np.maximum(w,1e-6)))@V.T).astype(np.float32); return ((M-mu)@W).astype(np.float32)
    X21=whiten(C,m23)
    # MTF robust z (2023), NaN→0 (봉 부족 분은 중립)
    valid_mtf=~np.isnan(MTF).any(1)
    med=np.nanmedian(MTF[m23&valid_mtf],0); iqr=(np.nanpercentile(MTF[m23&valid_mtf],75,0)-np.nanpercentile(MTF[m23&valid_mtf],25,0))/1.349
    MTFz=np.nan_to_num((MTF-med)/(iqr+1e-9)).clip(-10,10).astype(np.float32)
    Xmtf=np.column_stack([X21, MTFz]).astype(np.float32)   # 29차원
    print(f"[built] MTF 8dim, valid {valid_mtf.mean():.2f}, Xmtf {Xmtf.shape}")

    lab2=pd.read_parquet(LAB,columns=['day','min_of_day','mid']); lab2=lab2[lab2.day.isin(days)]
    fr=np.full(n,np.nan,np.float32); ok=mod+240<=1439
    fr[ok]=midgrid[drow[ok],mod[ok]+240]/midgrid[drow[ok],mod[ok]]-1
    qtr=np.char.add(yr.astype(str),np.char.add('Q',(((nrm['day'].str[5:7].astype(int)-1)//3+1)).astype(str).to_numpy()))
    qs=np.where((yr>=2024)&(mod%10==5))[0]
    nd=len(set(drow[qs])); ndte=len(set(drow[qs][~np.isin(qtr[qs],TRAIN_Q)]))

    def run(X,tag,thr):
        xsq=(X*X).sum(1); recs=[]; BLK=128
        for bi in range(0,len(qs),BLK):
            qb=qs[bi:bi+BLK]; ends=starts[np.maximum(drow[qb]-EXCL_DAYS,0)]; emax=int(ends.max())
            if emax<50000: continue
            d2=xsq[None,:emax]-2.0*(X[qb]@X[:emax].T)
            for j,q in enumerate(qb):
                e=int(ends[j])
                if e<50000: continue
                row=d2[j,:e]; kc=min(K_CAND,e-1)
                cand=np.argpartition(row,kc)[:kc]; order=cand[np.argsort(row[cand])]
                sel=greedy_h(drow[order],mod[order],240,N_IND); picks=order[sel]
                v=fr[picks]; v=v[~np.isnan(v)]; v=v[v!=0]
                if len(v)<MIN_VOTES: continue
                fup=(v>0).mean()
                if fup>=thr or fup<=1-thr:
                    frq=fr[q]
                    if np.isnan(frq) or frq==0: continue
                    recs.append((int(drow[q]),qtr[q],(1 if fup>=.5 else -1)*frq*1e4-FEE))
        R=pd.DataFrame(recs,columns=['qday','qtr','net']); te=~R.qtr.isin(TRAIN_Q)
        return dict(tag=tag,thr=thr,n=len(R),hit=(R.net+FEE>0).mean(),daily=R.net.sum()/nd,
                    te_n=int(te.sum()),te_daily=R[te].net.sum()/ndte if te.sum() else np.nan)

    print(f"\n===== 작업4: thr 낮춤 base21 vs MTF29 (4h, full/test 일수익) =====")
    print(f"{'thr':>5} | base21 n/hit/full/test | MTF29 n/hit/full/test")
    from time import time as _t; t0=_t()
    for thr in [0.70, 0.65, 0.60]:
        b=run(X21,'base21',thr); m=run(Xmtf,'mtf29',thr)
        print(f"{thr:.2f} | {b['n']:4d} {b['hit']:.3f} {b['daily']:+.2f}/{b['te_daily']:+.2f} | "
              f"{m['n']:4d} {m['hit']:.3f} {m['daily']:+.2f}/{m['te_daily']:+.2f}  ({_t()-t0:.0f}s)",flush=True)
    print(f"\n현행 base21 thr0.70 test +2.90. MTF 가 thr 낮춰도 정확도 유지+빈도↑면 빈도 해결.")

if __name__=='__main__':
    main()
