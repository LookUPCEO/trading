#!/usr/bin/env python3
"""[I] 19단계 — 긴 horizon 공간 (4h 너머). cross-day 연속 가격, 각 horizon 자체 유사도 lean.
horizon: 4h/6h/8h/12h/1d/2d/3d. net 단조 계속/peak, 거래당×빈도=일수익."""
import numpy as np, pandas as pd, json
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB='/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
K_CAND=1000; N_IND=100; EXCL_DAYS=3; MIN_VOTES=70; FEE=11.0; THR=0.70
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
HZ={'4h':240,'6h':360,'8h':480,'12h':720,'1d':1440,'2d':2880,'3d':4320}

def greedy_h(od,om,h,nt):
    acc={};out=[]
    for i in range(len(od)):
        d=od[i];m=om[i];lst=acc.get(d)
        if lst is not None:
            if any(abs(m-mm)<min(h,1439) for mm in lst): continue
            lst.append(m)
        else: acc[d]=[m]
        out.append(i)
        if len(out)>=nt: break
    return out

def main():
    nrm=pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    meta=json.load(open(f'{OUT}/reduce_norm_meta.json')); reps=[r for r in meta['reps'] if r!='spread_bp']
    yr=nrm['yr'].astype(int).to_numpy(); mod=nrm['min_of_day'].to_numpy()
    nrm_days=sorted(nrm['day'].unique()); dixn={d:i for i,d in enumerate(nrm_days)}
    drow=nrm['day'].map(dixn).to_numpy(); n=len(nrm)
    starts=np.searchsorted(drow,np.arange(len(nrm_days)))

    # 전체 1198일 달력연속 가격 시계열 (cross-day forward 용)
    full=pd.read_parquet(LAB,columns=['day','min_of_day','mid'])
    fdays=sorted(full['day'].unique()); dixf={d:i for i,d in enumerate(fdays)}
    # 달력 연속 확인
    import datetime as dt
    dd=[dt.date.fromisoformat(d) for d in fdays]
    gaps=sum(1 for i in range(1,len(dd)) if (dd[i]-dd[i-1]).days>1)
    Pf=np.full(len(fdays)*1440, np.nan, np.float32)
    gf_all=full['day'].map(dixf).to_numpy()*1440 + full['min_of_day'].to_numpy()
    Pf[gf_all]=full['mid'].to_numpy(np.float32)
    print(f"[load] nrm {n}행/{len(nrm_days)}일, full {len(fdays)}일 달력공백>1 {gaps}개 (연속이어야)", flush=True)

    # nrm 각 행의 full global minute
    gf=nrm['day'].map(dixf).to_numpy()*1440 + mod
    NG=len(Pf)
    FR={}   # horizon → 각 nrm 행 forward return (cross-day)
    for hn,H in HZ.items():
        fr=np.full(n,np.nan,np.float32)
        tgt=gf+H; ok=tgt<NG
        fr[ok]=Pf[tgt[ok]]/Pf[gf[ok]]-1
        FR[hn]=fr

    C=nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32); m23=yr==2023
    mu=C[m23].mean(0); S=np.cov((C[m23]-mu).T); w,V=np.linalg.eigh(S)
    W=(V@np.diag(1/np.sqrt(np.maximum(w,1e-6)))@V.T).astype(np.float32)
    X=((C-mu)@W).astype(np.float32); xsq=(X*X).sum(1)

    qs=np.where((yr>=2024)&(mod%10==5))[0]
    qtr=np.char.add(yr.astype(str),np.char.add('Q',(((nrm['day'].str[5:7].astype(int)-1)//3+1)).astype(str).to_numpy()))
    nd=len(set(drow[qs])); ndte=len(set(drow[qs][~np.isin(qtr[qs],TRAIN_Q)]))
    print(f"[query] {len(qs)}", flush=True)

    recs={hn:[] for hn in HZ}
    BLK=128; from time import time as _t; t0=_t()
    for bi in range(0,len(qs),BLK):
        qb=qs[bi:bi+BLK]; ends=starts[np.maximum(drow[qb]-EXCL_DAYS,0)]; emax=int(ends.max())
        if emax<50000: continue
        d2=xsq[None,:emax]-2.0*(X[qb]@X[:emax].T)
        for j,q in enumerate(qb):
            e=int(ends[j])
            if e<50000: continue
            row=d2[j,:e]; kc=min(K_CAND,e-1)
            cand=np.argpartition(row,kc)[:kc]; order=cand[np.argsort(row[cand])]
            # 독립: 최장 horizon 기준 보수 (240분 day내 + cross-day 자동독립)
            sel=greedy_h(drow[order],mod[order],240,N_IND); picks=order[sel]
            for hn in HZ:
                v=FR[hn][picks]; v=v[~np.isnan(v)]; v=v[v!=0]
                if len(v)<MIN_VOTES: continue
                fup=(v>0).mean()
                if fup>=THR or fup<=1-THR:
                    frq=FR[hn][q]
                    if np.isnan(frq) or frq==0: continue
                    recs[hn].append((int(drow[q]),qtr[q],(1 if fup>=.5 else -1)*frq*1e4-FEE))
        if bi%(BLK*100)==0: print(f"  q {bi}/{len(qs)} {_t()-t0:.0f}s",flush=True)

    print(f"\n===== 작업1+2+3: horizon 별 (thr0.70 자체 lean, cross-day) =====")
    print(f"{'hz':>4} | n(빈도/d) | hit | per-trade net | 일수익 full/test | day-CI")
    rows=[]
    for hn in HZ:
        R=pd.DataFrame(recs[hn],columns=['qday','qtr','net'])
        if len(R)<10: print(f"{hn:>4} | n={len(R)} (희소)"); continue
        te=~R.qtr.isin(TRAIN_Q)
        dm=R.groupby('qday')['net'].mean().to_numpy()
        bs=np.random.default_rng(7).choice(dm,(4000,len(dm)),replace=True).mean(axis=1)
        lo,hi=np.percentile(bs,[2.5,97.5])
        rows.append(dict(hz=hn,H=HZ[hn],n=len(R),hit=(R.net+FEE>0).mean(),pt=R.net.mean(),
                         daily=R.net.sum()/nd, daily_te=R[te].net.sum()/ndte if te.sum() else np.nan, lo=lo,hi=hi))
        print(f"{hn:>4} | {len(R):4d}({len(R)/nd:.3f}) | {(R.net+FEE>0).mean():.3f} | {R.net.mean():+7.1f} | "
              f"{R.net.sum()/nd:+.2f}/{R[te].net.sum()/ndte if te.sum() else float('nan'):+.2f} | [{lo:+.0f},{hi:+.0f}]")
    pd.DataFrame(rows).to_csv(f'{OUT}/longhz.csv',index=False)
    print(f"\n현행 4h 0.084%/day. net 단조 4h 이후 계속? peak? hit 동반? 길수록 n↓ CI↑.")
    print("일수익 = 거래당×빈도. 긴 horizon 자본회전 느림(하루 묶이면 그날 1거래).")

if __name__=='__main__':
    main()
