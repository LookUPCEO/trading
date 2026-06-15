#!/usr/bin/env python3
"""[I] 19단계 audit — 1d/12h 재검: 올바른 독립성(h=H, 창 비겹침) + cross-day gap 마스킹.
1d +16.5 가 진짜인지 (독립성·gap 인공물 제거 후)."""
import numpy as np, pandas as pd, json, datetime as dt
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB='/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
K_CAND=1500; N_IND=100; EXCL_DAYS=3; MIN_VOTES=70; FEE=11.0; THR=0.70
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
HZ={'12h':720,'1d':1440}

def greedy_gap(gmins, H, nt):
    """global-minute 기준 |Δ|>=H 만 독립 (창 비겹침). 정렬 가정 X → 그리디."""
    acc=[]; out=[]
    for i,g in enumerate(gmins):
        if all(abs(g-a)>=H for a in acc):
            acc.append(g); out.append(i)
            if len(out)>=nt: break
    return out

def main():
    nrm=pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    meta=json.load(open(f'{OUT}/reduce_norm_meta.json')); reps=[r for r in meta['reps'] if r!='spread_bp']
    yr=nrm['yr'].astype(int).to_numpy(); mod=nrm['min_of_day'].to_numpy()
    nrm_days=sorted(nrm['day'].unique()); dixn={d:i for i,d in enumerate(nrm_days)}
    drow=nrm['day'].map(dixn).to_numpy(); n=len(nrm)
    starts=np.searchsorted(drow,np.arange(len(nrm_days)))

    full=pd.read_parquet(LAB,columns=['day','min_of_day','mid'])
    fdays=sorted(full['day'].unique()); dixf={d:i for i,d in enumerate(fdays)}
    ddl=[dt.date.fromisoformat(d) for d in fdays]
    # gap 위치: day i 와 i-1 사이 공백>1 → 그 경계 넘는 forward 무효
    gap_after=np.zeros(len(fdays),bool)
    for i in range(1,len(fdays)):
        if (ddl[i]-ddl[i-1]).days>1: gap_after[i-1]=True
    print(f"[gap] 공백 위치: {[fdays[i] for i in range(len(fdays)) if gap_after[i]]}")
    Pf=np.full(len(fdays)*1440,np.nan,np.float32)
    Pf[full['day'].map(dixf).to_numpy()*1440+full['min_of_day'].to_numpy()]=full['mid'].to_numpy(np.float32)
    NG=len(Pf)
    gf=nrm['day'].map(dixf).to_numpy()*1440+mod
    gday=gf//1440
    # forward (gap 마스킹: g~g+H 사이에 gap_after day 있으면 무효)
    cum_gap=np.cumsum(gap_after)
    def fwd(H):
        fr=np.full(n,np.nan,np.float32); tgt=gf+H; ok=tgt<NG
        # 시작일~도착일 사이 gap 없어야
        sday=gday; eday=np.minimum((tgt)//1440, len(fdays)-1)
        nogap=(cum_gap[np.minimum(eday,len(fdays)-1)]-cum_gap[sday])==0
        m=ok&nogap
        fr[m]=Pf[tgt[m]]/Pf[gf[m]]-1
        return fr
    FR={hn:fwd(H) for hn,H in HZ.items()}

    C=nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32); m23=yr==2023
    mu=C[m23].mean(0); S=np.cov((C[m23]-mu).T); w,V=np.linalg.eigh(S)
    W=(V@np.diag(1/np.sqrt(np.maximum(w,1e-6)))@V.T).astype(np.float32)
    X=((C-mu)@W).astype(np.float32); xsq=(X*X).sum(1)
    gf_nrm=gf
    qs=np.where((yr>=2024)&(mod%10==5))[0]
    qtr=np.char.add(yr.astype(str),np.char.add('Q',(((nrm['day'].str[5:7].astype(int)-1)//3+1)).astype(str).to_numpy()))
    nd=len(set(drow[qs])); ndte=len(set(drow[qs][~np.isin(qtr[qs],TRAIN_Q)]))
    print(f"[query] {len(qs)} | 올바른 독립성 (창 비겹침 h=H) + gap 마스킹\n")

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
            gmins_order=gf_nrm[order]
            for hn,H in HZ.items():
                sel=greedy_gap(gmins_order, H, N_IND)   # 창 비겹침 독립
                picks=order[sel]
                v=FR[hn][picks]; v=v[~np.isnan(v)]; v=v[v!=0]
                if len(v)<MIN_VOTES: continue
                fup=(v>0).mean()
                if fup>=THR or fup<=1-THR:
                    frq=FR[hn][q]
                    if np.isnan(frq) or frq==0: continue
                    recs[hn].append((int(drow[q]),qtr[q],(1 if fup>=.5 else -1)*frq*1e4-FEE))
        if bi%(BLK*150)==0: print(f"  q {bi}/{len(qs)} {_t()-t0:.0f}s",flush=True)

    print(f"\n===== 1d/12h 재검 (올바른 독립성+gap 마스킹) =====")
    for hn in HZ:
        R=pd.DataFrame(recs[hn],columns=['qday','qtr','net'])
        if len(R)<10: print(f"{hn}: n={len(R)}"); continue
        te=~R.qtr.isin(TRAIN_Q)
        dm=R.groupby('qday')['net'].mean().to_numpy()
        bs=np.random.default_rng(7).choice(dm,(4000,len(dm)),replace=True).mean(axis=1)
        lo,hi=np.percentile(bs,[2.5,97.5])
        # 시기 분해
        q24=R[R.qtr.str[:4]=='2024'].net; q25=R[R.qtr.str[:4]=='2025'].net; q26=R[R.qtr.str[:4]=='2026'].net
        print(f"{hn}: n={len(R)}({len(R)/nd:.3f}/d, {len(set(R.qday))}일) hit{(R.net+FEE>0).mean():.3f} "
              f"pt{R.net.mean():+.1f} 일수익 full{R.net.sum()/nd:+.2f}/test{R[te].net.sum()/ndte if te.sum() else float('nan'):+.2f} "
              f"day-CI[{lo:+.0f},{hi:+.0f}]")
        print(f"     시기: 2024 n{len(q24)} {q24.mean()*0+ (q24>0).mean() if len(q24) else 0:.2f}hit pt{q24.mean() if len(q24) else float('nan'):+.0f} | "
              f"2025 n{len(q25)} pt{q25.mean() if len(q25) else float('nan'):+.0f} | 2026 n{len(q26)} pt{q26.mean() if len(q26) else float('nan'):+.0f}")
    print(f"\n현행 4h +8.40/day. 독립성 엄격(창 비겹침)+gap 제거 후에도 1d 양수면 후보, 무너지면 인공물.")

if __name__=='__main__':
    main()
