#!/usr/bin/env python3
"""[I] 20단계 — 1d 전면 검증: 반전원리(시간대/일주기) + CI(robust) + 견고성 + 더 긴 horizon + 자본중첩.
19단계 i_longhz 구조 재사용, 1d 집중. 4h baseline 대비."""
import numpy as np, pandas as pd, json, datetime as dt
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB='/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
K_CAND=1500; N_IND=100; EXCL_DAYS=3; MIN_VOTES=70; FEE=11.0; THR=0.70
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
HZ={'4h':240,'1d':1440,'2d':2880,'3d':4320,'5d':7200,'7d':10080}

def greedy_gap(gmins,H,nt):
    acc=[];out=[]
    for i,g in enumerate(gmins):
        if all(abs(g-a)>=H for a in acc):
            acc.append(g);out.append(i)
            if len(out)>=nt: break
    return out

def main():
    nrm=pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    meta=json.load(open(f'{OUT}/reduce_norm_meta.json')); reps=[r for r in meta['reps'] if r!='spread_bp']
    yr=nrm['yr'].astype(int).to_numpy(); mod=nrm['min_of_day'].to_numpy()
    ndays=sorted(nrm['day'].unique()); dixn={d:i for i,d in enumerate(ndays)}
    drow=nrm['day'].map(dixn).to_numpy(); n=len(nrm); starts=np.searchsorted(drow,np.arange(len(ndays)))
    full=pd.read_parquet(LAB,columns=['day','min_of_day','mid'])
    fdays=sorted(full['day'].unique()); dixf={d:i for i,d in enumerate(fdays)}
    ddl=[dt.date.fromisoformat(d) for d in fdays]
    gap_after=np.zeros(len(fdays),bool)
    for i in range(1,len(fdays)):
        if (ddl[i]-ddl[i-1]).days>1: gap_after[i-1]=True
    cum_gap=np.cumsum(gap_after)
    Pf=np.full(len(fdays)*1440,np.nan,np.float32)
    Pf[full['day'].map(dixf).to_numpy()*1440+full['min_of_day'].to_numpy()]=full['mid'].to_numpy(np.float32)
    NG=len(Pf); gf=nrm['day'].map(dixf).to_numpy()*1440+mod; gday=gf//1440
    def fwd(H):
        fr=np.full(n,np.nan,np.float32); tgt=gf+H; ok=tgt<NG
        eday=np.minimum(tgt//1440,len(fdays)-1)
        nogap=(cum_gap[eday]-cum_gap[gday])==0
        m=ok&nogap; fr[m]=Pf[tgt[m]]/Pf[gf[m]]-1; return fr
    FR={hn:fwd(H) for hn,H in HZ.items()}
    C=nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32); m23=yr==2023
    mu=C[m23].mean(0); S=np.cov((C[m23]-mu).T); w,V=np.linalg.eigh(S)
    W=(V@np.diag(1/np.sqrt(np.maximum(w,1e-6)))@V.T).astype(np.float32)
    X=((C-mu)@W).astype(np.float32); xsq=(X*X).sum(1)
    reps_signed=[r for r in reps if r in meta['signed']]; sidx=[reps.index(c) for c in reps_signed]
    qs=np.where((yr>=2024)&(mod%10==5))[0]
    qtr=np.char.add(yr.astype(str),np.char.add('Q',(((nrm['day'].str[5:7].astype(int)-1)//3+1)).astype(str).to_numpy()))
    nd=len(set(drow[qs])); ndte=len(set(drow[qs][~np.isin(qtr[qs],TRAIN_Q)]))
    print(f"[query] {len(qs)}, horizons {list(HZ)}",flush=True)

    # 결과: 각 horizon thr0.70 거래 (q, dir, frq, mod, qtr, qday)
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
            gmins=gf[order]
            for hn,H in HZ.items():
                sel=greedy_gap(gmins,min(H,1440),N_IND); picks=order[sel]   # 1d+ 는 1440 독립
                v=FR[hn][picks]; v=v[~np.isnan(v)]; v=v[v!=0]
                if len(v)<MIN_VOTES: continue
                fup=(v>0).mean()
                if fup>=THR or fup<=1-THR:
                    frq=FR[hn][q]
                    if np.isnan(frq) or frq==0: continue
                    dr=1 if fup>=.5 else -1
                    recs[hn].append((int(drow[q]),int(mod[q]),qtr[q],dr,float(frq)))
        if bi%(BLK*150)==0: print(f"  q {bi}/{len(qs)} {_t()-t0:.0f}s",flush=True)
    for hn in HZ: pd.DataFrame(recs[hn],columns=['qday','mod','qtr','dir','frq']).to_parquet(f'{OUT}/v1d_{hn}.parquet')

    def dayci(D, vals, a=2.5):
        if len(vals)<5: return np.nan,np.nan,np.nan
        dm=pd.Series(vals).groupby(D).mean().to_numpy()
        bs=np.random.default_rng(7).choice(dm,(5000,len(dm)),replace=True).mean(axis=1)
        return dm.mean(),np.percentile(bs,a),np.percentile(bs,100-a)

    print("\n===== 작업4: horizon 공간 (1d 너머) — 비단조 계속? =====")
    print(f"{'hz':>4} | n(일) | hit | per-trade | full/test | day-CI95 | med")
    for hn in HZ:
        D=pd.DataFrame(recs[hn],columns=['qday','mod','qtr','dir','frq'])
        if len(D)<10: print(f"{hn:>4} | n={len(D)}"); continue
        net=D['dir'].to_numpy()*D['frq'].to_numpy()*1e4-FEE; te=~D.qtr.isin(TRAIN_Q)
        m,lo,hi=dayci(D.qday.to_numpy(),net)
        print(f"{hn:>4} | {len(D)}({D.qday.nunique()}) | {(net>0).mean():.3f} | {net.mean():+7.1f} | "
              f"{net.sum()/nd:+.2f}/{net[te.to_numpy()].sum()/ndte if te.sum() else float('nan'):+.2f} | [{lo:+.0f},{hi:+.0f}] | {np.median(net):+.0f}")

    # ===== 1d 집중 =====
    D=pd.DataFrame(recs['1d'],columns=['qday','mod','qtr','dir','frq'])
    net=D['dir'].to_numpy()*D['frq'].to_numpy()*1e4-FEE
    print(f"\n===== 작업1: 6h/8h 반전 vs 1d — 시간대(진입 mod) 분포 (일주기?) =====")
    # 1d 거래의 진입 시간대 분포 + 시간대별 hit
    for lo_,hi_,nm in [(480,720,'아침08-12'),(720,1080,'오후12-18'),(1080,1320,'저녁18-22'),(240,480,'새벽04-08')]:
        mm=(D['mod']>=lo_)&(D['mod']<hi_)
        if mm.sum()<5: continue
        print(f"  {nm}(mod{lo_}-{hi_}): n={mm.sum()} hit{(net[mm.to_numpy()]>0).mean():.3f} pt{net[mm.to_numpy()].mean():+.0f}")
    print(f"  → 특정 시간대 쏠림/의존이면 일주기 패턴(취약). 고르면 매끄러운 edge.")

    print(f"\n===== 작업2: CI robust (1d) — outlier/중앙값 =====")
    m,lo,hi=dayci(D.qday.to_numpy(),net)
    no_top=net[np.argsort(np.abs(net))[:-5]]   # top5 |net| 제외
    print(f"  day-mean {m:+.1f} CI95 [{lo:+.0f},{hi:+.0f}] | median {np.median(net):+.0f} | top5제외 mean {no_top.mean():+.1f}")
    print(f"  Bonferroni(horizon 6셀, 99.6%): ", end='')
    _,lob,hib=dayci(D.qday.to_numpy(),net,a=0.42); print(f"[{lob:+.0f},{hib:+.0f}]")

    print(f"\n===== 작업3: 견고성 — 폴드(반기)/방향분해 =====")
    D['half']=D.qtr.str[:4]+np.where(D.qtr.str[5].astype(int)<=2,'H1','H2')
    for fd,g in D.groupby('half'):
        nn=g['dir'].to_numpy()*g['frq'].to_numpy()*1e4-FEE
        if len(nn)<5: continue
        print(f"  {fd}: n={len(nn)} hit{(nn>0).mean():.3f} pt{nn.mean():+.0f}")
    up=D['dir']>0
    print(f"  up-lean n={up.sum()} hit{(net[up.to_numpy()]>0).mean():.3f} pt{net[up.to_numpy()].mean():+.0f} | "
          f"down n={(~up).sum()} hit{(net[~up.to_numpy()]>0).mean():.3f} pt{net[~up.to_numpy()].mean():+.0f}")

    print(f"\n===== 작업5: 자본중첩 (1d hold, one-way 단일포지션, 발화중 보유면 skip) =====")
    # 시간순 정렬, 보유 종료(진입+1440분) 전 발화는 skip
    D2=D.sort_values(['qday','mod']).reset_index(drop=True)
    held_until=-1; taken=[]
    for _,r in D2.iterrows():
        gmin=r['qday']*1440+r['mod']
        if gmin>=held_until:
            taken.append(r); held_until=gmin+1440
    Dt=pd.DataFrame(taken)
    nett=Dt['dir'].to_numpy()*Dt['frq'].to_numpy()*1e4-FEE
    print(f"  중첩제거 후: n={len(Dt)} (전 {len(D)}), hit{(nett>0).mean():.3f} pt{nett.mean():+.0f} 일수익 {nett.sum()/nd:+.2f}")
    print(f"  4h baseline +8.40/day. 1d 중첩후 실제 일수익 vs 4h.")

if __name__=='__main__':
    main()
