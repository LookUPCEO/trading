#!/usr/bin/env python3
"""[I] 28-1단계 — 하락 특화 라벨 + kNN(닮음). 본인 지적: 27/28 은 GBM(예측)으로 봤으나
우리 방법은 닮음(kNN). 4h 상승이 작동한 건 kNN. → 28 하락 라벨을 kNN 으로 다시.
공간 3개로 닮은 과거 100개 찾고 미래 하락 방향 투표:
  (A) 21차원 거울(25/26 baseline) (B) 하락 특화 10라벨 단독 (C) 21+10 결합.
whitening 2023, greedy_h 이웃(검증 방식), pool=prefix(OOS), cross-day 미래(긴 horizon).
작업1 정렬: 각 공간의 이웃 미래 부호일치(방향 정렬되나=18단계 희석 점검).
down-lean fup<=thr, causal 독립, hit/net/OOS/CI/Bonferroni. lookahead 0."""
import os, numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB='/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
NEW=['bid_dep_chg30','bid_dep_chg60','ask_dep_chg30','book_thin_asym','bid_conc',
     'dn_rv_60','dn_rv_300','rv_skew_300','sell_accel','sell_spike']
HZ={'5m':5,'10m':10,'15m':15,'30m':30,'1h':60,'2h':120,'4h':240}
K_CAND=1000;N_NB=100;EXCL_DAYS=3;MINV=70;FEE=11.0
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']

def whiten(C, m23):
    C=np.clip(C, np.percentile(C[m23],1,0), np.percentile(C[m23],99,0))
    mu=C[m23].mean(0);S=np.cov((C[m23]-mu).T)
    if S.ndim==0: S=S.reshape(1,1)
    w,V=np.linalg.eigh(S);W=(V@np.diag(1/np.sqrt(np.maximum(w,1e-9)))@V.T)
    return ((C-mu)@W).astype(np.float32)

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

def dayci(qd,net,seed=7):
    if len(net)<5: return np.nan,np.nan,np.nan
    dm=pd.Series(net).groupby(qd).mean().to_numpy()
    bs=np.random.default_rng(seed).choice(dm,(4000,len(dm)),replace=True).mean(1)
    return dm.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)

def build():
    dl=pd.read_parquet(f'{OUT}/downlab.parquet')
    red=pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet')
    zc=[c for c in red.columns if c.startswith('z_')]
    M=dl.merge(red[['day','min_of_day']+zc],on=['day','min_of_day'],how='inner')
    M=M.sort_values(['day','min_of_day']).reset_index(drop=True)
    print(f"[build] downlab∩21dim = {len(M)} rows, {M.day.nunique()} days")
    yr=M['day'].str[:4].astype(int).to_numpy();mod=M['min_of_day'].to_numpy()
    month=M['day'].str[5:7].astype(int).to_numpy()
    qtr=np.array([f"{yr[i]}Q{(month[i]-1)//3+1}" for i in range(len(M))])
    days=sorted(M['day'].unique());dix={d:i for i,d in enumerate(days)}
    drow=M['day'].map(dix).to_numpy();starts=np.searchsorted(drow,np.arange(len(days)))
    # cross-day 미래 = 전체 labels.parquet 가격(모든 day 존재)
    full=pd.read_parquet(LAB,columns=['day','min_of_day','mid'])
    fdays=sorted(full['day'].unique());dixf={d:i for i,d in enumerate(fdays)}
    Pf=np.full(len(fdays)*1440,np.nan,np.float32)
    Pf[full['day'].map(dixf).to_numpy()*1440+full['min_of_day'].to_numpy()]=full['mid'].to_numpy(np.float32)
    gf=M['day'].map(dixf).to_numpy()*1440+mod;NG=len(Pf)
    FR={}
    for hn,H in HZ.items():
        f=np.full(len(M),np.nan,np.float32);t=gf+H;ok=t<NG;f[ok]=Pf[t[ok]]/Pf[gf[ok]]-1;FR[hn]=f
    m23=yr==2023
    SP={'21dim(거울)':whiten(M[zc].to_numpy(np.float64),m23),
        '하락10단독':whiten(M[NEW].to_numpy(np.float64),m23),
        '21+하락10':whiten(np.hstack([M[zc].to_numpy(np.float64),M[NEW].to_numpy(np.float64)]),m23)}
    return dict(M=M,drow=drow,mod=mod,qtr=qtr,starts=starts,FR=FR,SP=SP,days=days)

def search(D, Xname):
    X=D['SP'][Xname];xsq=(X*X).sum(1);drow=D['drow'];mod=D['mod'];starts=D['starts'];FR=D['FR']
    qs=np.where((np.array([q[:4] for q in D['M']['day']]).astype(int)>=2024)&(mod%10==5))[0]
    recs=[];BLK=128
    align=[]   # 이웃 미래 부호일치(정렬) 점검
    for bi in range(0,len(qs),BLK):
        qb=qs[bi:bi+BLK];ends=starts[np.maximum(drow[qb]-EXCL_DAYS,0)];emax=int(ends.max())
        if emax<20000: continue
        d2=xsq[None,:emax]-2.0*(X[qb]@X[:emax].T)
        for j,q in enumerate(qb):
            e=int(ends[j])
            if e<20000: continue
            row=d2[j,:e];kc=min(K_CAND,e-1)
            cand=np.argpartition(row,kc)[:kc];order=cand[np.argsort(row[cand])]
            od,om=drow[order],mod[order]
            rec=dict(q=int(q),qday=int(drow[q]),mod=int(mod[q]),qtr=D['qtr'][q])
            for hn in HZ:
                sel=greedy_h(od,om,HZ[hn],N_NB);picks=order[sel]
                v=FR[hn][picks];v=v[~np.isnan(v)];v=v[v!=0]
                rec[f'{hn}_n']=len(v);rec[f'{hn}_fup']=(v>0).mean() if len(v) else np.nan
                rec[f'{hn}_frq']=float(FR[hn][q])
            recs.append(rec)
            if Xname!='21dim(거울)':
                sel=greedy_h(od,om,240,N_NB);picks=order[sel];vv=FR['4h'][picks];vv=vv[~np.isnan(vv)&(vv!=0)]
                qf=FR['4h'][q]
                if len(vv)>30 and not np.isnan(qf) and qf!=0:
                    align.append(((vv>0).mean(), qf>0))
    R=pd.DataFrame(recs)
    return R, align

def analyze(R, name):
    print(f"\n----- {name}: down kNN (causal 독립) -----")
    print(f"{'hz':>4} {'thr':>5} | {'indN':>5} {'hit':>5} {'gross':>6} {'net':>6} | teHit dayCI(test)")
    cells=[]
    for hn in HZ:
        for thr in [.30,.35,.40]:
            ok=(R[f'{hn}_n']>=MINV)&~R[f'{hn}_frq'].isna()&(R[f'{hn}_frq']!=0)
            s=R[ok];L=s[s[f'{hn}_fup']<=thr].copy()
            if len(L)<5: continue
            frq=L[f'{hn}_frq'].to_numpy();L=L.assign(net=(-1)*frq*1e4-FEE,hit=(frq<0))
            o=np.lexsort((L['mod'].to_numpy(),L.qday.to_numpy()));acc={};keep=np.zeros(len(L),bool)
            mo=L['mod'].to_numpy();qd=L.qday.to_numpy()
            for idx in o:
                d=qd[idx];m=mo[idx];lst=acc.get(d)
                if lst is None: acc[d]=[m];keep[idx]=True;continue
                if all(abs(m-mm)>=HZ[hn] for mm in lst): lst.append(m);keep[idx]=True
            Li=L[keep];te=~Li.quarter.isin(TRAIN_Q) if 'quarter' in Li else ~Li.qtr.isin(TRAIN_Q)
            Lte=Li[te];dm,lo,hi=dayci(Lte.qday.to_numpy(),Lte.net.to_numpy())
            g=(Li.net+FEE).mean()
            cells.append((hn,thr,len(Li),Li.hit.mean(),g,Li.net.mean(),len(Lte),Lte.hit.mean() if len(Lte) else np.nan,lo,hi))
            if thr in (.30,.40):
                print(f"{hn:>4} ≤{thr:.2f} | {len(Li):>5} {Li.hit.mean():>5.3f} {g:>+6.1f} {Li.net.mean():>+6.1f} | "
                      f"{Lte.hit.mean() if len(Lte) else 0:>5.2f} [{lo:+.0f},{hi:+.0f}]")
    C=pd.DataFrame(cells,columns=['hz','thr','indN','hit','gross','net','teN','teHit','lo','hi'])
    surv=C[(C.gross>FEE)&(C.lo>0)&(C.teN>=5)]
    print(f"  Bonferroni {len(C)}. gross>fee & test CI>0 & teN>=5 생존: {len(surv)}")
    return C

def main():
    D=build()
    allC={}
    for nm in ['21dim(거울)','하락10단독','21+하락10']:
        R,align=search(D,nm)
        R=R.rename(columns={'qtr':'qtr'})
        allC[nm]=analyze(R,nm)
        if align:
            a=np.array([x[0] for x in align]);qd=np.array([x[1] for x in align])
            # 이웃 down 합의(1-fup) 높을수록 쿼리 실제 down 많나 (정렬)
            corr=np.corrcoef(1-a, (~qd).astype(float))[0,1]
            print(f"  [정렬] 이웃 down합의 vs 쿼리 실제down corr={corr:+.3f} (양수=정렬, 0=희석/무관)")
    print("\n현행 4h 상승 kNN hit 0.68. kNN(닮음)으로 하락 잡히면 방법문제, 안되면 시장본질 확정.")

if __name__=='__main__':
    main()
