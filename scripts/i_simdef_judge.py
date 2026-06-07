#!/usr/bin/env python3
"""[I] 6단계 판정: 진출 정의 OOS(2025Q3~) vs 베이스라인. 사전등록 기준."""
import numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
H=['30m','1h','4h']
TEST_Q=['2025Q3','2025Q4','2026Q1','2026Q2']

def events(R,h,thr=.70,fee=11):
    ok=(R[f'{h}_n']>=70)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
    s=R[ok]
    lean=(s[f'{h}_fup']>=thr)|(s[f'{h}_fup']<=1-thr)
    L=s[lean]
    sgn=np.where(L[f'{h}_fup']>=.5,1.,-1.)
    return L, sgn*L[f'{h}_frq'].to_numpy()*1e4-fee

def dmci(L,vals,alphas=(2.5,0.08),nb=6000):
    dm=pd.Series(vals).groupby(L.qday.to_numpy()).mean().to_numpy()
    bs=np.random.default_rng(7).choice(dm,(nb,len(dm)),replace=True).mean(axis=1)
    o=[dm.mean()]
    for a in alphas: o+=[np.percentile(bs,a),np.percentile(bs,100-a)]
    return o

# 베이스라인 = stage5 v2 parquet (동일 파이프라인)
base=pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
srcs={'BASELINE(21d)':base}
for d in ['V','VOC','O']:
    srcs[d]=pd.read_parquet(f'{OUT}/simdef/def_{d}_p2.parquet')
# simdef parquet 에 quarter 있음; v2 에도 있음
print("===== OOS(2025Q3~) thr70 결합 판정 (day-mean net, T+T 11bp) =====")
print("기준: (a) 베이스라인 초과 (b) 95%>0 후보 / Bonf-31(99.84%)>0 확정")
rows=[]
for name,R in srcs.items():
    Rt=R[R.quarter.isin(TEST_Q)]
    Ls,vs=[],[]
    for h in H:
        L,v=events(Rt,h)
        Ls.append(L); vs.append(v)
    Lc=pd.concat(Ls); vc=np.concatenate(vs)
    if len(Lc)<5:
        print(f"{name:14s} OOS n={len(Lc)} (판정불가)"); continue
    dmean,lo95,hi95,loB,hiB=dmci(Lc,vc)
    tdays=Rt.qday.nunique()
    daily=vc.sum()/tdays
    rows.append((name,len(Lc),Lc.qday.nunique(),dmean,lo95,hi95,loB,hiB,daily))
    print(f"{name:14s} n={len(Lc):3d}({Lc.qday.nunique():3d}일) net {dmean:+7.1f} | 95%[{lo95:+7.1f},{hi95:+7.1f}]"
          f"{'✓' if lo95>0 else '✗'} | Bonf[{loB:+7.1f},{hiB:+7.1f}]{'✓' if loB>0 else '✗'} | 일수익 {daily:+.1f}bp/day")
print()
print("===== horizon 별 OOS net (점추정) =====")
for name,R in srcs.items():
    Rt=R[R.quarter.isin(TEST_Q)]
    line=f"{name:14s} "
    for h in H:
        L,v=events(Rt,h)
        line+=f"{h} n={len(L):3d} {v.mean() if len(v)>4 else float('nan'):+7.1f} | "
    print(line)
print()
print("===== 전체기간(2024+) 결합 net (참고 — train 오염 아님, 동일 사전지정 룰) =====")
for name,R in srcs.items():
    Ls,vs=[],[]
    for h in H:
        L,v=events(R,h)
        Ls.append(L); vs.append(v)
    Lc=pd.concat(Ls); vc=np.concatenate(vs)
    dmean,lo95,hi95,loB,hiB=dmci(Lc,vc)
    print(f"{name:14s} n={len(Lc):3d} net {dmean:+6.1f} | 95%[{lo95:+6.1f},{hi95:+6.1f}]{'✓' if lo95>0 else '✗'} "
          f"| Bonf[{loB:+6.1f},{hiB:+6.1f}]{'✓' if loB>0 else '✗'} | 일수익(851d) {vc.sum()/R.qday.nunique():+.1f}bp/day")
