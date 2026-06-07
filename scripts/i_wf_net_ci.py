#!/usr/bin/env python3
"""[I] 5단계 교정: CI 를 net(gross-11) 직접 + day-mean 일관 통계로. 이전 gross CI 인용 정정."""
import numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
R=pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
Rh=pd.read_parquet(f'{OUT}/lean70_v2_per_query_hfine.parquet')

def events(R,h,thr=.70):
    ok=(R[f'{h}_n']>=70)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
    s=R[ok]
    lean=(s[f'{h}_fup']>=thr)|(s[f'{h}_fup']<=1-thr)
    L=s[lean]
    sgn=np.where(L[f'{h}_fup']>=.5,1.,-1.)
    return L, sgn*L[f'{h}_frq'].to_numpy()*1e4

def daymean_ci(L,vals,alphas=(2.5,0.25),nb=6000,seed=7):
    dm=pd.Series(vals).groupby(L.qday.to_numpy()).mean().to_numpy()
    bs=np.random.default_rng(seed).choice(dm,(nb,len(dm)),replace=True).mean(axis=1)
    out=[dm.mean()]
    for a in alphas: out+= [np.percentile(bs,a),np.percentile(bs,100-a)]
    return out

print("h    n   evt_mean | day_mean | net95% [lo,hi] | netBonf99.5% [lo,hi]  (net=gross-11 직접)")
for src,h in [(R,'5m'),(R,'10m'),(Rh,'15m'),(Rh,'20m'),(R,'30m'),(Rh,'45m'),(R,'1h'),
              (Rh,'2h'),(Rh,'3h'),(R,'4h'),(Rh,'6h')]:
    L,ret=events(src,h)
    if len(L)<15: print(f"{h:4s} n={len(L)} (소표본)"); continue
    net=ret-11
    dmean,lo95,hi95,lo995,hi995=daymean_ci(L,net)
    tag95='✓' if lo95>0 else '✗'; tagB='✓' if lo995>0 else '✗'
    print(f"{h:4s} {len(L):4d} {net.mean():+7.1f} | {dmean:+7.1f} | [{lo95:+6.1f},{hi95:+6.1f}]{tag95} | [{lo995:+6.1f},{hi995:+6.1f}]{tagB}")
print()
print("결합 포트폴리오 (net 직접, day-mean):")
for name,hs,srcs,quarters in [
    ('30m+1h+4h 전체',['30m','1h','4h'],[R,R,R],None),
    ('45m+1h+4h 전체',['45m','1h','4h'],[Rh,R,R],None),
    ('30m+1h+4h 2025Q3~',['30m','1h','4h'],[R,R,R],['2025Q3','2025Q4','2026Q1','2026Q2']),
    ('1h+4h 2025Q3~',['1h','4h'],[R,R],['2025Q3','2025Q4','2026Q1','2026Q2'])]:
    Ls,rs=[],[]
    for h,src in zip(hs,srcs):
        L,ret=events(src,h)
        if quarters is not None:
            m=L.quarter.isin(quarters).to_numpy(); L=L[m]; ret=ret[m]
        Ls.append(L); rs.append(ret-11)
    Lc=pd.concat(Ls); net=np.concatenate(rs)
    dmean,lo95,hi95,lo995,hi995=daymean_ci(Lc,net)
    print(f"{name:22s} n={len(Lc):3d} ({Lc.qday.nunique()}일) evt {net.mean():+6.1f} day {dmean:+6.1f} "
          f"| 95% [{lo95:+6.1f},{hi95:+6.1f}]{'✓' if lo95>0 else '✗'} | 99.5% [{lo995:+6.1f},{hi995:+6.1f}]{'✓' if lo995>0 else '✗'}")
