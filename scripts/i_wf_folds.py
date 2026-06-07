#!/usr/bin/env python3
"""[I] 5단계 작업1~4: 폴드 일관성 + 누적 OOS CI(+Bonferroni) + thr 촘촘 단조."""
import numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
R=pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
H=['5m','10m','30m','1h','4h']

def events(R,h,thr):
    ok=(R[f'{h}_n']>=70)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
    s=R[ok]
    lean=(s[f'{h}_fup']>=thr)|(s[f'{h}_fup']<=1-thr)
    L=s[lean]
    sgn=np.where(L[f'{h}_fup']>=.5,1.,-1.)
    ret=sgn*L[f'{h}_frq'].to_numpy()*1e4
    return L,ret

def day_ci(L,ret,alpha=2.5,nb=4000,seed=7):
    if len(L)<5: return np.nan,np.nan
    dm=pd.Series(ret).groupby(L.qday.to_numpy()).mean().to_numpy()
    bs=np.random.default_rng(seed).choice(dm,(nb,len(dm)),replace=True).mean(axis=1)
    return np.percentile(bs,alpha),np.percentile(bs,100-alpha)

print("===== 작업1+2: 반기 폴드 (모든 쿼리가 자기 pool 에 OOS — 룰 사전지정) =====")
R['fold']=R.quarter.str[:4]+np.where(R.quarter.str[5].astype(int)<=2,'H1','H2')
folds=sorted(R.fold.unique())
for h in ['30m','1h','4h']:
    line=f"{h:3s}: "
    pos=0;tot=0
    for fd in folds:
        L,ret=events(R[R.fold==fd],h,.70)
        if len(L)<5: line+=f"{fd} n<5 | "; continue
        tot+=1; pos+=(ret.mean()-11>0)
        line+=f"{fd} n={len(L)} net{ret.mean()-11:+.0f} | "
    print(line+f"  → 폴드 양수 {pos}/{tot}")

print("\n===== 작업3: 누적 OOS CI (전 2024+ = OOS; 95% + Bonferroni 10셀 99.5%) =====")
for h in ['30m','1h','4h']:
    L,ret=events(R,h,.70)
    lo95,hi95=day_ci(L,ret)
    lo995,hi995=day_ci(L,ret,alpha=0.25)
    print(f"{h:3s}: n={len(L)} ({L.qday.nunique()}일) gross {ret.mean():+.1f} net(T+T) {ret.mean()-11:+.1f} "
          f"| 95% [{lo95:+.1f},{hi95:+.1f}] | Bonf 99.5% [{lo995:+.1f},{hi995:+.1f}] (0 제외 여부)")
# 결합 (3 horizon 합산 포트폴리오 — 같은 쿼리 중복 이벤트는 별개 거래로)
Ls=[];rets=[]
for h in ['30m','1h','4h']:
    L,ret=events(R,h,.70); Ls.append(L); rets.append(ret)
Lc=pd.concat(Ls); rc=np.concatenate(rets)
lo,hi=day_ci(Lc,rc); lo2,hi2=day_ci(Lc,rc,alpha=0.25)
print(f"결합: n={len(Lc)} ({Lc.qday.nunique()}일) gross {rc.mean():+.1f} net {rc.mean()-11:+.1f} "
      f"| 95% [{lo:+.1f},{hi:+.1f}] | 99.5% [{lo2:+.1f},{hi2:+.1f}]")
# 2025Q3+ 만 결합 (이전 'test' 구간 한정 결합)
tq=['2025Q3','2025Q4','2026Q1','2026Q2']
m=Lc.quarter.isin(tq).to_numpy()
lo,hi=day_ci(Lc[m],rc[m])
print(f"결합(2025Q3~ 한정): n={int(m.sum())} gross {rc[m].mean():+.1f} net {rc[m].mean()-11:+.1f} | 95% [{lo:+.1f},{hi:+.1f}]")

print("\n===== 작업4: thr 촘촘 (0.60~0.78) — 매끄러운 단조인가 톱니인가 =====")
print("thr  | " + " | ".join(f"{h:>22s}" for h in ['30m','1h','4h']))
for thr in [.60,.62,.64,.66,.68,.70,.72,.74,.76,.78]:
    line=f"{thr:.2f} |"
    for h in ['30m','1h','4h']:
        L,ret=events(R,h,thr)
        if len(L)<10: line+=f"   n={len(L):4d}    -     |"; continue
        line+=f" n={len(L):4d} hit{(ret>0).mean():.2f} g{ret.mean():+5.1f} |"
    print(line)
