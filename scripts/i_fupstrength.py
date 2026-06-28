#!/usr/bin/env python3
"""[I] 31단계 확장 — fup 강도별 거래당 ("적게 크게", 본인 질문).
30/31: earliest-crossing(라이브) 전체 fup0.70 = 거래당 -4.1, OOS +0.26 marginal.
질문: 강한 fup(0.85+)만 골라 거래당↑ → marginal 극복? (레버리지=위험 아닌 선별).
  작업2 fup 강도별 거래당(단조?) ③ "적게 크게" 일수익 OOS ④ fup이 |move|+방향 동시 예측?
dense(매분) earliest-crossing causal 독립. 강할수록 표본↓(9단계 빈도붕괴) — n 명시. lookahead 0."""
import numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
FEE=11.0; H=240
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']

def dayci(qd,net,seed=7):
    if len(net)<5: return np.nan,np.nan,np.nan
    dm=pd.Series(net).groupby(qd).mean().to_numpy()
    bs=np.random.default_rng(seed).choice(dm,(4000,len(dm)),replace=True).mean(1)
    return dm.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)

def earliest(qd,mod,fup,thr):
    """fup>=thr 최초 진입(시간순), 같은 day 240분 비겹침. keep mask."""
    order=np.lexsort((mod,qd));acc={};keep=np.zeros(len(qd),bool)
    for i in order:
        if fup[i]<thr: continue
        d=qd[i];m=mod[i];lst=acc.get(d)
        if lst is None: acc[d]=[m];keep[i]=True;continue
        if all(abs(m-x)>=H for x in lst): lst.append(m);keep[i]=True
    return keep

def main():
    R=pd.read_parquet(f'{OUT}/lean70_v2_per_query_dense.parquet')
    nrm=pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    R['mod']=nrm['min_of_day'].to_numpy()[R['q'].to_numpy()]
    ok=(R['4h_n']>=70)&~R['4h_frq'].isna()&(R['4h_frq']!=0)
    s=R[ok].copy().sort_values(['qday','mod']).reset_index(drop=True)
    fup=s['4h_fup'].to_numpy();frq=s['4h_frq'].to_numpy();qd=s.qday.to_numpy();mod=s['mod'].to_numpy()
    qtr=s.quarter.to_numpy();net=frq*1e4-FEE;mv=np.abs(frq)*1e4
    te=~np.isin(qtr,TRAIN_Q);nd_all=len(np.unique(qd));nd_te=len(np.unique(qd[te]))
    print(f"[load] dense 롱후보 {len(s)}, 달력일 {nd_all}/te {nd_te}")
    print("작업1 요약(31단계): stride-10 +98.7=매분(+53.0) 91분위 행운. earliest(라이브)=−4.1, OOS +0.26.")

    print("\n===== 작업2: fup 강도별 거래당 (earliest-crossing entry, fup@진입 버킷) =====")
    keep70=earliest(qd,mod,fup,.70)
    fk=fup[keep70];nk=net[keep70];hk=(frq[keep70]>0);mvk=mv[keep70];qk=qd[keep70];tek=te[keep70]
    print(f"  earliest 진입 fup 분포: p50 {np.percentile(fk,50):.3f} p90 {np.percentile(fk,90):.3f} max {fk.max():.3f}")
    print(f"  {'fup@진입':>12} | {'n':>4} {'hit':>5} {'거래당':>7} {'|move|':>6} | {'teN':>4} te거래당")
    for lo,hi in [(.70,.75),(.75,.85),(.85,.95),(.95,1.01)]:
        m=(fk>=lo)&(fk<hi)
        if m.sum()<3: print(f"  [{lo:.2f},{hi:.2f}) | n={m.sum()} 표본부족"); continue
        tn=(tek&m).sum()
        print(f"  [{lo:.2f},{hi:.2f}) | {m.sum():>4} {hk[m].mean():>5.3f} {nk[m].mean():>+7.1f} {mvk[m].mean():>6.0f} | "
              f"{tn:>4} {nk[tek&m].mean() if tn else float('nan'):>+7.1f}")
    print("  → earliest 는 막 0.70 넘은 순간이라 강fup@진입 드묾(위 n). 강fup 거래당 큰가/표본 충분한가.")

    print("\n===== 작업3: '적게 크게' — 진입 thr 올림 (강한 fup만, causal earliest) =====")
    print(f"  {'진입thr':>7} | {'n':>4} {'hit':>5} {'거래당':>7} {'빈도/일':>7} {'일수익':>7} | te 거래당/일수익 CI")
    for thr in [.70,.75,.80,.85,.90]:
        k=earliest(qd,mod,fup,thr);n=k.sum()
        if n<5: print(f"  {thr:.2f} | n={n} 표본부족"); continue
        tn=(k&te).sum()
        dm,lo2,hi2=dayci(qd[k&te],net[k&te])
        td=net[k&te].sum()/nd_te if tn else float('nan')
        print(f"  {thr:.2f} | {n:>4} {(frq[k]>0).mean():>5.3f} {net[k].mean():>+7.1f} {n/nd_all:>7.3f} {net[k].sum()/nd_all:>+7.2f} | "
              f"te{tn} {net[k&te].mean() if tn else float('nan'):>+6.1f}/{td:>+5.2f} [{lo2:+.0f},{hi2:+.0f}]")
    print("  → 강fup 일수익(빈도↓×거래당↑)이 전체0.70 대비 오르나. OOS 살아있나(표본?).")

    print("\n===== 작업4: fup 강도가 |move|+방향 동시 예측? (무dedup, 표본 확보) =====")
    print(f"  {'fup':>12} | {'n':>5} {'hit(방향)':>8} {'|move|bp':>8} {'net':>6}")
    for lo,hi in [(.70,.72),(.72,.75),(.75,.80),(.80,.90),(.90,1.01)]:
        m=(fup>=lo)&(fup<hi)
        if m.sum()<20: continue
        print(f"  [{lo:.2f},{hi:.2f}) | {m.sum():>5} {(frq[m]>0).mean():>8.3f} {mv[m].mean():>8.0f} {net[m].mean():>+6.1f}")
    # OOS: 고fup 무dedup 도 OOS 살아있나 (방향+크기)
    print("  OOS(test) 고fup 무dedup:")
    for lo in [.75,.80]:
        m=(fup>=lo)&te
        if m.sum()<10: print(f"    fup>={lo}: teN={m.sum()} 표본부족"); continue
        print(f"    fup>={lo}: teN={m.sum()} hit {(frq[m]>0).mean():.3f} |move| {mv[m].mean():.0f}bp net {net[m].mean():+.1f}")
    print("\n  → fup↑ 가 hit(방향)+|move|(크기) 둘 다 올리면 '방향+큰움직임' 신호. OOS 살아야 거래 가능.")

if __name__=='__main__':
    main()
