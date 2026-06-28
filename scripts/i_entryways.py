#!/usr/bin/env python3
"""[I] 32단계 — 라이브를 백테처럼 거래 가능한가 + 31 맞나 틀렸나.
선행 발견: 데몬fup vs 백테fup 4995분 corr 0.961, 평균차 0.000 → 같은 신호(31 틀리지 않음).
  단 0.70 경계서 ±0.01 노이즈로 어느 분 잡힐지 흔들림 → shadow 우호적=소표본 행운.
여기: causal 진입 방식 다 해보기(earliest/높은기준/지속/상승) + 분포(평균함정) + 진짜 edge CI.
미래 보는 방식(lookahead) 제외. dense(매분 2024+). lookahead 0."""
import numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
FEE=11.0; H=240
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']

def dayci(qd,net,seed=7):
    if len(net)<5: return np.nan,np.nan,np.nan
    dm=pd.Series(net).groupby(qd).mean().to_numpy()
    bs=np.random.default_rng(seed).choice(dm,(4000,len(dm)),replace=True).mean(1)
    return dm.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)

def greedy(qd,mod,elig):
    order=np.lexsort((mod,qd));acc={};keep=np.zeros(len(qd),bool)
    for i in order:
        if not elig[i]: continue
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
    qtr=s.quarter.to_numpy();net=frq*1e4-FEE;te=~np.isin(qtr,TRAIN_Q)
    nd_all=len(np.unique(qd));nd_te=len(np.unique(qd[te]))
    # 직전분 fup (상승 판정용)
    fup_prev=np.full(len(s),np.nan)
    for i in range(1,len(s)):
        if qd[i]==qd[i-1] and mod[i]==mod[i-1]+1: fup_prev[i]=fup[i-1]
    # 지속(런길이)
    runlen=np.ones(len(s),int)
    for i in range(1,len(s)):
        if qd[i]==qd[i-1] and mod[i]==mod[i-1]+1 and fup[i]>=.70 and fup[i-1]>=.70: runlen[i]=runlen[i-1]+1
    print(f"[load] dense 롱후보 {len(s)}, 달력일 {nd_all}/te {nd_te}")
    print("작업1·2: 데몬fup vs 백테fup 4995분 corr 0.961·평균차 0.000 = 같은 신호(31 안 틀림).")
    print("  0.70 경계 ±0.01 노이즈로 어느 분 잡힐지 흔들림 → shadow 우호적 6~7건=소표본+경계노이즈 행운.")

    print("\n===== 작업3: causal 진입 방식 다 해보기 (240블록, OOS) =====")
    print(f"{'진입방식':>16} | {'n':>4} {'hit':>5} {'mean':>6} {'median':>6} {'win%':>5} {'일수익':>6} | te net/일수익 CI")
    ways={
      'earliest 0.70(현행)': fup>=.70,
      '높은기준 0.72': fup>=.72,
      '높은기준 0.75': fup>=.75,
      '지속3분(runlen>=3)': (fup>=.70)&(runlen>=3),
      '지속5분(runlen>=5)': (fup>=.70)&(runlen>=5),
      '상승중(fup>직전)': (fup>=.70)&(fup>fup_prev),
      '상승+0.72': (fup>=.72)&(fup>fup_prev),
    }
    for nm,elig in ways.items():
        keep=greedy(qd,mod,elig);n=keep.sum()
        if n<5: print(f"{nm:>16} | n={n} 표본부족"); continue
        nk=net[keep];hk=(frq[keep]>0)
        tn=(keep&te).sum();dm,lo,hi=dayci(qd[keep&te],net[keep&te])
        td=net[keep&te].sum()/nd_te if tn else np.nan
        print(f"{nm:>16} | {n:>4} {hk.mean():>5.3f} {nk.mean():>+6.1f} {np.median(nk):>+6.1f} {(nk>0).mean()*100:>4.0f}% {nk.sum()/nd_all:>+6.2f} | "
              f"te{tn} {net[keep&te].mean() if tn else np.nan:>+6.1f}/{td:>+5.2f} [{lo:+.0f},{hi:+.0f}]")

    print("\n===== 작업4: earliest 분포 (평균함정 — 방향 맞나, 왜 net≈0) =====")
    keep=greedy(qd,mod,fup>=.70);nk=net[keep];gk=frq[keep]*1e4;mvk=np.abs(gk)
    print(f"  earliest n={keep.sum()}: hit(방향) {(gk>0).mean():.3f}(>0.5=방향 우위 실재)")
    print(f"  gross 평균 {gk.mean():+.1f}bp vs fee {FEE} → net {nk.mean():+.1f}")
    print(f"  |move| 평균 {mvk.mean():.0f}bp. 방향우위 {(gk>0).mean()-0.5:+.3f} × |move| ≈ {((gk>0).mean()-0.5)*2*mvk.mean():.0f}bp gross 와 fee 가 거의 상쇄")
    print(f"  net 분위: p10 {np.percentile(nk,10):+.0f} p50 {np.percentile(nk,50):+.0f} p90 {np.percentile(nk,90):+.0f} (꼬리 큼)")
    qd_e=qd[keep]
    dm,lo,hi=dayci(qd_e,nk); dmt,lot,hit_=dayci(qd[keep&te],net[keep&te])
    print(f"  전체 거래당 {dm:+.1f} CI[{lo:+.0f},{hi:+.0f}] | OOS {dmt:+.1f} CI[{lot:+.0f},{hit_:+.0f}]")
    print("\n  '백테 +90.5'=stride-10 무dedup 91분위 행운. 무dedup 매분 +53.0은 겹치는 4h(상관)=레버리지.")
    # 무dedup 겹침 비율
    L=fup>=.70; dd=qd[L];mm=mod[L];order=np.argsort(dd*10000+mm)
    overlap=0
    for i in range(1,len(order)):
        a,b=order[i-1],order[i]
        if dd[a]==dd[b] and abs(mm[a]-mm[b])<240: overlap+=1
    print(f"  무dedup {L.sum()}건 중 직전과 같은날 240분내 겹침 {overlap} = {overlap/L.sum()*100:.0f}% (독립 아님=상관/레버리지)")

if __name__=='__main__':
    main()
