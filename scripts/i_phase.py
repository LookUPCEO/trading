#!/usr/bin/env python3
"""[I] 31단계 — earliest-crossing 재검토: 0.084%가 진짜인가 측정 인공물인가.
30단계 발견: 검증 헤드라인(4h 롱 +90.5bp stride-10)이 라이브 실제(매분 earliest-crossing)론 -4.1.
가름: ① stride 영향(10분 띄엄 vs 매분) ② 진입 phase(earliest vs 구간 깊이/확인후)
  ③ 진짜 edge 추정 ④ 진입 개선 여지(causal만). dense(매분 2024+) 사용. lookahead 0."""
import numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
FEE=11.0; H=240
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']

def dayci(qd,net,seed=7):
    if len(net)<5: return np.nan,np.nan,np.nan
    dm=pd.Series(net).groupby(qd).mean().to_numpy()
    bs=np.random.default_rng(seed).choice(dm,(4000,len(dm)),replace=True).mean(1)
    return dm.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)

def causal_greedy(qd,mod,eligible):
    """eligible(bool) 중 시간순 earliest, 같은 day 240분 비겹침. keep mask 반환."""
    order=np.lexsort((mod,qd))
    acc={};keep=np.zeros(len(qd),bool)
    for i in order:
        if not eligible[i]: continue
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
    fup=s['4h_fup'].to_numpy(); frq=s['4h_frq'].to_numpy(); qd=s.qday.to_numpy(); mod=s['mod'].to_numpy()
    qtr=s.quarter.to_numpy(); netlong=frq*1e4-FEE   # 롱 net
    te=~np.isin(qtr,TRAIN_Q)
    nd_all=len(np.unique(qd)); nd_te=len(np.unique(qd[te]))
    print(f"[load] dense 롱후보 {len(s)} (4h n>=70), 달력일 {nd_all}/te {nd_te}")

    print("\n===== 작업1: stride 영향 (롱 4h thr0.70, 무dedup) — mod%10 잔차별 =====")
    L=fup>=.70
    print(f"  전체(매분) n={L.sum()} hit {(frq[L]>0).mean():.3f} net {netlong[L].mean():+.1f}")
    res=[]
    for r in range(10):
        m=L&(mod%10==r)
        res.append(netlong[m].mean());
    print("  mod%10 잔차별 net:", " ".join(f"{r}:{v:+.0f}" for r,v in enumerate(res)))
    print(f"  잔차 평균 {np.mean(res):+.1f} std {np.std(res):.1f} | mod5(stride-10 parquet) {res[5]:+.1f}")
    # 부트스트랩: 매분에서 n=62(stride-10 크기) 뽑으면 +98.7 가 흔한가
    rng=np.random.default_rng(1); idxL=np.where(L)[0]
    boot=[netlong[rng.choice(idxL,62,replace=False)].mean() for _ in range(2000)]
    print(f"  매분서 n=62 랜덤추출 net 분포: 중앙 {np.median(boot):+.1f}, 95%상한 {np.percentile(boot,97.5):+.1f}, +98.7 분위 {(np.array(boot)<98.7).mean()*100:.0f}%")

    print("\n===== 작업2: 진입 phase — fup 레벨별 + 진입 시점별 (핵심) =====")
    print("  (A) 무dedup 진입 fup 레벨별 net (높은 fup=강한가):")
    for lo,hi in [(.70,.72),(.72,.75),(.75,.80),(.80,.90),(.90,1.01)]:
        m=(fup>=lo)&(fup<hi)
        if m.sum()>20: print(f"    fup[{lo:.2f},{hi:.2f}): n={m.sum()} hit {(frq[m]>0).mean():.3f} net {netlong[m].mean():+.1f}")
    # run-length: 같은 day 연속(mod 인접) fup>=0.70 런 길이 (causal: 진입 전까지 지속분)
    runlen=np.ones(len(s),int)
    for i in range(1,len(s)):
        if qd[i]==qd[i-1] and mod[i]==mod[i-1]+1 and fup[i]>=.70 and fup[i-1]>=.70:
            runlen[i]=runlen[i-1]+1
    print("  (B) earliest(런 1분차) vs 지속확인후 진입 (causal, runlen=확인분):")
    for R_ in [1,3,5,10,15]:
        elig=(fup>=.70)&(runlen>=R_)
        keep=causal_greedy(qd,mod,elig)
        n=keep.sum()
        if n<5: print(f"    confirm{R_-1}분: n={n} (표본부족)"); continue
        dm,lo2,hi2=dayci(qd[keep&te],netlong[keep&te])
        print(f"    confirm{R_-1}분(runlen>={R_}): n={n} hit {(frq[keep]>0).mean():.3f} net {netlong[keep].mean():+.1f} | "
              f"te net {netlong[keep&te].mean():+.1f} 일수익 {netlong[keep&te].sum()/nd_te:+.2f} CI[{lo2:+.0f},{hi2:+.0f}]")

    print("\n===== 작업3: thr 올려 진입(causal earliest, dense) — 강한 신호만 =====")
    print(f"  {'thr':>5} | {'n':>4} {'hit':>5} {'net':>6} 빈도/일 일수익 | te net/일수익 CI")
    for thr in [.70,.72,.75,.78,.80,.85]:
        elig=fup>=thr
        keep=causal_greedy(qd,mod,elig)
        n=keep.sum()
        if n<5: print(f"  {thr:.2f} | n={n} 표본부족"); continue
        dm,lo2,hi2=dayci(qd[keep&te],netlong[keep&te])
        print(f"  {thr:.2f} | {n:>4} {(frq[keep]>0).mean():>5.3f} {netlong[keep].mean():>+6.1f} {n/nd_all:>6.3f} {netlong[keep].sum()/nd_all:>+6.2f} | "
              f"{netlong[keep&te].mean():>+6.1f}/{netlong[keep&te].sum()/nd_te:>+5.2f} [{lo2:+.0f},{hi2:+.0f}]")

    print("\n===== 작업4: 비-causal 상한 참고 (구간 최고 fup 진입 = 못 하는 진입) =====")
    # 240창 내 최고 fup 분 진입(미래 봄=비causal) — phase 효과 상한
    keep_e=causal_greedy(qd,mod,fup>=.70)
    print(f"  earliest(라이브 실제): n={keep_e.sum()} net {netlong[keep_e].mean():+.1f}")
    print("  → confirm/thr 로 causal 개선되면 라이브 진입 바꿀 여지. 안 되면 earliest 가 한계.")
    print("\nshadow 실측 6건 롱 gross: +1.5/+158.9/+95.5/-22.7/+7.6 (4/5 양수, 평균 +48 — 소표본 우호적)")

if __name__=='__main__':
    main()
