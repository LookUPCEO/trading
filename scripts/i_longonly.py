#!/usr/bin/env python3
"""[I] 30단계 — 롱 전용 vs 롱+숏 (숏 제거 효과).
질문: 라이브=양방향(fup>=.70 롱 / fup<=.30 숏). 숏은 25~28-1 6단계로 약함/예측불가 확정.
  숏 빼면 거래당↑(숏 음수)이나 빈도↓ → 일수익 순효과 계산.
데이터: dense(stride=1=매분=라이브 운영점, 825k). 25단계 stride10 숏 4h(+60.9 n17)는 플룩 —
  dense 숏(n~190 코인플립)이 정직. causal 독립(240분 비겹침), 4h thr0.70.
지표: 거래당 net, 빈도(이벤트/달력일), 일수익(=거래당×빈도=총net/일수), OOS train→test, day-CI.
일수익 denom = 달력일수(매일 운영, 대부분 무거래). lookahead 0."""
import numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
FEE=11.0; H=240
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']

def causal_indep(L):
    o=np.lexsort((L['mod'].to_numpy(),L.qday.to_numpy()))
    acc={};keep=np.zeros(len(L),bool);mo=L['mod'].to_numpy();qd=L.qday.to_numpy()
    for idx in o:
        d=qd[idx];m=mo[idx];lst=acc.get(d)
        if lst is None: acc[d]=[m];keep[idx]=True;continue
        if all(abs(m-mm)>=H for mm in lst): lst.append(m);keep[idx]=True
    return L[keep]

def dayci(qd,net,seed=7):
    if len(net)<5: return np.nan,np.nan,np.nan
    dm=pd.Series(net).groupby(qd).mean().to_numpy()
    bs=np.random.default_rng(seed).choice(dm,(4000,len(dm)),replace=True).mean(1)
    return dm.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)

def main():
    R=pd.read_parquet(f'{OUT}/lean70_v2_per_query_dense.parquet')
    nrm=pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    R['mod']=nrm['min_of_day'].to_numpy()[R['q'].to_numpy()]
    ok=(R['4h_n']>=70)&~R['4h_frq'].isna()&(R['4h_frq']!=0)
    s=R[ok].copy()
    # 달력일수 (전체 / test) — 매일 운영 가정 denom
    nd_all=s.qday.nunique(); nd_te=s[~s.quarter.isin(TRAIN_Q)].qday.nunique()
    print(f"[load] dense 4h 후보 {len(s)} (n>=70), 달력일 전체 {nd_all} / test {nd_te}")

    def stratum(mask, name):
        L=s[mask].copy()
        dirn=np.where(L['4h_fup']>=.5,1,-1)
        L['net']=dirn*L['4h_frq'].to_numpy()*1e4-FEE
        L['hit']=(dirn*L['4h_frq'].to_numpy()>0)
        Li=causal_indep(L)
        te=~Li.quarter.isin(TRAIN_Q)
        full=dict(n=len(Li),hit=Li.hit.mean(),net=Li.net.mean(),
                  freq=len(Li)/nd_all, daily=Li.net.sum()/nd_all)
        T=Li[te]
        dm,lo,hi=dayci(T.qday.to_numpy(),T.net.to_numpy())
        full.update(te_n=len(T),te_hit=T.hit.mean() if len(T) else np.nan,
                    te_net=T.net.mean() if len(T) else np.nan,
                    te_daily=T.net.sum()/max(nd_te,1),te_lo=lo,te_hi=hi)
        return Li,full

    # 방법론 정합 (진입타이밍/stride 민감도) — 정직성: dense-causal=라이브 실제 방식
    print("\n===== 방법론 정합: A stride-10(검증op-pt) / B dense무dedup / C dense-causal(=라이브) =====")
    A=pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
    okA=(A['4h_n']>=70)&~A['4h_frq'].isna()&(A['4h_frq']!=0);sA=A[okA]
    def quick(df,longside):
        m=(df['4h_fup']>=.70) if longside else (df['4h_fup']<=.30)
        L=df[m];dirn=1 if longside else -1;net=dirn*L['4h_frq'].to_numpy()*1e4-FEE
        return len(L),(dirn*L['4h_frq'].to_numpy()>0).mean(),net.mean()
    for nm,ls in [('롱',True),('숏',False)]:
        a=quick(sA,ls);b=quick(s,ls)
        print(f"  {nm}: A stride10 n{a[0]} hit{a[1]:.3f} {a[2]:+.1f} | B dense무dedup n{b[0]} hit{b[1]:.3f} {b[2]:+.1f}")
    print("  (C dense-causal=아래 작업1. A 숏 +60.9 n17=플룩, B 숏 음수 n249. 진입 earliest-cross 가 신호 약화)")

    Llong,fl=stratum((s['4h_fup']>=.70).to_numpy(),'롱')
    Lshort,fsh=stratum((s['4h_fup']<=.30).to_numpy(),'숏')
    Lboth,fb=stratum(((s['4h_fup']>=.70)|(s['4h_fup']<=.30)).to_numpy(),'양방향')

    print("\n===== 작업1: 롱 전용 vs 양방향 (dense, causal 독립, 4h thr0.70) =====")
    print(f"{'전략':>8} | {'n':>5} {'hit':>5} {'거래당net':>8} {'빈도/일':>7} {'일수익bp':>8} | OOS(test) net/일수익/CI")
    for nm,f in [('롱전용',fl),('양방향(현행)',fb),('숏전용',fsh)]:
        print(f"{nm:>8} | {f['n']:>5} {f['hit']:>5.3f} {f['net']:>+8.1f} {f['freq']:>7.3f} {f['daily']:>+8.3f} | "
              f"te {f['te_net']:>+7.1f} / {f['te_daily']:>+6.2f}bp/일 [{f['te_lo']:+.0f},{f['te_hi']:+.0f}]")

    print("\n===== 작업2: 숏 기여 분해 =====")
    print(f"  숏 전용: n={fsh['n']} hit {fsh['hit']:.3f} 거래당 {fsh['net']:+.1f}bp (음수=깎음)")
    print(f"  숏 비중: 전체 거래 {fb['n']} 중 숏 {fsh['n']} = {fsh['n']/fb['n']*100:.0f}%")
    print(f"  숏이 일수익에 더하는 기여: {fsh['daily']:+.3f}bp/일 (양방향 {fb['daily']:.3f} = 롱 {fl['daily']:.3f} + 숏 {fsh['daily']:.3f})")
    print(f"  → 숏 제거 시 일수익 변화: {fl['daily']:.3f} - {fb['daily']:.3f} = {fl['daily']-fb['daily']:+.3f}bp/일")

    print("\n===== 작업3: 롱 전용 일수익 (핵심) =====")
    print(f"  롱 전용 전체: 거래당 {fl['net']:+.1f}bp · 빈도 {fl['freq']:.3f}/일 · 일수익 {fl['daily']:+.3f}bp/일")
    print(f"  양방향 전체: 거래당 {fb['net']:+.1f}bp · 빈도 {fb['freq']:.3f}/일 · 일수익 {fb['daily']:+.3f}bp/일")
    print(f"  OOS(test): 롱 {fl['te_daily']:+.2f}bp/일 vs 양방향 {fb['te_daily']:+.2f}bp/일")
    win_full = fl['daily']>fb['daily']; win_te = fl['te_daily']>fb['te_daily']
    print(f"  → 롱 전용이 일수익 더 높나: 전체 {'YES' if win_full else 'NO'} / OOS {'YES' if win_te else 'NO'}")
    print(f"  (숏 빼면 빈도 {fb['freq']:.3f}→{fl['freq']:.3f}/일 = {(1-fl['freq']/fb['freq'])*100:.0f}% 감소)")

    # 시기별 (롱 vs 양방향 거래당)
    print("\n  시기별 거래당 net (롱 / 양방향):")
    for y in [2024,2025,2026]:
        lm=Llong[Llong.qyr==y]; bm=Lboth[Lboth.qyr==y]
        if len(lm)>=5: print(f"    {y}: 롱 n={len(lm)} {lm.net.mean():+.1f}bp | 양방향 n={len(bm)} {bm.net.mean():+.1f}bp")

    print("\n===== 작업4: 라이브 반영 검토 =====")
    print(f"  롱 전용 OOS 일수익 {fl['te_daily']:+.2f} vs 양방향 {fb['te_daily']:+.2f}bp/일.")
    print(f"  숏 거래당 {fsh['net']:+.1f}bp (25~28-1 숏 약함 일관). 숏 제거 = 거래당↑·일수익 {'↑' if win_full else '≈'}.")
    print(f"  one-way 단일포지션이라 롱전용=구현 단순화. 라이브 반영은 OOS 확실+과적합 아닐 때만.")

if __name__=='__main__':
    main()
