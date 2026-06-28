#!/usr/bin/env python3
"""[I] 34단계 — 21단계(진짜) vs 31단계(환상) 충돌 정면 해소.
핵심: stage5 폴드 5/5 양수·CI 0 제외는 어떤 측정이었나 → stride-10 무dedup(모든 10번째 분).
  같은 폴드를 라이브 실제(dense earliest-crossing, 240블록)로 다시 → 5/5 유지되나 무너지나.
세 측정 나란히: A stride-10 무dedup(stage5) / B dense 무dedup / C dense earliest(라이브).
4h thr0.70 양방향(lean). day-cluster CI. lookahead 0."""
import numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
FEE=11.0; H=240
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']

def dayci(qd,net,seed=7,nb=6000):
    if len(net)<5: return np.nan,np.nan,np.nan
    dm=pd.Series(net).groupby(qd).mean().to_numpy()
    bs=np.random.default_rng(seed).choice(dm,(nb,len(dm)),replace=True).mean(1)
    return dm.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)

def lean_net(df):
    """양방향 lean (fup>=.70 롱 / <=.30 숏). net = dir*frq*1e4 - fee."""
    m=(df['4h_fup']>=.70)|(df['4h_fup']<=.30)
    L=df[m].copy(); dirn=np.where(L['4h_fup']>=.5,1,-1)
    L['net']=dirn*L['4h_frq'].to_numpy()*1e4-FEE
    L['hit']=(dirn*L['4h_frq'].to_numpy()>0)
    return L

def earliest_combined(df):
    """라이브: 어느 방향이든 신호면 240블록(MAX_CONCURRENT=1). 시간순 greedy."""
    m=(df['4h_fup']>=.70)|(df['4h_fup']<=.30)
    L=df[m].copy().sort_values(['qday','mod'])
    qd=L['qday'].to_numpy();mo=L['mod'].to_numpy()
    order=np.lexsort((mo,qd));acc={};keep=np.zeros(len(L),bool)
    for i in order:
        d=qd[i];mm=mo[i];lst=acc.get(d)
        if lst is None: acc[d]=[mm];keep[i]=True;continue
        if all(abs(mm-x)>=H for x in lst): lst.append(mm);keep[i]=True
    L=L[keep];dirn=np.where(L['4h_fup']>=.5,1,-1)
    L['net']=dirn*L['4h_frq'].to_numpy()*1e4-FEE
    L['hit']=(dirn*L['4h_frq'].to_numpy()>0)
    return L

def folds(L):
    L=L.copy(); L['fold']=L.quarter.str[:4]+np.where(L.quarter.str[5].astype(int)<=2,'H1','H2')
    out={}
    for fd in sorted(L.fold.unique()):
        s=L[L.fold==fd]
        # day-weighted net (stage5 와 동일 day-mean)
        dm=pd.Series(s['net'].to_numpy()).groupby(s.qday.to_numpy()).mean().mean()
        out[fd]=(len(s),dm)
    return out

def report(name,L):
    nd=L.qday.nunique()
    dmean,lo,hi=dayci(L.qday.to_numpy(),L['net'].to_numpy())
    fd=folds(L)
    npos=sum(1 for _,(n,v) in fd.items() if v>0)
    print(f"\n  [{name}] n={len(L)} ({nd}일) hit {L['hit'].mean():.3f} | 거래당 {L['net'].mean():+.1f} day-net {dmean:+.1f} CI[{lo:+.0f},{hi:+.0f}] {'(0 제외!)' if lo>0 else '(0 포함)'}")
    print(f"    반기 폴드 ({npos}/{len(fd)} 양수): "+" ".join(f"{k}{v:+.0f}({n})" for k,(n,v) in fd.items()))
    return npos,len(fd),lo

def main():
    A=pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')   # stride-10 (stage5 원본)
    A=A[(A['4h_n']>=70)&~A['4h_frq'].isna()&(A['4h_frq']!=0)]
    D=pd.read_parquet(f'{OUT}/lean70_v2_per_query_dense.parquet')
    nrm=pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    D['mod']=nrm['min_of_day'].to_numpy()[D['q'].to_numpy()]
    D=D[(D['4h_n']>=70)&~D['4h_frq'].isna()&(D['4h_frq']!=0)]
    print("===== 작업1·3: 같은 폴드, 세 측정 (stage5 stride-10 vs 라이브 earliest) =====")
    print("질문: stage5 의 5/5 양수·CI 0제외가 라이브 실제(earliest)서도 유지되나?")
    report("A stride-10 무dedup (stage5 원본)", lean_net(A))
    report("B dense 무dedup (매분)", lean_net(D))
    pe,fe,loe=report("C dense earliest (라이브 실제)", earliest_combined(D))

    print("\n===== 작업2: shadow 가 어느 측정에 해당하나 =====")
    print("  shadow 기록데몬(14414): 매분 fup>=0.70 다 기록(블록 없음) = B 무dedup 모집단(우호적 +53).")
    print("  라이브(54043): MAX_CONCURRENT=1 첫 신호만 = C earliest(≈본전).")
    print("  → shadow 우호적 6~7건 = B(무dedup) 모집단 추출 → 31 백테(C earliest)와 모순 아님.")

    print("\n===== 작업4: 거래당-가중 vs 일-가중 (−4.1 vs +5.3 정체) =====")
    Le=earliest_combined(D)
    print(f"  C earliest: 거래당평균 {Le['net'].mean():+.1f} (꼬리손실 영향) vs 일평균 {pd.Series(Le['net'].to_numpy()).groupby(Le.qday.to_numpy()).mean().mean():+.1f} (날 동일가중)")
    print(f"  방향 hit {Le['hit'].mean():.3f}(>0.5 약우위 실재). net p50 {np.percentile(Le['net'],50):+.0f}")

if __name__=='__main__':
    main()
