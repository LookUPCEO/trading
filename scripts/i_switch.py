#!/usr/bin/env python3
"""[I] 33단계 — 탄력적 진입 (강해지면 갈아타기). 본인 의도: 6-1/6-2(나빠지면 청산)와 다름 —
"같은 방향 더 강해지면(fup↑) 청산+재진입(업그레이드)". 31단계 강영역(+134)을 잡되 수수료 2배.
작업1 진입 후 hold 중 fup 더 오르나(0.85 도달 빈도) ② 갈아타기 규칙(causal) ③ 이득인가(수수료 2배 넘나) OOS.
dense(매분) earliest 0.70 진입. 미래 안 봄(현재 fup으로만 결정, 같은 방향). cross-day 가격. lookahead 0."""
import numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB='/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
FEE=11.0; H=240
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']

def dayci(qd,net,seed=7):
    if len(net)<5: return np.nan,np.nan,np.nan
    dm=pd.Series(net).groupby(qd).mean().to_numpy()
    bs=np.random.default_rng(seed).choice(dm,(4000,len(dm)),replace=True).mean(1)
    return dm.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)

def main():
    R=pd.read_parquet(f'{OUT}/lean70_v2_per_query_dense.parquet')
    nrm=pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    R['mod']=nrm['min_of_day'].to_numpy()[R['q'].to_numpy()]
    days=sorted(nrm['day'].unique()); dix={d:i for i,d in enumerate(days)}; ndays=len(days)
    R['dayi']=R['qday'].to_numpy()
    # fup 그리드 (n>=70 인 분만 유효; 갈아타기 결정용)
    fupG=np.full(ndays*1440,np.nan,np.float32)
    okn=R['4h_n']>=70
    fupG[R.loc[okn,'dayi'].to_numpy()*1440+R.loc[okn,'mod'].to_numpy()]=R.loc[okn,'4h_fup'].to_numpy()
    # cross-day 가격
    full=pd.read_parquet(LAB,columns=['day','min_of_day','mid']); full=full[full.day.isin(days)]
    P=np.full(ndays*1440,np.nan,np.float32)
    P[full['day'].map(dix).to_numpy()*1440+full['min_of_day'].to_numpy()]=full['mid'].to_numpy(np.float32)
    NG=len(P)
    # earliest 0.70 진입 (causal greedy 240블록)
    s=R[(R['4h_n']>=70)&~R['4h_frq'].isna()&(R['4h_frq']!=0)].copy().sort_values(['dayi','mod'])
    fup=s['4h_fup'].to_numpy();qd=s['dayi'].to_numpy();mod=s['mod'].to_numpy();qtr=s.quarter.to_numpy()
    order=np.lexsort((mod,qd));acc={};keep=np.zeros(len(s),bool)
    for i in order:
        if fup[i]<.70: continue
        d=qd[i];m=mod[i];lst=acc.get(d)
        if lst is None: acc[d]=[m];keep[i]=True;continue
        if all(abs(m-x)>=H for x in lst): lst.append(m);keep[i]=True
    E=s[keep].copy(); Eqd=E['dayi'].to_numpy(); Emod=E['mod'].to_numpy(); Eqtr=E.quarter.to_numpy()
    gm=Eqd*1440+Emod
    te=~np.isin(Eqtr,TRAIN_Q); nd_all=len(np.unique(Eqd)); nd_te=len(np.unique(Eqd[te]))
    dirn=1  # 롱 (fup>=0.70)
    print(f"[load] earliest 0.70 진입 {len(E)}건, 달력일 {nd_all}/te {nd_te}")

    # 작업1: 진입 후 hold(4h) 중 max fup, 0.85 도달 빈도
    maxf=np.full(len(E),np.nan); first85=np.full(len(E),-1,int); first_thr={t:np.full(len(E),-1,int) for t in [.78,.82,.85]}
    for j in range(len(E)):
        d=Eqd[j];m0=Emod[j]; hi=min(m0+H,1439)
        seg=fupG[d*1440+m0 : d*1440+hi+1]
        if len(seg)==0: continue
        valid=~np.isnan(seg)
        if valid.any(): maxf[j]=np.nanmax(seg)
        for t in first_thr:
            w=np.where(valid & (seg>=t))[0]
            if len(w): first_thr[t][j]=m0+int(w[0])
    print("\n===== 작업1: 진입 후 hold 중 fup 더 오르나 =====")
    print(f"  진입 fup 평균 {fup[keep].mean():.3f} | hold 중 max fup 평균 {np.nanmean(maxf):.3f}, p90 {np.nanpercentile(maxf,90):.3f}")
    for t in [.78,.82,.85]:
        reach=(first_thr[t]>=0)
        print(f"  hold 중 fup>={t} 도달: {reach.sum()}/{len(E)} = {reach.mean()*100:.0f}%")
    print("  → 31단계 진입 fup max 0.762 였으나 hold 중엔? 도달 드물면 갈아탈 기회 적음.")

    # 작업2/3: 갈아타기 (첫 m' where fup>=X 면 청산+재진입, 수수료 2배)
    print("\n===== 작업2/3: 갈아타기 net vs 고정 4h (수수료 2배 반영, OOS) =====")
    fixed_net=dirn*(P[np.minimum(gm+H,NG-1)]/P[gm]-1)*1e4-FEE
    print(f"  {'갈아타기thr':>10} | {'갈아탄건':>7} | {'전략 net':>8} {'고정 net':>8} | te 전략/고정 일수익")
    for X in [.78,.82,.85]:
        mp=first_thr[X]  # 갈아탈 분 (없으면 -1)
        net=fixed_net.copy()  # 못 갈아탄 건 = 고정
        sw=np.where(mp>=0)[0]
        for j in sw:
            gmp=Eqd[j]*1440+mp[j]
            seg1=P[gmp]/P[gm[j]]-1                       # 진입~갈아탐
            seg2=P[min(gmp+H,NG-1)]/P[gmp]-1             # 갈아탄 후 4h
            net[j]=dirn*(seg1+seg2)*1e4-2*FEE           # 수수료 2배
        dms,los,his=dayci(Eqd[te],net[te]); dmf,lof,hif=dayci(Eqd[te],fixed_net[te])
        print(f"  {X:.2f} | {len(sw):>7} | {net.mean():>+8.1f} {fixed_net.mean():>+8.1f} | "
              f"te {net[te].sum()/nd_te:>+5.2f}/{fixed_net[te].sum()/nd_te:>+5.2f} CI[{los:+.0f},{his:+.0f}]")
    print("  → 갈아타기 net 이 고정 넘으면 강영역 잡음. 못 넘으면 수수료/표본에 막힘.")

    # 작업4: 갈아탄 건만 따로 (강영역 실제 잡았나) + 피라미딩 메모
    print("\n===== 작업4: 갈아탄 건만 분리 (강영역 진짜 이득?) =====")
    for X in [.82,.85]:
        mp=first_thr[X]; sw=np.where(mp>=0)[0]
        if len(sw)<3: print(f"  thr{X}: 갈아탄 {len(sw)}건 표본부족"); continue
        sn=np.array([dirn*((P[Eqd[j]*1440+mp[j]]/P[gm[j]]-1)+(P[min(Eqd[j]*1440+mp[j]+H,NG-1)]/P[Eqd[j]*1440+mp[j]]-1))*1e4-2*FEE for j in sw])
        fn=fixed_net[sw]
        tesw=te[sw]
        print(f"  thr{X}: 갈아탄 {len(sw)}건 — 갈아타기 {sn.mean():+.1f} vs 같은건 고정 {fn.mean():+.1f} | te {tesw.sum()}건")
    print("  피라미딩(강해지면 추가): one-way 넷팅=같은 방향 추가는 평단가 합산(독립 포지션 X)")
    print("  → 강영역 잡아도 수수료 2배+진입~갈아탐 구간 노출. OOS 표본(갈아탄 건) 적으면 미확정.")

if __name__=='__main__':
    main()
