#!/usr/bin/env python3
"""[I] 26단계 — down 신호 전면 재검증 (표본 보강 + down 전용).
25단계가 down-4h-thr0.70 을 test n=4 로 '판정불능' → 표본 보강해 제대로 본다.
표본 보강 2축: (1) 임계완화 0.30/0.35/0.40 (2) dense stride=1 검색(현 artifact, 825k 쿼리).
  dense = lean70_v2_per_query_dense.parquet (현 labels_norm_reduced 로 생성 → q↔현 nrm).
독립성: 같은 day 미래창 비겹침 greedy(강도순) → 진짜 독립 down 건수.
대칭 가정 버림: down 전용 임계/horizon. 단 down 전용 튜닝=과적합 경계(OOS+Bonferroni).
lookahead 0: frq=쿼리 자신 미래. 라이브 검증만(4h 유지)."""
import numpy as np, pandas as pd, json, os
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
FEE=11.0; MINV=70
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
HZ=['15m','20m','30m','45m','1h','2h','3h','4h']
HMIN={'15m':15,'20m':20,'30m':30,'45m':45,'1h':60,'2h':120,'3h':180,'4h':240}
PARQ=os.environ.get('PARQ',f'{OUT}/lean70_v2_per_query_dense.parquet')

def dayci(qd,net,seed=7):
    if len(net)<5: return np.nan,np.nan,np.nan
    dm=pd.Series(net).groupby(qd).mean().to_numpy()
    bs=np.random.default_rng(seed).choice(dm,(4000,len(dm)),replace=True).mean(axis=1)
    return dm.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)

def greedy_indep(mod,qday,h):
    """같은 day 내 |Δmin|>=h 만 독립 채택 (강도순 입력 가정). bool mask 반환."""
    acc={};keep=np.zeros(len(mod),bool)
    for i in range(len(mod)):
        d=qday[i];m=mod[i];lst=acc.get(d)
        if lst is None: acc[d]=[m];keep[i]=True;continue
        if all(abs(m-mm)>=h for mm in lst): lst.append(m);keep[i]=True
    return keep

def main():
    R=pd.read_parquet(PARQ)
    # min_of_day 복원 (현 nrm) — dense 는 현 nrm 로 생성됨
    nrm=pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    R['mod']=nrm['min_of_day'].to_numpy()[R['q'].to_numpy()]
    print(f"[load] {PARQ.split('/')[-1]} : {len(R)} 쿼리")

    print("\n===== 작업1: down 표본 — 임계×horizon n (all / 독립) =====")
    print(f"{'hz':>4} | "+" | ".join(f"≤{t:.2f} all/indep" for t in [.30,.35,.40]))
    for h in HZ:
        ok=(R[f'{h}_n']>=MINV)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
        s=R[ok];line=f"{h:>4} | "
        for thr in [.30,.35,.40]:
            m=(s[f'{h}_fup']<=thr).to_numpy()
            L=s[m].copy()
            if len(L)==0: line+=f"{'0/0':>16} | ";continue
            stg=(0.5-L[f'{h}_fup']).to_numpy();o=np.argsort(-stg)
            keep=greedy_indep(L['mod'].to_numpy()[o],L.qday.to_numpy()[o],HMIN[h])
            line+=f"{len(L):>6}/{keep.sum():<6} | "
        print(line)
    print("  ↑ 임계완화로 all n 급증하나 독립 n(=고유 down 에피소드)은? 강세장서 구조적 한계")

    print("\n===== 작업2+3: down 전용 최적 (임계×horizon) hit/net/OOS/CI =====")
    print(f"{'hz':>4} {'thr':>5} | {'n':>4} {'indN':>4} {'hit':>5} {'gross':>6} {'net':>6} {'teN':>4} {'teHit':>5} | dayCI(test)")
    cells=[]
    for h in HZ:
        for thr in [.30,.35,.40]:
            ok=(R[f'{h}_n']>=MINV)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
            s=R[ok];m=(s[f'{h}_fup']<=thr).to_numpy();L=s[m].copy()
            if len(L)<5: continue
            frq=L[f'{h}_frq'].to_numpy();net=(-1)*frq*1e4-FEE
            L=L.assign(net=net,hit=(frq<0))
            stg=(0.5-L[f'{h}_fup']).to_numpy();o=np.argsort(-stg)
            keep=greedy_indep(L['mod'].to_numpy()[o],L.qday.to_numpy()[o],HMIN[h])
            Li=L.iloc[o[keep]]   # 독립 부분집합
            te=~Li.quarter.isin(TRAIN_Q);Lte=Li[te]
            dm,lo,hi=dayci(Lte.qday.to_numpy(),Lte.net.to_numpy())
            g=(Li.net+FEE).mean()
            cells.append((h,thr,len(L),len(Li),Li.hit.mean(),g,Li.net.mean(),len(Lte),
                          Lte.hit.mean() if len(Lte) else np.nan,dm,lo,hi))
            print(f"{h:>4} ≤{thr:.2f} | {len(L):>4} {len(Li):>4} {Li.hit.mean():>5.3f} {g:>+6.1f} {Li.net.mean():>+6.1f} "
                  f"{len(Lte):>4} {Lte.hit.mean() if len(Lte) else 0:>5.2f} | [{lo:+.0f},{hi:+.0f}]")

    print(f"\n===== 작업3 핵심: down-4h-thr0.70 표본 보강 후 (25단계 n=4 였음) =====")
    h='4h';ok=(R[f'{h}_n']>=MINV)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
    s=R[ok];L=s[s[f'{h}_fup']<=0.30].copy()
    frq=L[f'{h}_frq'].to_numpy();L=L.assign(net=(-1)*frq*1e4-FEE,hit=(frq<0))
    stg=(0.5-L['4h_fup']).to_numpy();o=np.argsort(-stg)
    keep=greedy_indep(L['mod'].to_numpy()[o],L.qday.to_numpy()[o],240);Li=L.iloc[o[keep]]
    te=~Li.quarter.isin(TRAIN_Q)
    dm,lo,hi=dayci(Li[te].qday.to_numpy(),Li[te].net.to_numpy())
    dmf,lof,hif=dayci(Li.qday.to_numpy(),Li.net.to_numpy())
    print(f"  down 4h thr0.70: all n={len(L)} 독립 n={len(Li)} (25단계 stride10 n=17)")
    print(f"    full: hit {Li.hit.mean():.3f} net {Li.net.mean():+.1f} dayCI[{lof:+.0f},{hif:+.0f}]")
    print(f"    test: n={te.sum()} hit {Li[te].hit.mean() if te.sum() else 0:.3f} net {Li[te].net.mean() if te.sum() else 0:+.1f} dayCI[{lo:+.0f},{hi:+.0f}]")
    for y in [2024,2025,2026]:
        my=(Li.qyr==y)
        if my.sum()>=3: print(f"    {y}: n={my.sum()} hit {Li[my].hit.mean():.3f} net {Li[my].net.mean():+.1f}")

    print(f"\n===== 작업4: down 거래화 — fee 넘고 OOS 생존 셀 있나 (Bonferroni) =====")
    C=pd.DataFrame(cells,columns=['h','thr','n','indN','hit','gross','net','teN','teHit','dm','lo','hi'])
    nb=len(C)
    surv=C[(C.gross>FEE)&(C.lo>0)&(C.teN>=5)]
    print(f"  시도 셀 {nb} (Bonferroni 분모). gross>fee & test dayCI>0 & teN>=5 생존: {len(surv)}")
    if len(surv): print(surv.to_string(index=False))
    else: print("  → down 거래가능(fee초과+OOS 생존) 셀 0")
    # 현행 4h 양방향 baseline 에 down 추가 효과
    print(f"\n  현행 4h 양방향(up+down thr0.70) 이미 down 포함 — down 단독으로 새 빈도/수익 추가 가능 셀: {len(surv)}")
    C.to_csv(f'{OUT}/down_cells.csv',index=False)

if __name__=='__main__':
    main()
