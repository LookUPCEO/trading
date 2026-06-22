#!/usr/bin/env python3
"""[I] 27-1단계 — 하락 단기 분단위 정밀 (5~50분 촘촘히).
감사: 27단계 shortest=30m, 26단계 shortest=15m → 5/10/25/40/50m 미탐. 25단계 15m/20m
hit 0.66 은 stride10 소표본. → dense stride=1 로 5~50분 8개 촘촘히, 표본 확보.
핵심(작업3): '하락=급락이라 짧아도 폭 클 수 있나' = 짧은 horizon down 의 GROSS 가 fee 넘나
  (5단계 fee/|move| 벽의 예외인가). down |move| vs up |move| 대조.
causal 독립(시간순, 26단계서 강도순=선택lookahead 확인). OOS+Bonferroni. lookahead 0."""
import numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
PARQ=f'{OUT}/lean70_v2_per_query_downfine.parquet'
FEE=11.0; MINV=70
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
HZ=['5m','10m','15m','20m','25m','30m','40m','50m']
HMIN={'5m':5,'10m':10,'15m':15,'20m':20,'25m':25,'30m':30,'40m':40,'50m':50}

def dayci(qd,net,seed=7):
    if len(net)<5: return np.nan,np.nan,np.nan
    dm=pd.Series(net).groupby(qd).mean().to_numpy()
    bs=np.random.default_rng(seed).choice(dm,(4000,len(dm)),replace=True).mean(axis=1)
    return dm.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)

def indep_causal(L,h):
    """시간순(causal) 독립 = 거래가능 (먼저 온 신호 take, 이후 h분 블록)."""
    o=np.lexsort((L['mod'].to_numpy(),L.qday.to_numpy()))
    acc={};keep=np.zeros(len(L),bool)
    mod=L['mod'].to_numpy();qd=L.qday.to_numpy()
    for idx in o:
        d=qd[idx];m=mod[idx];lst=acc.get(d)
        if lst is None: acc[d]=[m];keep[idx]=True;continue
        if all(abs(m-mm)>=h for mm in lst): lst.append(m);keep[idx]=True
    return L[keep]

def main():
    R=pd.read_parquet(PARQ)
    nrm=pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    R['mod']=nrm['min_of_day'].to_numpy()[R['q'].to_numpy()]
    print(f"[load] downfine dense {len(R)} 쿼리, horizons {HZ}")

    print("\n===== 작업2: 하락 단기 분단위 hit/gross/net (causal 독립) =====")
    print(f"{'hz':>4} {'thr':>5} | {'allN':>6} {'indN':>6} {'hit':>5} {'gross':>6} {'net':>6} | {'teN':>4} {'teHit':>5} dayCI(test)")
    cells=[]
    for h in HZ:
        for thr in [.30,.35,.40]:
            ok=(R[f'{h}_n']>=MINV)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
            s=R[ok];L=s[s[f'{h}_fup']<=thr].copy()
            if len(L)<5: continue
            frq=L[f'{h}_frq'].to_numpy();L=L.assign(net=(-1)*frq*1e4-FEE,hit=(frq<0))
            Li=indep_causal(L,HMIN[h]);te=~Li.quarter.isin(TRAIN_Q);Lte=Li[te]
            dm,lo,hi=dayci(Lte.qday.to_numpy(),Lte.net.to_numpy())
            g=(Li.net+FEE).mean()
            cells.append((h,thr,len(L),len(Li),Li.hit.mean(),g,Li.net.mean(),len(Lte),
                          Lte.hit.mean() if len(Lte) else np.nan,dm,lo,hi))
            print(f"{h:>4} ≤{thr:.2f} | {len(L):>6} {len(Li):>6} {Li.hit.mean():>5.3f} {g:>+6.1f} {Li.net.mean():>+6.1f} | "
                  f"{len(Lte):>4} {Lte.hit.mean() if len(Lte) else 0:>5.2f} [{lo:+.0f},{hi:+.0f}]")

    print("\n===== 작업3: 짧은 하락이 급락이라 폭 큰가 — down |move| vs up |move| (gross, thr0.30) =====")
    print(f"{'hz':>4} | down gross(공매도 맞은폭) | up gross | down|mv| up|mv| 평균 | fee={FEE}")
    for h in HZ:
        ok=(R[f'{h}_n']>=MINV)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0);s=R[ok]
        Ld=s[s[f'{h}_fup']<=.30];Lu=s[s[f'{h}_fup']>=.70]
        dg=(-1)*Ld[f'{h}_frq'].to_numpy()*1e4;ug=Lu[f'{h}_frq'].to_numpy()*1e4
        dmv=np.abs(Ld[f'{h}_frq'].to_numpy()*1e4).mean();umv=np.abs(Lu[f'{h}_frq'].to_numpy()*1e4).mean()
        print(f"{h:>4} | {dg.mean():>+7.1f} (n{len(Ld)}) | {ug.mean():>+7.1f} (n{len(Lu)}) | "
              f"{dmv:>5.0f} {umv:>5.0f} | {'down 폭>fee' if dg.mean()>FEE else 'fee미달'}")
    print("  → 하락이 급락이라 짧아도 gross>fee 면 5단계 벽 예외. 아니면 짧은하락도 작은움직임.")

    print("\n===== 작업4: 거래화 + OOS (Bonferroni) =====")
    C=pd.DataFrame(cells,columns=['h','thr','allN','indN','hit','gross','net','teN','teHit','dm','lo','hi'])
    nb=len(C)
    surv=C[(C.gross>FEE)&(C.lo>0)&(C.teN>=5)]
    print(f"  시도 {nb} (Bonferroni 분모). gross>fee & test dayCI>0 & teN>=5 생존: {len(surv)}")
    if len(surv): print(surv.to_string(index=False))
    else: print("  → 짧은 하락도 거래가능(fee초과+OOS생존) 셀 0")
    # 25단계 15m/20m 0.66 재확인
    print("\n  25단계 15m/20m hit 0.66~0.67 dense 재확인:")
    for h in ['15m','20m']:
        ok=(R[f'{h}_n']>=MINV)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0);s=R[ok]
        for thr in [.30,.40]:
            L=s[s[f'{h}_fup']<=thr];hit=(L[f'{h}_frq']<0).mean()
            print(f"    {h}≤{thr:.2f}: dense n={len(L)} hit {hit:.3f} (25단계 stride10 0.66~0.67)")
    C.to_csv(f'{OUT}/downfine_cells.csv',index=False)

if __name__=='__main__':
    main()
