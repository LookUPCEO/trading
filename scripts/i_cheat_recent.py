import numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
R=pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
H=['5m','10m','30m','1h','4h']
print('=== cheat injection (분석 배선 sanity): 매치 votes 를 쿼리 자신의 미래로 치환 ===')
for h in ['30m','1h']:
    ok=(R[f'{h}_n']>=70)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
    s=R[ok].copy()
    cheat_fup=np.where(s[f'{h}_frq']>0,1.0,0.0)   # 미래정보 주입
    lean=(cheat_fup>=.7)|(cheat_fup<=.3)
    sgn=np.where(cheat_fup>=.5,1.0,-1.0)
    ret=sgn*s[f'{h}_frq'].to_numpy()*1e4
    print(f"{h}: cheat lean rate {lean.mean()*100:.0f}% (정상이면 ~100%), cheat hit {(ret[lean]>0).mean():.3f} (정상이면 1.0), gross {ret[lean].mean():+.1f}bp (비정상 양수=배선 정상)")
print()
print('=== test 기간 (2025Q3~2026, OOS) thr70 집계 ===')
test_q=['2025Q3','2025Q4','2026Q1','2026Q2']
rows=[]
for h in H:
    ok=(R[f'{h}_n']>=70)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)&R.quarter.isin(test_q)
    s=R[ok]
    lean=(s[f'{h}_fup']>=.7)|(s[f'{h}_fup']<=.3)
    L=s[lean]
    ndays=s.qday.nunique()
    if len(L)<5:
        rows.append((h,len(L),np.nan,np.nan,np.nan,np.nan)); continue
    sgn=np.where(L[f'{h}_fup']>=.5,1.0,-1.0)
    ret=sgn*L[f'{h}_frq'].to_numpy()*1e4
    dm=pd.Series(ret).groupby(L.qday.to_numpy()).mean().to_numpy()
    rng=np.random.default_rng(7)
    bs=rng.choice(dm,(4000,len(dm)),replace=True).mean(axis=1)
    lo,hi=np.percentile(bs,[2.5,97.5])
    rows.append((h,len(L),(ret>0).mean(),ret.mean(),lo,hi))
T=pd.DataFrame(rows,columns=['h','n','hit','gross','ci_lo','ci_hi'])
T['net_TT']=T.gross-11
print(T.round(2).to_string(index=False))
print()
print('=== 일수익 환산 (851일 전체 / test 기간) thr70, net T+T ===')
all_days=R.qday.nunique()
tdays=R[R.quarter.isin(test_q)].qday.nunique()
for h in H:
    ok=(R[f'{h}_n']>=70)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
    s=R[ok]; lean=(s[f'{h}_fup']>=.7)|(s[f'{h}_fup']<=.3); L=s[lean]
    sgn=np.where(L[f'{h}_fup']>=.5,1.,-1.); ret=sgn*L[f'{h}_frq'].to_numpy()*1e4
    full=(ret-11).sum()/all_days
    m=L.quarter.isin(test_q).to_numpy()
    test=(ret[m]-11).sum()/tdays if m.sum()>0 else np.nan
    print(f"{h:3s}: 전체 {full:+.2f}bp/day (n={len(L)}) | test기간 {test:+.2f}bp/day (n={m.sum()})")
