import numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
R=pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
H=['5m','10m','30m','1h','4h']
print('=== audit A: thr70 분해 (방향별/outlier/유효day) ===')
for h in H:
    ok=(R[f'{h}_n']>=70)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
    s=R[ok]
    lean=(s[f'{h}_fup']>=.7)|(s[f'{h}_fup']<=.3)
    L=s[lean].copy()
    if len(L)<10: continue
    sgn=np.where(L[f'{h}_fup']>=.5,1.0,-1.0)
    ret=sgn*L[f'{h}_frq'].to_numpy()*1e4
    up=sgn>0; dn=~up
    udays=L.qday.nunique()
    # outlier 기여: 상위3 절대값 제외 평균
    o=np.argsort(-np.abs(ret))[:3]
    ret_t=np.delete(ret,o)
    med=np.median(ret)
    print(f"{h:3s}: n={len(L)} 유효day={udays} | up-lean {up.sum()} (hit {(ret[up]>0).mean():.2f}, gross {ret[up].mean():+.1f}) "
          f"| down-lean {dn.sum()} (hit {(ret[dn]>0).mean():.2f}, gross {ret[dn].mean():+.1f}) "
          f"| med {med:+.1f} | top3 제외 평균 {ret_t.mean():+.1f}")
print()
print('=== audit B: drift 벤치마크 (같은 분기, 같은 방향 random 진입 대비 초과) ===')
# 분기×방향 평균 미래수익 (전체 쿼리 = 무조건 진입 기준)
for h in ['30m','1h','4h']:
    ok=(R[f'{h}_n']>=70)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
    s=R[ok]
    lean=(s[f'{h}_fup']>=.7)|(s[f'{h}_fup']<=.3)
    L=s[lean].copy()
    sgn=np.where(L[f'{h}_fup']>=.5,1.0,-1.0)
    ret=sgn*L[f'{h}_frq'].to_numpy()*1e4
    # 벤치: 그 분기 전체 쿼리의 frq 평균 (long 이면 +, short 면 -)
    qmean=s.groupby('quarter')[f'{h}_frq'].mean()*1e4
    bench=np.array([qmean[q] for q in L.quarter])*sgn
    exc=ret-bench
    print(f"{h:3s}: gross {ret.mean():+.1f} | drift벤치 {bench.mean():+.1f} | 초과 {exc.mean():+.1f}bp "
          f"(초과>0 비율 {(exc>0).mean():.2f})")
print()
print('=== audit C: 1h/4h thr70 이벤트 시간 군집 (같은 날 몇 개) ===')
for h in ['1h','4h']:
    ok=(R[f'{h}_n']>=70)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
    s=R[ok]
    lean=(s[f'{h}_fup']>=.7)|(s[f'{h}_fup']<=.3)
    L=s[lean]
    per=L.groupby('qday').size()
    print(f"{h:3s}: 이벤트 {len(L)} / day {len(per)} — day당 max {per.max()}, day당 분포 {per.value_counts().sort_index().to_dict()}")
