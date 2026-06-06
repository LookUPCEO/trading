import numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
R=pd.read_parquet(f'{OUT}/lean70_per_query.parquet')
H=['5m','10m','30m','1h','4h']
print('=== 에피소드 분석 (thr70: 인접 격자점 연속 = 같은 사건?) ===')
nrm_mod=None
for h in H:
    ok=R[f'{h}_n']>=70
    s=R[ok].copy()
    lean=(s[f'{h}_fup']>=.7)|(s[f'{h}_fup']<=.3)
    L=s[lean].sort_values('q')
    if len(L)==0: continue
    # q index 는 분 단위 행 인덱스 (10분 격자) — 같은 날 & q 차이 <= 6행(=60분) 이면 같은 에피소드
    new_ep=(L.qday.diff()!=0)|(L.q.diff()>60)
    n_ep=int(new_ep.sum())
    print(f'{h:3s}: 쏠림 {len(L)}건 → 에피소드 {n_ep}개 (60분 병합) → 일평균 {n_ep/s.qday.nunique():.3f}')
print()
print('=== horizon 중복 (thr70 쏠림 쿼리의 겹침) ===')
sets={h:set(R[(R[f'{h}_n']>=70)&((R[f'{h}_fup']>=.7)|(R[f'{h}_fup']<=.3))].q) for h in H}
import itertools
uni=set().union(*sets.values())
print('합집합:',len(uni),'| 각:',{h:len(sets[h]) for h in H})
for a,b in [('5m','10m'),('5m','30m'),('30m','1h'),('1h','4h')]:
    inter=len(sets[a]&sets[b]); print(f'{a}∩{b}: {inter} (={inter/max(len(sets[a]),1)*100:.0f}% of {a})')
print()
print('=== thr65 (참고: 하루 몇 건 감각) ===')
for h in H:
    ok=R[f'{h}_n']>=70; s=R[ok]
    lean=(s[f'{h}_fup']>=.65)|(s[f'{h}_fup']<=.35)
    L=s[lean].sort_values('q')
    new_ep=(L.qday.diff()!=0)|(L.q.diff()>60)
    print(f'{h:3s}: {lean.sum()}건({lean.mean()*100:.2f}%) → 에피소드 {int(new_ep.sum())} → 일평균 {int(new_ep.sum())/s.qday.nunique():.2f}')
print()
print('=== train(2024Q1~2025Q2) vs test(2025Q3~) — 룰 고정 OOS (thr70) ===')
tr=R.quarter.isin(['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2'])
for h in H:
    ok=R[f'{h}_n']>=70
    a=R[ok&tr]; b=R[ok&~tr]
    la=((a[f'{h}_fup']>=.7)|(a[f'{h}_fup']<=.3)).mean()
    lb=((b[f'{h}_fup']>=.7)|(b[f'{h}_fup']<=.3)).mean()
    print(f'{h:3s}: train {la*100:.2f}% → test {lb*100:.2f}%')
