#!/usr/bin/env python3
"""[I] 18단계 작업4 — 봉 AND/OR 결합 (봉=공간 1분~4h). 봉별 방향=모멘텀 부호.
AND(k봉 동방향→정확도) / OR(1봉이라도→빈도). 4h 미래 hit/freq/net OOS. 유사도와 다른 신호원."""
import itertools
import numpy as np, pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB = '/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
FEE = 11.0
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
BARS=[1,3,5,15,30,60,120,240]   # 분 봉 공간 (1분~4h)

lab=pd.read_parquet(LAB,columns=['yr','day','min_of_day','mid'])
days=sorted(lab['day'].unique()); dix={d:i for i,d in enumerate(days)}
drow=lab['day'].map(dix).to_numpy(); mod=lab['min_of_day'].to_numpy()
ND=len(days); n=len(lab)
mid=np.full((ND,1440),np.nan,np.float32); mid[drow,mod]=lab['mid'].to_numpy(np.float32)
yr=lab['yr'].astype(int).to_numpy()
qtr=np.char.add(yr.astype(str),np.char.add('Q',(((lab['day'].str[5:7].astype(int)-1)//3+1)).astype(str).to_numpy()))

# 봉별 방향 = 봉 모멘텀 부호 (직전 완성봉 close vs 그 이전 봉 close, causal). day 내.
def bar_dir(B):
    """각 분의 '직전 완성봉 모멘텀 부호' (봉 close - 이전 봉 close). t≤0."""
    out=np.zeros(n, np.int8)
    for di in range(ND):
        row=mid[di]
        bar_of_min=np.arange(1440)//B
        # 봉 close (각 봉 마지막 유효 분)
        s=pd.Series(row, index=np.arange(1440))
        bc=s.groupby(np.arange(1440)//B).last()
        mom=bc.diff()   # 봉간 변화
        # 분 → 직전 완성봉(현재 봉 미완성 → shift1)의 모멘텀 부호
        d=np.sign(mom.shift(1).reindex(bar_of_min).to_numpy())
        rr=np.where(drow==di)[0]
        if len(rr): out[rr]=np.nan_to_num(d[mod[rr]]).astype(np.int8)
    return out

print(f"[load] {n} 분행, 봉 공간 {BARS}")
DIRS={B:bar_dir(B) for B in BARS}
print("봉별 방향 계산 완료")

# 미래 4h
fr=np.full(n,np.nan,np.float32); ok=mod+240<=1439
fr[ok]=mid[drow[ok],mod[ok]+240]/mid[drow[ok],mod[ok]]-1
valid=~np.isnan(fr)&(fr!=0)
is_tr=np.isin(qtr,TRAIN_Q)
# 쿼리 = 2024+ 10분격자 (빈도 측정 일관)
qsel=(yr>=2024)&(mod%10==5)&valid
nd=len(set(drow[qsel])); ndte=len(set(drow[qsel&~is_tr]))

# ===== redundancy 확인: 봉 방향끼리 corr =====
print("\n===== 봉 방향 상관 (봉이 서로 다른 정보인가) =====")
D=np.column_stack([DIRS[B] for B in BARS])
m=qsel
cc=np.corrcoef(D[m].T)
print("   "+" ".join(f"{B:>4}" for B in BARS))
for i,B in enumerate(BARS):
    print(f"{B:>3} "+" ".join(f"{cc[i,j]:+.2f}" for j in range(len(BARS))))

def stats(mask):
    m=mask&qsel
    if m.sum()<10: return None
    # 방향 = AND/OR 가 정한 dir (아래서 전달). 여기선 net 계산용 분리
    return m

print("\n===== 작업4: AND (k봉 동방향) — 정확도 (4h hit/net, 빈도) =====")
print("k = 동방향 봉 수 (8봉 중). dir = 다수결.")
Dq=D.copy()
agree_up=(Dq>0).sum(1); agree_dn=(Dq<0).sum(1)
cdir=np.where(agree_up>=agree_dn, 1, -1); kmax=np.maximum(agree_up,agree_dn)
for kth in [4,5,6,7,8]:
    m=(kmax>=kth)&qsel
    if m.sum()<10: print(f"  AND k>={kth}: n={m.sum()}"); continue
    net=cdir[m]*fr[m]*1e4-FEE; freq=m.sum()/nd
    tr=m&is_tr; te=m&~is_tr
    net_te=cdir[te]*fr[te]*1e4-FEE
    print(f"  AND k>={kth}: n={m.sum()} 빈도{freq:.2f}/d hit{(net>0).mean():.3f} net{net.mean():+.1f} "
          f"일수익 full{net.sum()/nd:+.2f}/test{net_te.sum()/ndte:+.2f}")

print("\n===== 작업4: OR (1봉이라도 강방향) + 단일 봉별 방향 예측력 =====")
for B in BARS:
    m=(DIRS[B]!=0)&qsel
    if m.sum()<10: continue
    d=DIRS[B]; net=d[m]*fr[m]*1e4-FEE; te=m&~is_tr
    net_te=d[te]*fr[te]*1e4-FEE
    print(f"  봉{B:>3}분 단독: n={m.sum()} hit{(net>0).mean():.3f} net{net.mean():+.1f} test일수익{net_te.sum()/ndte:+.2f}")

print(f"\n현행 단일4h 유사도 thr0.70 = 0.084%/day(full), test +2.90bp. 봉 모멘텀은 다른 신호원.")
print("AND 가 hit↑(정확도)·OR/단독이 빈도↑ 면서 현행 넘으면 의미. OOS 생존 필수.")
