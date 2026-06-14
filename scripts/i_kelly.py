#!/usr/bin/env python3
"""[I] 14단계 ② — Kelly 베팅 (합의깊이 k → hit → fractional Kelly 크기). 빈도 안 늚, 크기만.
일수익 + 파산위험 (bankroll 경로 max drawdown). full vs 1/4 Kelly. 기존 합의 데이터 재사용."""
import itertools
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
FEE = 11.0
TRAIN_Q = ['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
H5 = ['30m','45m','1h','2h','4h']

v2 = pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
hf = pd.read_parquet(f'{OUT}/lean70_v2_per_query_hfine.parquet')
R = v2.merge(hf[['q']+[c for c in hf.columns if c.split('_')[0] in ('45m','2h')]], on='q', how='inner')
R['is_tr'] = R.quarter.isin(TRAIN_Q)

def consensus(df, thr=0.70):
    sig = np.zeros((len(df), len(H5)), np.int8)
    for j,h in enumerate(H5):
        f=df[f'{h}_fup'].to_numpy(); nn=df[f'{h}_n'].to_numpy()
        st=np.where(nn>=70, np.maximum(f,1-f), np.nan); sd=np.where(f>=.5,1,-1)
        sig[:,j]=np.where((st>=thr)&~np.isnan(st), sd, 0)
    cdir=np.sign(sig.sum(1))
    k=np.where(cdir>0,(sig>0).sum(1),np.where(cdir<0,(sig<0).sum(1),0))
    return cdir, k.astype(int)

cd, k = consensus(R)
frq4 = R['4h_frq'].to_numpy(); v = ~np.isnan(frq4)&(frq4!=0)&(k>=1)
# 거래 = 합의 발생 쿼리 (4h hold). ret(비율, 자본대비) = dir*frq4 - fee_frac
ret_frac = (cd*frq4 - FEE/1e4)   # 비율 (per-trade, 자본 1 단위)
sub = R[v].copy(); sub['k']=k[v]; sub['ret']=ret_frac[v]; sub['is_tr']=R.is_tr.to_numpy()[v]

print(f"[load] 합의 거래 {len(sub)}건 (k>=1)")
# k별 train hit/edge → Kelly f* (이진근사: f* = edge/odds; 여기선 평균수익/분산 연속 Kelly = mean/var)
print("\n===== k별 train 통계 + Kelly f* (mean/var, 연속) =====")
kf = {}
for kk in [1,2,3,4,5]:
    m = (sub.k==kk)&sub.is_tr if kk<5 else (sub.k>=5)&sub.is_tr
    r = sub[m].ret
    if len(r)<5: print(f"k={kk}: train n={len(r)}"); kf[kk]=0; continue
    mu=r.mean(); var=r.var()
    fstar = mu/var if var>0 else 0   # 연속 Kelly (자본배수)
    kf[kk]=max(fstar,0)
    print(f"k={kk}: train n={len(r)} hit{(r>0).mean():.2f} mean{mu*1e4:+.0f}bp std{r.std()*1e4:.0f} f*={fstar:.1f}x")

# Kelly 분수 캡 (파산방지): f = min(frac_kelly * f*_train, MAXLEV). 강한 신호 크게.
print("\n===== OOS(2025Q3~) bankroll 시뮬: 고정 vs Kelly(분수) =====")
te = sub[~sub.is_tr].sort_values('q').reset_index(drop=True)
ndte = R[~R.is_tr].qday.nunique()
def simulate(sizes, label):
    bank=1.0; path=[1.0]; peak=1.0; maxdd=0
    for r,sz in zip(te.ret.to_numpy(), sizes):
        bank *= (1 + sz*r)   # 복리, 자본대비 sz배 노출
        if bank<=0: bank=1e-9
        peak=max(peak,bank); maxdd=max(maxdd,(peak-bank)/peak); path.append(bank)
    tot=(bank-1)*100; daily=tot/ndte
    print(f"  {label}: 최종 {bank:.3f} (총 {tot:+.1f}%, 일 {daily:+.3f}%) | maxDD {maxdd*100:.0f}% | n={len(te)}")
    return bank, maxdd
# 고정 1x
simulate(np.ones(len(te)), "고정 1x (현행)")
# Kelly 분수들 (k별 f*_train 적용, 캡)
for frac, cap in [(1.0, 5), (0.5, 3), (0.25, 2)]:
    sizes = np.array([min(frac*kf.get(int(kk),0), cap) for kk in te.k])
    simulate(sizes, f"{frac:.2f}-Kelly (k별 f*, cap {cap}x)")
print(f"\n현행 단일4h 고정 ~0.084%/day. Kelly 는 빈도 안 늘리고 크기만 → 일수익 오르나 maxDD 대가.")
print("⚠️ 파산위험 = maxDD. full Kelly maxDD 큼. 합의 k 표본 작아(k4,5 n<20) f* 불안정.")
