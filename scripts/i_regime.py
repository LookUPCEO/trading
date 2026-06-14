#!/usr/bin/env python3
"""[I] 14단계 ③ — regime 게이트. 시장 상태별 4h thr0.70 hit → 게이트 순효과(빈도×hit).
regime = 기존 라벨(causal): adx(추세강도), rv(변동성), |ma_slope_120|(방향성). t≤0."""
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
FEE = 11.0
TRAIN_Q = ['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']

R = pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
nrm = pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
# regime 라벨 (쿼리 시점, causal — 정규화공간 z 값)
for col in ['z_adx_14','z_atr_14','z_ma_slope_120','z_rv_ratio']:
    if col in nrm.columns: R[col]=nrm[col].iloc[R.q].to_numpy()
R['is_tr']=R.quarter.isin(TRAIN_Q)
nd=R.qday.nunique(); ndte=R[~R.is_tr].qday.nunique()

def ev(df):
    ok=(df['4h_n']>=70)&~df['4h_frq'].isna()&(df['4h_frq']!=0)
    s=df[ok]; lean=(s['4h_fup']>=.7)|(s['4h_fup']<=.3); L=s[lean]
    sgn=np.where(L['4h_fup']>=.5,1.,-1.)
    L=L.copy(); L['net']=sgn*L['4h_frq'].to_numpy()*1e4-FEE
    return L
L=ev(R)
print(f"[load] 4h thr0.70 이벤트 {len(L)} (full)")
print(f"baseline 전체: hit{(L.net+FEE>0).mean():.3f} 일수익 {L.net.sum()/nd:+.2f} | test {L[~L.is_tr].net.sum()/ndte:+.2f}")

print("\n===== ③ regime 별 hit/net (전체) — 어느 상태가 강한가 =====")
regimes = []
# adx 추세강도, atr 변동성, |slope| 방향성 — 중앙값 기준 hi/lo
for col, name in [('z_adx_14','추세강도ADX'),('z_atr_14','변동성ATR'),('z_rv_ratio','vol급증rv'),('z_ma_slope_120','추세방향slope')]:
    if col not in L.columns: continue
    val = L[col].to_numpy()
    if 'slope' in col: val=np.abs(val)
    med = np.median(L[L.is_tr][col].abs() if 'slope' in col else L[L.is_tr][col])  # train 중앙값 (causal 임계)
    for lab, m in [('hi', val>med), ('lo', val<=med)]:
        s=L[m]
        if len(s)<10: continue
        regimes.append((f"{name}-{lab}", m))
        print(f"  {name}-{lab} (>{med:.2f}): n={len(s)} hit{(s.net+FEE>0).mean():.3f} net{s.net.mean():+.1f} 일수익(전체일) {s.net.sum()/nd:+.2f}")

print("\n===== regime 게이트 OOS (train 최고 hit regime → test) =====")
# train 에서 hit 높은 단일 regime 선택 → test 순효과 (빈도 줆 vs hit↑)
best=None; besthit=-1
for nm, m in regimes:
    tr = m & L.is_tr.to_numpy()
    if tr.sum()<20: continue
    h = (L[tr].net+FEE>0).mean()
    if h>besthit: besthit=h; best=(nm,m)
if best:
    nm,m = best
    te = m & ~L.is_tr.to_numpy()
    s=L[te]
    dm = s.groupby('qday')['net'].mean()
    bs = np.random.default_rng(7).choice(dm.to_numpy(),(4000,len(dm)),replace=True).mean(axis=1) if len(dm)>=5 else [np.nan]
    print(f"train 최고 regime = {nm} (train hit {besthit:.3f})")
    print(f"  → test: n={te.sum()} hit{(s.net+FEE>0).mean():.3f} net{s.net.mean():+.1f} 일수익 {s.net.sum()/ndte:+.2f} "
          f"CI[{np.percentile(bs,2.5):+.0f},{np.percentile(bs,97.5):+.0f}]")
    print(f"  baseline(게이트X) test 일수익 {L[~L.is_tr].net.sum()/ndte:+.2f} — 게이트가 넘나(빈도줆 순효과)")
print("\n⚠️ mark18-R regime 분류 33~50% 약했음. 게이트는 빈도 줄여 일수익 순효과 불확실.")
