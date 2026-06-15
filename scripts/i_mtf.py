#!/usr/bin/env python3
"""[I] 18단계 — 멀티 타임프레임 봉 지표 redundancy 게이트 (8단계 교훈).
분 mid → 5m/15m/30m/60m 봉 → 봉별 RSI/MACD/boll_pos/stoch. 기존 21차원과 corr.
redundant 면 8단계 재현 (효과 없음). 독립이면 추가 검증."""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from i_labeling import rsi, macd, stoch, sma

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB = '/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'

lab = pd.read_parquet(LAB, columns=['yr','day','min_of_day','mid'])
nrm = pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet')
print(f"[load] {len(lab)} 분행")

# ---- 봉 리샘플 (day 내, 분 mid → Nm 봉 close), 봉별 지표 → 각 분에 last 완성봉 값 (causal) ----
BARS = [5, 15, 30, 60]
def mtf_indicators(day_df):
    """한 day: 분 시리즈 → 각 봉 지표를 분 단위로 ffill (last 완성봉, causal)."""
    m = day_df['min_of_day'].to_numpy(); mid = day_df['mid'].to_numpy()
    full = pd.Series(np.nan, index=np.arange(1440)); full.iloc[m] = mid
    out = {}
    for B in BARS:
        # 봉 close = 각 B분 구간 마지막 분 mid. 봉 index = minute//B
        bar_close = full.groupby(full.index // B).last()   # 봉별 마지막값
        bclose = bar_close.reset_index(drop=True)
        r = rsi(bclose, 14); mc,_,_ = macd(bclose); bm=sma(bclose,20); bsd=bclose.rolling(20,min_periods=20).std(ddof=0)
        bpos = (bclose-bm)/(bsd+1e-12)
        sk,_ = stoch(bclose, bclose, bclose, 14, 3)
        # 분 → 그 분이 속한 봉의 '직전 완성봉' (현재 봉 미완성 → shift 1, causal)
        bar_of_min = np.arange(1440)//B
        for nm, ser in [('rsi',r),('macd',mc),('bpos',bpos),('stoch',sk)]:
            sv = ser.shift(1).reindex(bar_of_min).to_numpy()   # 직전 완성봉
            out[f'{nm}_{B}m'] = sv
    res = pd.DataFrame(out); res['min_of_day']=np.arange(1440).astype('int64'); res['day']=str(day_df['day'].iloc[0])
    return res

# 샘플 일자 (전수 비싸므로 redundancy 는 충분 표본)
days = sorted(lab['day'].unique())
samp = days[::7][:120]   # ~120일
parts=[]
for d in samp:
    dd = lab[lab.day==d]
    if len(dd)<300: continue
    parts.append(mtf_indicators(dd))
M = pd.concat(parts, ignore_index=True)
# 기존 라벨 join (같은 day,min) — day 를 plain str 로 캐스팅 (arrow/object 불일치 방지)
key=['day','min_of_day']
M['day']=M['day'].astype(str); M['min_of_day']=M['min_of_day'].astype('int64')
FULL=pd.read_parquet(LAB)
FULL['day']=FULL['day'].astype(str); FULL['min_of_day']=FULL['min_of_day'].astype('int64')
lab_cols=[c for c in FULL.columns if c not in ['yr','day','sec','min_of_day','mid']]
J = M.merge(FULL[key+lab_cols], on=key, how='inner')
mtf_cols = [c for c in M.columns if c not in ('day','min_of_day')]
# 컬럼별 NaN 비율 (60m macd 등은 하루 26봉 안 돼 전부 NaN → 제외)
nanrate = J[mtf_cols].isna().mean()
mtf_cols = [c for c in mtf_cols if nanrate[c] < 0.5]
print('제외(>50% NaN, 봉 부족):', [c for c in M.columns if c not in ('day','min_of_day') and nanrate[c]>=0.5])
print(f"[redundancy] {len(J)} 행 ({J.day.nunique()}일)")

print("\n===== 작업2 (게이트): MTF 봉 지표 vs 기존 47라벨 max|corr| =====")
print(f"{'MTF 지표':14s} | max|corr| 라벨 | 두번째")
import numpy as np
redundant=[]; independent=[]
for mc_ in mtf_cols:
    mm = J[mc_].notna()
    cors = {lc: abs(np.corrcoef(J.loc[mm,mc_], J.loc[mm,lc])[0,1]) for lc in lab_cols if J.loc[mm,lc].std()>0 and J.loc[mm,mc_].std()>0}
    top = sorted(cors.items(), key=lambda x:-x[1])[:2]
    tag = 'REDUNDANT' if top[0][1]>0.9 else ('약중복' if top[0][1]>0.7 else '독립후보')
    print(f"{mc_:14s} | {top[0][1]:.3f} {top[0][0]:12s} | {top[1][1]:.3f} {top[1][0]:10s} {tag}")
    (redundant if top[0][1]>0.9 else independent).append((mc_, top[0][1]))
print(f"\nredundant(>0.9): {len(redundant)} / {len(mtf_cols)} | 독립후보(<0.7): {sum(1 for _,c in independent if c<0.7)}")
ind_strong = [m for m,c in independent if c<0.7]
print(f"독립후보(<0.7): {ind_strong if ind_strong else '없음 — 8단계처럼 전부 redundant'}")
# 봉 지표끼리도 (5m RSI vs 15m RSI 등 — 봉 간 중복)
print("\n참고: 같은 지표 봉 간 corr (5m vs 15m vs 30m vs 60m):")
for base in ['rsi','macd','bpos','stoch']:
    cs=[c for c in mtf_cols if c.startswith(base)]
    if len(cs)>=2:
        cm=J[cs].corr().to_numpy(); off=cm[np.triu_indices(len(cs),1)]; off=off[~np.isnan(off)]
        print(f"  {base}: 봉간 corr {off.min():.2f}~{off.max():.2f}")
