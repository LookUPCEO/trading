#!/usr/bin/env python3
"""
[I] 전체기간 확장 — 작업1b/2/4: 전수 스캔 + 203일 vs 1198일 일관성 + 시기 분포.

⚠️ 확장+일관성만 (거래/예측 X).
- 일관성: 겹치는 날의 라벨 동일성. 단 bigflow_5m/bigflow_norm 은 causal 임계 소스가
  '이전 처리일 q95' 라서 STEP=6(6일 전) vs STEP=1(1일 전)이 정당하게 다름 — 분리 보고.
- 시기 분포: 연도/분기별 일수·행수, 빠진 날, 과대표 여부.
"""
import numpy as np
import pandas as pd

D = '/Users/mark/Desktop/Mark/mark19/research/i_labeling'
META = ['yr', 'day', 'sec', 'min_of_day', 'mid']

full = pd.read_parquet(f'{D}/labels.parquet')
sub = pd.read_parquet(f'{D}/labels_step6.parquet')
labels = [c for c in full.columns if c not in META]
print(f"[load] full {full.shape} ({full.day.nunique()}일), step6 {sub.shape} ({sub.day.nunique()}일)")

# ---- 작업1b: 전수 NaN/inf/극단 ----
X = full[labels].to_numpy(float)
n_inf = int(np.isinf(X).sum()); n_huge = int((np.abs(X) > 1e8).sum())
nan_rate = full[labels].isna().mean().sort_values(ascending=False)
print(f"\n[전수 스캔] inf={n_inf}, |x|>1e8={n_huge}")
print("NaN 상위 4:", {k: round(v, 4) for k, v in nan_rate.head(4).items()})
rng_viol = {
    'rsi_14': int(((full.rsi_14 < 0) | (full.rsi_14 > 100)).sum()),
    'stoch_k': int(((full.stoch_k < 0) | (full.stoch_k > 100)).sum()),
    'adx_14': int(((full.adx_14 < 0) | (full.adx_14 > 100)).sum()),
    'obi5': int((full.obi5.abs() > 1).sum()), 'flow_5m': int((full.flow_5m.abs() > 1 + 1e-9).sum()),
    'body_ratio': int((full.body_ratio.abs() > 1 + 1e-9).sum()),
}
print("범위 위반:", rng_viol, "(전부 0 이어야)")

# ---- 작업2: 겹치는 날 일관성 ----
common_days = sorted(set(sub.day.unique()) & set(full.day.unique()))
f2 = full[full.day.isin(common_days)].set_index(['day', 'min_of_day']).sort_index()
s2 = sub[sub.day.isin(common_days)].set_index(['day', 'min_of_day']).sort_index()
assert len(f2) == len(s2), (len(f2), len(s2))
expect_diff = ['bigflow_5m', 'bigflow_norm']   # causal 임계 소스가 처리 간격 의존 — 정당한 차이
bad, ok45 = [], []
for c in labels:
    a, b = f2[c].to_numpy(float), s2[c].to_numpy(float)
    m = ~(np.isnan(a) & np.isnan(b))
    mismatch_nan = int((np.isnan(a) != np.isnan(b)).sum())
    d = np.abs(a[m] - b[m])
    mx = float(np.nanmax(d)) if m.any() else 0.0
    if c in expect_diff:
        print(f"[일관성/예상차이] {c}: max|Δ|={mx:.4g} (causal 임계 소스 6일전→1일전 — 정당)")
    elif mx > 1e-12 or mismatch_nan > 0:
        bad.append((c, mx, mismatch_nan))
    else:
        ok45.append(c)
print(f"[일관성] 동일(1e-12): {len(ok45)}/45  |  불일치: {bad if bad else '없음'}")

# bigflow 차이의 크기 감각 (분포 영향)
for c in expect_diff:
    a, b = f2[c].to_numpy(float), s2[c].to_numpy(float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.any():
        r = np.corrcoef(a[m], b[m])[0, 1]
        print(f"  {c}: corr(step1,step6)={r:.4f} (값은 다르나 같은 신호인지)")

# ---- 작업4: 시기 분포 ----
print("\n[시기 분포]")
full['q'] = full.day.str[:7]
per_yr = full.groupby('yr').agg(days=('day', 'nunique'), rows=('day', 'size'))
print(per_yr.to_string())
all_days = pd.to_datetime(sorted(full.day.unique()))
gaps = pd.Series(all_days).diff().dt.days
big_gaps = [(str(all_days[i-1].date()), str(all_days[i].date()), int(g))
            for i, g in enumerate(gaps) if g and g > 3]
print(f"달력 공백(>3일): {big_gaps if big_gaps else '없음'}")
rows_per_day = full.groupby('day').size()
print(f"행/일: med {rows_per_day.median():.0f}, min {rows_per_day.min()} ({rows_per_day.idxmin()}), "
      f"max {rows_per_day.max()}")
mo = full.groupby(full.day.str[:7]).day.nunique()
print(f"월별 일수: min {mo.min()} ({mo.idxmin()}), max {mo.max()} — 과대표 여부")
