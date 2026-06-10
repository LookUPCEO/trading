#!/usr/bin/env python3
"""[I] 6-5 — thr 0.685~0.71 소수점 정밀 (점프 위치). 기존 v2 parquet fup 재임계.
표본 한계: 소수점 빈 n 작음 → 각 빈 n + Wilson CI 필수, 노이즈 명시."""
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
H = ['30m', '1h', '4h']
TRAIN_Q = ['2024Q1', '2024Q2', '2024Q3', '2024Q4', '2025Q1', '2025Q2']
FEE = 11.0

R = pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
R['is_train'] = R.quarter.isin(TRAIN_Q)

def build(df):
    evs = []
    for h in H:
        ok = (df[f'{h}_n'] >= 70) & ~df[f'{h}_frq'].isna() & (df[f'{h}_frq'] != 0)
        s = df[ok]
        st = np.maximum(s[f'{h}_fup'].to_numpy(), 1 - s[f'{h}_fup'].to_numpy())
        sgn = np.where(s[f'{h}_fup'] >= .5, 1, -1)
        evs.append(pd.DataFrame(dict(h=h, qday=s.qday.to_numpy(), train=s.is_train.to_numpy(),
                                     votes=s[f'{h}_n'].to_numpy(), strength=st,
                                     net=sgn * s[f'{h}_frq'].to_numpy() * 1e4 - FEE)))
    return pd.concat(evs, ignore_index=True)

E = build(R)
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c - h, c + h)

print("===== 작업1: strength 0.685~0.71 소수점 곡선 (결합, 평균 X — 분포) =====")
print("bin            |   n  | win%(net>0) Wilson95     | med net | mean net")
edges = np.arange(0.685, 0.7125, 0.0025)
for i in range(len(edges) - 1):
    lo, hi = edges[i], edges[i+1]
    s = E[(E.strength >= lo) & (E.strength < hi)]
    if len(s) == 0:
        print(f"[{lo:.4f},{hi:.4f}) |    0 | -"); continue
    k = int((s.net > 0).sum()); wl, wh = wilson(k, len(s))
    print(f"[{lo:.4f},{hi:.4f}) | {len(s):4d} | {k/len(s)*100:5.1f}% [{wl*100:4.1f},{wh*100:4.1f}] | "
          f"{s.net.median():+7.1f} | {s.net.mean():+7.1f}")

print("\n실제 격자값 (votes 양자화 — strength 가 취하는 값들, 0.68~0.72):")
vals = np.sort(E[(E.strength >= 0.68) & (E.strength <= 0.72)].strength.unique())
print(" ".join(f"{v:.4f}" for v in vals[:30]))

print("\n===== 작업2: 점프 위치 — strength 정렬 rolling 승률 (윈도 150) =====")
Es = E.sort_values('strength').reset_index(drop=True)
w = 150
for center in [0.66, 0.67, 0.68, 0.685, 0.69, 0.695, 0.70, 0.705, 0.71]:
    idx = np.searchsorted(Es.strength.to_numpy(), center)
    a, b = max(0, idx - w), min(len(Es), idx + w)
    seg = Es.iloc[a:b]
    k = int((seg.net > 0).sum()); wl, wh = wilson(k, len(seg))
    print(f"strength≈{center:.3f}: 주변 {len(seg)}건 승률 {k/len(seg)*100:.1f}% [{wl*100:.1f},{wh*100:.1f}] "
          f"med net {seg.net.median():+.1f}")

print("\n===== 작업3: OOS — 점프 위치 train vs test =====")
for split, df in [('TRAIN', E[E.train]), ('TEST', E[~E.train])]:
    print(f"[{split}]  strength: ", end='')
    for lo, hi in [(0.66, 0.685), (0.685, 0.70), (0.70, 0.715), (0.715, 1.01)]:
        s = df[(df.strength >= lo) & (df.strength < hi)]
        if len(s) < 5: print(f"[{lo:.3f},{hi:.3f}) n={len(s)} -  ", end=''); continue
        k = int((s.net > 0).sum())
        print(f"[{lo:.3f},{hi:.3f}) n={len(s)} win{k/len(s)*100:.0f}% med{s.net.median():+.0f}  ", end='')
    print()

print("\n===== 작업4: 빈도/일수익 — '더 일찍' thr 후보 (strength>=X, 전체기간 결합) =====")
nd = R.qday.nunique()
for X in [0.685, 0.690, 0.695, 0.70]:
    s = E[E.strength >= X]
    st = E[(E.strength >= X) & (~E.train)]
    ndte = R[~R.is_train].qday.nunique()
    # day-cluster CI (전체)
    dm = s.groupby('qday')['net'].mean().to_numpy()
    bs = np.random.default_rng(7).choice(dm, (4000, len(dm)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"strength>={X:.3f}: 전체 n={len(s)} 일수익 {s.net.sum()/nd:+.2f}bp/day [CI {lo:+.1f},{hi:+.1f}] "
          f"| test n={len(st)} {st.net.sum()/ndte:+.2f}bp/day")
print(f"\n현행 baseline strength>=0.70 = {E[E.strength>=0.70].net.sum()/nd:+.2f}bp/day (전체)")
