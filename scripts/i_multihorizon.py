#!/usr/bin/env python3
"""[I] 7단계 — 시기×horizon 구조 + horizon 상관 + multi-horizon 동시운용.
기존 lean70_v2_per_query.parquet (81,682 쿼리, 30m/1h/4h fup+frq) 재사용."""
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
H = ['30m', '1h', '4h']
FEE = 11.0
TRAIN_Q = ['2024Q1', '2024Q2', '2024Q3', '2024Q4', '2025Q1', '2025Q2']
R = pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')

def ev(df, h, thr=0.70):
    ok = (df[f'{h}_n'] >= 70) & ~df[f'{h}_frq'].isna() & (df[f'{h}_frq'] != 0)
    s = df[ok]; lean = (s[f'{h}_fup'] >= thr) | (s[f'{h}_fup'] <= 1 - thr)
    L = s[lean]; sgn = np.where(L[f'{h}_fup'] >= .5, 1., -1.)
    return L, sgn, sgn * L[f'{h}_frq'].to_numpy() * 1e4 - FEE
def ci(qday, net):
    if len(net) < 5: return np.nan, np.nan, np.nan
    dm = pd.Series(net).groupby(np.asarray(qday)).mean().to_numpy()
    bs = np.random.default_rng(7).choice(dm, (4000, len(dm)), replace=True).mean(axis=1)
    return dm.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

print("===== 작업1: 시기 × horizon (분기별 thr0.70 net, day-mean) =====")
qtrs = sorted(R.quarter.unique())
print("h    | " + " ".join(f"{q[2:]:>8s}" for q in qtrs))
for h in H:
    row = []
    for qt in qtrs:
        L, _, net = ev(R[R.quarter == qt], h)
        if len(L) < 8: row.append("   -    "); continue
        dm = pd.Series(net).groupby(L.qday.to_numpy()).mean().mean()
        row.append(f"{dm:+5.0f}({len(L):2d})")
    print(f"{h:4s} | " + " ".join(row))
# 최근(2025Q3+) 30m 독립 분석
print("\n최근 2025Q3~ 각 horizon (폴드 통계):")
te = R[~R.quarter.isin(TRAIN_Q)]
for h in H:
    L, _, net = ev(te, h)
    dm, lo, hi = ci(L.qday.to_numpy(), net)
    print(f"  {h}: n={len(L)} net day {dm:+.1f} [{lo:+.1f},{hi:+.1f}] hit {(net+FEE>0).mean():.3f}")

print("\n===== 작업2: horizon 간 상관 (독립 정보인가 중복인가) =====")
# 모든 쿼리에서 fup 의 horizon 간 corr (votes>=70 공통)
ok = (R['30m_n'] >= 70) & (R['1h_n'] >= 70) & (R['4h_n'] >= 70)
sub = R[ok]
print("fup pairwise corr:")
for a, b in [('30m', '1h'), ('30m', '4h'), ('1h', '4h')]:
    print(f"  {a}-{b}: {np.corrcoef(sub[f'{a}_fup'], sub[f'{b}_fup'])[0,1]:.3f}")
# thr0.70 신호 동시 발생률 + 같은 방향률
sig = {}
for h in H:
    s = sub.copy()
    s[f'{h}_sig'] = np.where(s[f'{h}_fup'] >= .7, 1, np.where(s[f'{h}_fup'] <= .3, -1, 0))
    sub = s
for a, b in [('30m', '1h'), ('30m', '4h'), ('1h', '4h')]:
    both = ((sub[f'{a}_sig'] != 0) & (sub[f'{b}_sig'] != 0))
    if both.sum() > 0:
        agree = (sub.loc[both, f'{a}_sig'] == sub.loc[both, f'{b}_sig']).mean()
        pa = (sub[f'{a}_sig'] != 0).mean(); pb = (sub[f'{b}_sig'] != 0).mean()
        joint = both.mean()
        print(f"  {a}&{b} 동시신호: {both.sum()}건 (독립기대 {pa*pb*len(sub):.0f}, lift {joint/(pa*pb):.1f}x), 같은방향 {agree:.2f}")

print("\n===== 작업3+4: multi-horizon 동시운용 (양수 horizon 30m/1h/4h, 자본 1/3씩) =====")
# 각 horizon 독립 이벤트 → (qday, net/3). 합산 = 포트폴리오. 단일 4h 대비.
nd = R.qday.nunique()
def portfolio(df, label):
    rows = []
    for h in H:
        L, _, net = ev(df, h)
        for qd, nt in zip(L.qday.to_numpy(), net):
            rows.append((qd, nt))  # 자본배분은 일수익 환산에서
    P = pd.DataFrame(rows, columns=['qday', 'net'])
    ndd = df.qday.nunique()
    # 일수익: 동시운용 = 각 신호에 자본 1/3 (3 horizon 분산) → Σnet/3/일수
    daily_full = P.net.sum() / ndd          # 자본 전액 매 신호 (중첩 무시 상한)
    daily_split = P.net.sum() / 3 / ndd      # 1/3 분산 (현실)
    dm, lo, hi = ci(P.qday.to_numpy(), P.net)
    # 단일 4h
    L4, _, net4 = ev(df, '4h')
    d4 = net4.sum() / ndd
    print(f"[{label}] 동시 n={len(P)} (4h 단독 {len(L4)}) | per-trade net {P.net.mean():+.1f} "
          f"[{lo:+.1f},{hi:+.1f}]")
    print(f"  일수익: 동시(자본전액) {daily_full:+.2f} / 동시(1/3분산) {daily_split:+.2f} / 단일4h {d4:+.2f} bp/day "
          f"({ndd}일)")
    return P
P_all = portfolio(R, '전체기간')
P_te = portfolio(te, '2025Q3~ OOS')

print("\n===== 작업4b: 합의(2+ horizon 같은방향) 강화 효과 =====")
sub['nsig'] = (sub['30m_sig'] != 0).astype(int) + (sub['1h_sig'] != 0).astype(int) + (sub['4h_sig'] != 0).astype(int)
# 합의 = 2+ horizon 신호 & 같은 방향
for k in [1, 2, 3]:
    m = sub.nsig >= k
    if m.sum() < 5: print(f"  {k}+ horizon 신호: n={m.sum()}"); continue
    # 방향 = 신호 horizon 다수결, net = 4h 미래 (대표) — 합의시 hit 비교
    s2 = sub[m]
    # 같은 방향 합의만
    dirsum = s2['30m_sig'] + s2['1h_sig'] + s2['4h_sig']
    consensus = np.abs(dirsum) >= k  # k개가 같은방향
    sc = s2[consensus]
    if len(sc) < 5: print(f"  {k}+ 합의: n={len(sc)}"); continue
    d = np.sign(sc['30m_sig'] + sc['1h_sig'] + sc['4h_sig'])
    # 1h 미래로 hit (중간 horizon 대표)
    net1h = d.to_numpy() * sc['1h_frq'].to_numpy() * 1e4 - FEE
    m2 = ~np.isnan(net1h)
    print(f"  {k}+ horizon 동방향 합의: n={m2.sum()} hit(1h) {(net1h[m2]+FEE>0).mean():.3f} net {net1h[m2].mean():+.1f}")

print("\n===== 작업5: audit (전체기간 동시 포트폴리오) =====")
med = np.median(P_all.net); top3 = P_all.net.abs().nlargest(3).sum()
nox = P_all.net[~P_all.net.abs().isin(P_all.net.abs().nlargest(3))]
print(f"  per-trade: mean {P_all.net.mean():+.1f} med {med:+.1f} top3제외 {nox.mean():+.1f}")
print(f"  multiple testing: horizon 조합 시도 = 단일3 + 합의(1+/2+/3+) = 보고 전부 (cherry-pick X)")
