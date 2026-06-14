#!/usr/bin/env python3
"""[I] 10-1 — fee 넘는+보완 특정 짝. horizon 5개 10짝 전수, 강신호 오답상관 표본 최대."""
import itertools
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
FEE = 11.0
TRAIN_Q = ['2024Q1', '2024Q2', '2024Q3', '2024Q4', '2025Q1', '2025Q2']
H5 = ['30m', '45m', '1h', '2h', '4h']
PAIRS = list(itertools.combinations(H5, 2))

v2 = pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
hf = pd.read_parquet(f'{OUT}/lean70_v2_per_query_hfine.parquet')
R = v2.merge(hf[['q'] + [c for c in hf.columns if c.split('_')[0] in ('45m', '2h')]], on='q', how='inner')
R['is_tr'] = R.quarter.isin(TRAIN_Q)
nd = R.qday.nunique(); nd_te = R[~R.is_tr].qday.nunique()
print(f"[load] {len(R)} 쿼리 ({nd}일), {len(PAIRS)}짝 (Bonferroni 분모 10)")

def info(df, h, thr):
    f = df[f'{h}_fup'].to_numpy(); n = df[f'{h}_n'].to_numpy(); frq = df[f'{h}_frq'].to_numpy()
    st = np.where(n >= 70, np.maximum(f, 1 - f), np.nan)
    d = np.where(f >= .5, 1, -1)
    fire = (st >= thr) & ~np.isnan(st)
    correct = (d * frq > 0)
    correct = np.where(np.isnan(frq) | (frq == 0), False, correct)
    valid = ~np.isnan(frq) & (frq != 0)
    return d, fire & valid, correct, frq

def ci(qday, net):
    if len(net) < 5: return np.nan, np.nan, np.nan
    dm = pd.Series(net).groupby(qday).mean().to_numpy()
    bs = np.random.default_rng(7).choice(dm, (5000, len(dm)), replace=True).mean(axis=1)
    return dm.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

print("\n===== 작업1: 짝별 강신호 오답 분할표 (thr0.70 동시발화) =====")
print("짝            동시 | 둘맞 A만 B만 둘틀 | A틀릴때B맞 B틀릴때A맞 | 표본충분?")
comp_pairs = []
for a, b in PAIRS:
    da, fa, ca, _ = info(R, a, 0.70); db, fb, cb, _ = info(R, b, 0.70)
    both = fa & fb
    if both.sum() < 5:
        print(f"{a}-{b:4s} {both.sum():4d} | 동시발화 부족"); continue
    n11 = (ca & cb & both).sum(); n10 = (ca & ~cb & both).sum()
    n01 = (~ca & cb & both).sum(); n00 = (~ca & ~cb & both).sum()
    aw = (~ca & both); bw = (~cb & both)
    pA = cb[aw].mean() if aw.sum() else np.nan   # A틀릴때 B맞
    pB = ca[bw].mean() if bw.sum() else np.nan
    flag = "보완후보" if (pA > 0.5 or pB > 0.5) and both.sum() >= 15 else ""
    print(f"{a}-{b:4s} {both.sum():4d} | {n11:4d}{n10:4d}{n01:4d}{n00:4d} | {pA:.2f}({aw.sum():2d})  {pB:.2f}({bw.sum():2d}) | {flag}")
    if (pA > 0.5 or pB > 0.5): comp_pairs.append((a, b, both.sum()))

print("\n  thr0.65 동시발화로 표본 늘려 (0.65 자체는 fee 미달 — 보완성 안정성만):")
for a, b in [('1h', '4h'), ('2h', '4h'), ('1h', '2h')]:
    da, fa, ca, _ = info(R, a, 0.65); db, fb, cb, _ = info(R, b, 0.65)
    both = fa & fb
    if both.sum() < 5: print(f"  {a}-{b}: n={both.sum()}"); continue
    aw = (~ca & both); pA = cb[aw].mean() if aw.sum() else np.nan
    print(f"  {a}-{b}: 동시 {both.sum()}, A틀릴때 B맞 {pA:.2f} (n_Awrong={aw.sum()})")

print(f"\n===== 작업2~4: 보완후보 짝 선택 운용 (4h hold, 단일 4h 대비) =====")
# 단일 4h 베이스라인
d4, f4, c4, frq4 = info(R, '4h', 0.70)
net4 = d4[f4] * frq4[f4] * 1e4 - FEE
dm4, lo4, hi4 = ci(R.qday.to_numpy()[f4], net4)
print(f"베이스라인 단일4h: n={f4.sum()} hit{c4[f4].mean():.3f} day{dm4:+.1f}[{lo4:+.0f},{hi4:+.0f}] 일수익 {net4.sum()/nd:+.2f} test {(d4[f4&~R.is_tr.to_numpy()]*frq4[f4&~R.is_tr.to_numpy()]*1e4-FEE).sum()/nd_te:+.2f}")
print()
seen = set()
for a, b, ncof in comp_pairs:
    if (a, b) in seen: continue
    seen.add((a, b))
    da, fa, ca, frqa = info(R, a, 0.70); db, fb, cb, frqb = info(R, b, 0.70)
    # 합집합(짝 한정): 둘 중 하나라도 발화 → 그 horizon(둘 다면 4h쪽=긴쪽) 4h hold
    union = fa | fb
    # 방향: 발화한 것 (둘 다면 합의; 충돌이면 패스)
    diru = np.where(fb, db, da)   # 4h(b) 우선
    conflict = fa & fb & (da != db)
    use = union & ~conflict
    frq4_ = R['4h_frq'].to_numpy(); v = ~np.isnan(frq4_) & (frq4_ != 0)
    m = use & v
    net = diru[m] * frq4_[m] * 1e4 - FEE
    dm, lo, hi = ci(R.qday.to_numpy()[m], net)
    mte = m & ~R.is_tr.to_numpy()
    nette = diru[mte] * frq4_[mte] * 1e4 - FEE
    # 교집합(둘 다 동방향)
    inter = fa & fb & (da == db)
    mi = inter & v
    neti = da[mi] * frq4_[mi] * 1e4 - FEE
    print(f"[{a}+{b}] 합집합: n={m.sum()} hit{(net>0).mean():.3f} day{dm:+.1f}[{lo:+.0f},{hi:+.0f}] 일수익 full{net.sum()/nd:+.2f} test{nette.sum()/nd_te:+.2f}")
    if mi.sum() >= 5:
        dmi, loi, hii = ci(R.qday.to_numpy()[mi], neti)
        print(f"        교집합(동방향): n={mi.sum()} hit{(neti>0).mean():.3f} day{dmi:+.1f}[{loi:+.0f},{hii:+.0f}] 일수익 {neti.sum()/nd:+.2f}")
print(f"\n현행: 단일4h full 8.40/test 2.90, 결합 15.1/test 8.1 bp/day. Bonferroni 분모 10짝.")
