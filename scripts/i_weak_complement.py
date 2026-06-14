#!/usr/bin/env python3
"""[I] 10-2 — 약신호域(thr<0.70) 내 fee 넘는 보완. 약밴드 동의 = 사전식별 필터.
horizon 5개 10짝. 6-4 OOS 전멸 전력 → 강한 보정+OOS."""
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
qday = R.qday.to_numpy(); tr = R.is_tr.to_numpy()

def get(h):
    f = R[f'{h}_fup'].to_numpy(); n = R[f'{h}_n'].to_numpy(); frq = R[f'{h}_frq'].to_numpy()
    st = np.where(n >= 70, np.maximum(f, 1 - f), np.nan)
    d = np.where(f >= .5, 1, -1)
    err = np.where(np.isnan(frq) | (frq == 0), np.nan, (d * frq < 0).astype(float))
    return st, d, frq, err
G = {h: get(h) for h in H5}
frq4 = R['4h_frq'].to_numpy(); v4 = ~np.isnan(frq4) & (frq4 != 0)

def ci(qd, net):
    if len(net) < 5: return np.nan, np.nan, np.nan
    dm = pd.Series(net).groupby(qd).mean().to_numpy()
    bs = np.random.default_rng(7).choice(dm, (5000, len(dm)), replace=True).mean(axis=1)
    return dm.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

print("[load]", len(R), "쿼리")
print("\n===== 작업1: 짝별 오답상관 strength 밴드별 (약신호域, 강신호 제외) =====")
BANDS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70)]
print(f"{'짝':10s} " + " ".join(f"[{a:.2f},{b:.2f})" for a, b in BANDS))
for a, b in [('30m', '4h'), ('1h', '4h'), ('45m', '2h'), ('30m', '1h')]:
    sa, _, _, ea = G[a]; sb, _, _, eb = G[b]
    row = f"{a}-{b:6s} "
    for lo, hi in BANDS:
        # 두 horizon 다 이 밴드 안 (약신호 동시)
        m = (sa >= lo) & (sa < hi) & (sb >= lo) & (sb < hi) & ~np.isnan(ea) & ~np.isnan(eb)
        if m.sum() < 30: row += f"  n{m.sum():<4d}   "; continue
        c = np.corrcoef(ea[m], eb[m])[0, 1]
        row += f" {c:+.2f}({m.sum():4d})"
    print(row)

print("\n===== 작업2+3: 약밴드 동의 필터 (두 horizon strength∈[BAND,0.70) 동방향) vs 단일 =====")
print("4h hold 실현. '동의'=둘 다 약하게 같은방향 (강신호 0.70+ 제외).")
rows = []
for a, b in PAIRS:
    sa, da, _, _ = G[a]; sb, db, _, _ = G[b]
    for blo in [0.55, 0.60, 0.62]:
        weak_a = (sa >= blo) & (sa < 0.70)
        weak_b = (sb >= blo) & (sb < 0.70)
        agree = weak_a & weak_b & (da == db) & v4
        if agree.sum() < 20:
            rows.append(dict(pair=f"{a}+{b}", blo=blo, n=int(agree.sum()), skip=True)); continue
        net = da[agree] * frq4[agree] * 1e4 - FEE
        ntr = agree & tr; nte = agree & ~tr
        net_tr = (da[ntr] * frq4[ntr] * 1e4 - FEE)
        net_te = (da[nte] * frq4[nte] * 1e4 - FEE)
        rows.append(dict(pair=f"{a}+{b}", blo=blo, n=int(agree.sum()),
                         hit=float((net > 0).mean()), net=float(net.mean()),
                         daily=float(net.sum() / nd),
                         net_tr=float(net_tr.mean()) if len(net_tr) else np.nan,
                         daily_te=float(net_te.sum() / nd_te) if len(net_te) else np.nan,
                         n_te=int(nte.sum()), skip=False))
T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/weak_complement.csv', index=False)
good = T[~T.skip].sort_values('net_tr', ascending=False)
print(f"유효 셀(n>=20) {len(good)} / 30 (Bonferroni 분모 30)")
print("상위 train net 12 (train 선택용):")
print(good.head(12)[['pair','blo','n','hit','net','net_tr','daily','n_te','daily_te']].round(2).to_string(index=False))

print("\n===== 작업4: train 최선 → OOS 판정 (day-cluster CI) =====")
# 단일 약신호 베이스 (한 horizon 만 약하게, 4h hold) — 비교용
for h in ['1h', '4h']:
    sa, da, _, _ = G[h]
    m = (sa >= 0.60) & (sa < 0.70) & v4
    net = da[m] * frq4[m] * 1e4 - FEE
    print(f"  [단일 약신호 {h} 0.60-0.70]: n={m.sum()} hit{(net>0).mean():.3f} net{net.mean():+.1f} 일수익{net.sum()/nd:+.2f}")
print()
for _, r in good[good.net_tr > 0].head(5).iterrows():
    a, b = r.pair.split('+'); sa, da, _, _ = G[a]; sb, db, _, _ = G[b]
    agree = (sa >= r.blo) & (sa < 0.70) & (sb >= r.blo) & (sb < 0.70) & (da == db) & v4
    te = agree & ~tr
    net_te = da[te] * frq4[te] * 1e4 - FEE
    dm, lo, hi = ci(qday[te], net_te)
    tag = "✓생존" if lo > 0 else "✗"
    print(f"  {r.pair} blo{r.blo}: train net {r.net_tr:+.1f} → test n={te.sum()} hit{(net_te>0).mean():.3f} "
          f"net{net_te.mean():+.1f} day[{lo:+.0f},{hi:+.0f}]{tag} 일수익{net_te.sum()/nd_te:+.2f}")
print(f"\n현행: 단일4h thr0.70 full 8.40/test 2.90, 결합 15.1/test 8.1. Bonferroni 분모 30.")
print("6-4 전력: 약신호 이익 부분집합 OOS 0/5 전멸.")
