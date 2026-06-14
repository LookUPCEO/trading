#!/usr/bin/env python3
"""[I] 9단계 — 합의 신호 두 길 (A 드물게 크게 / B 합의+thr낮춤).
horizon 5개 {30m,45m,1h,2h,4h} (v2+hfine 병합, 전부 stage5 thr0.70 net 양수).
합의깊이 k = 같은방향 발화 horizon 수. 거래 실현 = 1h·4h hold."""
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
FEE = 11.0
TRAIN_Q = ['2024Q1', '2024Q2', '2024Q3', '2024Q4', '2025Q1', '2025Q2']
H5 = ['30m', '45m', '1h', '2h', '4h']
TRADE_H = ['1h', '4h']

v2 = pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
hf = pd.read_parquet(f'{OUT}/lean70_v2_per_query_hfine.parquet')
cols_hf = ['q'] + [c for c in hf.columns if c.split('_')[0] in ('45m', '2h')]
R = v2.merge(hf[cols_hf], on='q', how='inner')
nd = R.qday.nunique()
print(f"[load] merged {len(R)} 쿼리 ({nd}일), horizons {H5}")

def consensus(df, thr):
    sig = np.zeros((len(df), len(H5)), np.int8)
    for j, h in enumerate(H5):
        f = df[f'{h}_fup'].to_numpy()
        st = np.where(df[f'{h}_n'].to_numpy() >= 70, np.maximum(f, 1 - f), np.nan)
        sd = np.where(f >= .5, 1, -1)
        fire = (st >= thr) & ~np.isnan(st)
        sig[:, j] = np.where(fire, sd, 0)
    cdir = np.sign(sig.sum(1))
    k = np.where(cdir > 0, (sig > 0).sum(1), np.where(cdir < 0, (sig < 0).sum(1), 0))
    return cdir, k.astype(int)

def net_at(df, cdir, mask, trade_h):
    frq = df[f'{trade_h}_frq'].to_numpy()
    ok = mask & ~np.isnan(frq) & (frq != 0)
    return df['qday'].to_numpy()[ok], cdir[ok] * frq[ok] * 1e4 - FEE

def ci(qday, net):
    if len(net) < 5: return np.nan, np.nan, np.nan
    dm = pd.Series(net).groupby(qday).mean().to_numpy()
    bs = np.random.default_rng(7).choice(dm, (5000, len(dm)), replace=True).mean(axis=1)
    return dm.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

# ===== 작업1: thr0.70 합의깊이 k별 =====
print("\n===== 작업1: thr0.70 합의깊이 k별 hit/net (day-mean CI) =====")
cdir, k = consensus(R, 0.70)
for kk in [1, 2, 3, 4, 5]:
    m = (k == kk) if kk < 5 else (k >= 5)
    line = f"k={kk} ({int(m.sum()):4d}q): "
    for th in TRADE_H:
        qd, net = net_at(R, cdir, m, th)
        if len(net) < 5: line += f"{th} n<5  "; continue
        dm, lo, hi = ci(qd, net)
        line += f"{th} n={len(net):3d} hit{(net+FEE>0).mean():.2f} net{net.mean():+6.1f} day[{lo:+.0f},{hi:+.0f}] | "
    print(line)

# ===== 작업2: 길 A — 고합의 자본 크게 =====
print("\n===== 작업2: 길 A — 고합의(3+/4+) 4h 실현, size 배수 (size=1 기준) =====")
print("(일수익 = Σnet/일수 × size. size 는 Kelly/파산위험 별도 — 레버 날조 X)")
for kth in [3, 4]:
    m = k >= kth
    qd, net = net_at(R, cdir, m, '4h')
    if len(net) < 5: print(f"  {kth}+ 합의: n={len(net)} (<5)"); continue
    dm, lo, hi = ci(qd, net)
    te = R.quarter.to_numpy()[None]  # placeholder
    daily = net.sum() / nd
    # OOS
    okte = ~pd.Series(qd).isin([])  # qday 는 dayidx 아님; quarter 별 분해는 아래 작업4
    print(f"  {kth}+ 합의: n={len(net)} ({len(np.unique(qd))}일) hit{(net+FEE>0).mean():.2f} "
          f"per-trade net {net.mean():+.1f} [{lo:+.0f},{hi:+.0f}] | 일수익(size1) {daily:+.2f}bp/day")
print(f"  현행 단일4h thr0.70 = 15.1bp/day (full), test 8.1")

# ===== 작업3: 길 B — 합의 × thr 낮춤 =====
print("\n===== 작업3: 길 B — '합의된 낮은 thr' vs '단독 thr' (4h 실현) =====")
print("thr  | solo(k>=1) n/hit/net day | 합의 2+ n/hit/net day | 합의 3+ n/hit/net day")
for thr in [0.65, 0.68, 0.70]:
    cd, kk = consensus(R, thr)
    cells = []
    for kth in [1, 2, 3]:
        m = kk >= kth
        qd, net = net_at(R, cd, m, '4h')
        if len(net) < 5: cells.append("n<5"); continue
        dm, lo, hi = ci(qd, net)
        cells.append(f"n{len(net):4d} h{(net+FEE>0).mean():.2f} {dm:+5.1f}[{lo:+.0f},{hi:+.0f}]")
    print(f"{thr:.2f} | {cells[0]:28s} | {cells[1]:28s} | {cells[2]}")

# ===== 작업4: 합의깊이 × thr 일수익 히트맵 (train/test) =====
print("\n===== 작업4: 일수익 bp/day (4h 실현) — train | test (2025Q3~) =====")
te_mask = ~R.quarter.isin(TRAIN_Q)
nd_tr = R[~te_mask].qday.nunique(); nd_te = R[te_mask].qday.nunique()
print("thr  k>= | " + "  ".join(f"k{kk}" for kk in [1, 2, 3]))
rows = []
ntests = 0
for thr in [0.65, 0.68, 0.70]:
    cd, kk = consensus(R, thr)
    for kth in [1, 2, 3]:
        m = kk >= kth
        for split, smask, ndd in [('tr', ~te_mask, nd_tr), ('te', te_mask, nd_te)]:
            qd, net = net_at(R, cd, m & smask.to_numpy(), '4h')
            daily = net.sum() / ndd if len(net) else np.nan
            rows.append(dict(thr=thr, k=kth, split=split, n=len(net), daily=daily,
                             hit=float((net+FEE>0).mean()) if len(net) else np.nan))
        ntests += 1
T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/consensus_grid.csv', index=False)
piv = T.pivot_table(index=['thr', 'k'], columns='split', values='daily')
print(piv.round(2).to_string())
print(f"\nBonferroni 분모 (thr×k 셀) = {ntests} (×실현 2 = {ntests*2})")
# OOS 생존: test daily > 단일4h test(8.1) & n 충분
print("\n현행 단일4h: full 15.1 / test 8.1 bp/day. 이를 test 에서 넘는 셀:")
win = T[(T.split=='te') & (T.daily > 8.1) & (T.n >= 20)]
if len(win): print(win.to_string(index=False))
else: print("  없음 (합의/thr 조합이 test 에서 현행 못 넘음)")
