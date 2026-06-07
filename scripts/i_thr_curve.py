#!/usr/bin/env python3
"""[I] 6-3 — thr 곡선 (0.60~0.72) + 균형점 train→OOS 검증. 기존 v2 parquet 재임계."""
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
H = ['30m', '1h', '4h']
THRS = [round(0.60 + 0.01 * i, 2) for i in range(13)]
TRAIN_Q = ['2024Q1', '2024Q2', '2024Q3', '2024Q4', '2025Q1', '2025Q2']
FEE = 11.0

R = pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
R['is_train'] = R.quarter.isin(TRAIN_Q)

def events(df, h, thr):
    ok = (df[f'{h}_n'] >= 70) & ~df[f'{h}_frq'].isna() & (df[f'{h}_frq'] != 0)
    s = df[ok]
    lean = (s[f'{h}_fup'] >= thr) | (s[f'{h}_fup'] <= 1 - thr)
    L = s[lean]
    sgn = np.where(L[f'{h}_fup'] >= .5, 1., -1.)
    return L, sgn * L[f'{h}_frq'].to_numpy() * 1e4 - FEE

def stats(df, h, thr, ndays):
    L, net = events(df, h, thr)
    if len(L) < 5:
        return dict(n=len(L), hit=np.nan, net=np.nan, daily=np.nan)
    return dict(n=len(L), hit=float((net + FEE > 0).mean()), net=float(net.mean()),
                daily=float(net.sum() / ndays))

def ci(df, h_list, thr):
    Ls, vs = [], []
    for h in h_list:
        L, net = events(df, h, thr)
        Ls.append(L); vs.append(net)
    Lc = pd.concat(Ls); vc = np.concatenate(vs)
    if len(Lc) < 5: return np.nan, np.nan, np.nan, 0
    dm = pd.Series(vc).groupby(Lc.qday.to_numpy()).mean().to_numpy()
    bs = np.random.default_rng(7).choice(dm, (4000, len(dm)), replace=True).mean(axis=1)
    return dm.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(Lc)

tr = R[R.is_train]; te = R[~R.is_train]
nd_tr = tr.qday.nunique(); nd_te = te.qday.nunique(); nd_all = R.qday.nunique()
print(f"train {nd_tr}일 / test {nd_te}일 / 전체 {nd_all}일")

print("\n===== 작업1: thr 곡선 (train | test) — net/trade bp, 일수익 bp/day =====")
rows = []
for h in H + ['COMB']:
    print(f"\n[{h}]")
    print("thr  | train: n    net  daily | test:  n    net  daily")
    for thr in THRS:
        if h == 'COMB':
            sa = {k: 0.0 for k in ['n', 'net', 'daily']}
            for src, nm, nd in [(tr, 'tr', nd_tr), (te, 'te', nd_te)]:
                Ls, vs = [], []
                for hh in H:
                    L, net = events(src, hh, thr)
                    Ls.append(L); vs.append(net)
                vc = np.concatenate(vs)
                sa[nm] = dict(n=len(vc), net=float(vc.mean()) if len(vc) else np.nan,
                              daily=float(vc.sum() / nd))
            a, b = sa['tr'], sa['te']
        else:
            a = stats(tr, h, thr, nd_tr); b = stats(te, h, thr, nd_te)
        rows.append(dict(h=h, thr=thr, n_tr=a['n'], net_tr=a['net'], daily_tr=a['daily'],
                         n_te=b['n'], net_te=b['net'], daily_te=b['daily']))
        print(f"{thr:.2f} | {a['n']:5d} {a['net'] if a['net']==a['net'] else float('nan'):+7.1f} {a['daily']:+6.2f} "
              f"| {b['n']:5d} {b['net'] if b['net']==b['net'] else float('nan'):+7.1f} {b['daily']:+6.2f}")
T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/thr_curve.csv', index=False)

print("\n===== 작업2+3: 균형점 (train argmax, net>0 제한) → OOS 판정 =====")
for h in H + ['COMB']:
    sub = T[(T.h == h) & (T.net_tr > 0)]
    if len(sub) == 0:
        print(f"{h}: train net>0 thr 없음"); continue
    best = sub.loc[sub.daily_tr.idxmax()]
    thr = best.thr
    hl = H if h == 'COMB' else [h]
    dm_tr, lo_tr, hi_tr, ntr = ci(tr, hl, thr)
    dm_te, lo_te, hi_te, nte = ci(te, hl, thr)
    te_row = T[(T.h == h) & (T.thr == thr)].iloc[0]
    # test 에서의 순위 (train argmax 가 test 에서도 상위인가)
    sub_te = T[(T.h == h)].dropna(subset=['daily_te'])
    rank_te = (sub_te.daily_te > te_row.daily_te).sum() + 1
    print(f"{h:4s}: train균형 thr={thr:.2f} (daily {best.daily_tr:+.2f}, n={best.n_tr}) → "
          f"test daily {te_row.daily_te:+.2f} (전체 {len(sub_te)}개 중 {rank_te}위) | "
          f"test day-mean net {dm_te:+.1f} [{lo_te:+.1f},{hi_te:+.1f}] (n={nte})")

print("\n===== 작업4: 목표 대비 (전체기간, 참고용 thr70 베이스라인 vs 균형 후보) =====")
for h, thr in [('4h', 0.70), ('COMB', 0.70), ('COMB', 0.68), ('COMB', 0.66)]:
    hl = H if h == 'COMB' else [h]
    Ls, vs = [], []
    for hh in hl:
        L, net = events(R, hh, thr)
        Ls.append(L); vs.append(net)
    vc = np.concatenate(vs)
    print(f"{h} thr{thr:.2f}: 전체 일수익 {vc.sum()/nd_all:+.2f}bp/day (n={len(vc)}) — 목표 50bp 의 {vc.sum()/nd_all/50*100:.0f}%")
