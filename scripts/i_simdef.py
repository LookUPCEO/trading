#!/usr/bin/env python3
"""
[I] 6단계 — "닮음" 정의 (라벨 부분공간) 비교. 사전등록된 31개 정의.

phase1 (선택, train 만): 2024Q1~2025Q2 쿼리 stride 40 screening.
phase2 (판정, OOS 만): top-3 + 베이스라인, 2024+ stride 10 full → 2025Q3~ 만 판정.
env: PHASE=1|2, DEFS=콤마구분 정의이름(phase2), STRIDE 오버라이드.
"""
import os, json, itertools
import numpy as np
import pandas as pd
from time import time as _t

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB = '/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
HORIZONS = {'30m': 30, '1h': 60, '4h': 240}
K_CAND = 1000
N_IND = 100
EXCL_DAYS = 3
MIN_VOTES = 70

GROUPS = {
    'T': ['ma_dev_5','ma_dev_15','ma_slope_30','ma_slope_120','ma_dev_240','ma_slope_240','rsi_30'],
    'V': ['atr_14','rv_ratio','adx_14'],
    'O': ['obi50','obi_wtd','dobi5_30','dobi5_60'],
    'F': ['flow_30','flow_1m','vol_z','bigflow_norm'],
    'C': ['body_ratio','upper_wick','lower_wick'],
}
def all_defs():
    ds = {}
    ks = list(GROUPS)
    for r in range(1, 6):
        for comb in itertools.combinations(ks, r):
            ds['+'.join(comb)] = sum((GROUPS[k] for k in comb), [])
    return ds   # 31개, 'T+V+O+F+C' = 베이스라인(21차원)

def greedy_h(od, om, h, n_target):
    acc = {}; out = []
    for i in range(len(od)):
        d = od[i]; m = om[i]
        lst = acc.get(d)
        if lst is not None:
            if any(abs(m - mm) < h for mm in lst): continue
            lst.append(m)
        else:
            acc[d] = [m]
        out.append(i)
        if len(out) >= n_target: break
    return out

def main():
    phase = os.environ.get('PHASE', '1')
    stride = int(os.environ.get('STRIDE', '40' if phase == '1' else '10'))
    nrm = pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(
        ['day', 'min_of_day']).reset_index(drop=True)
    yr = nrm['yr'].astype(int).to_numpy()
    mod = nrm['min_of_day'].to_numpy()
    days = sorted(nrm['day'].unique()); day_ix = {d: i for i, d in enumerate(days)}
    drow = nrm['day'].map(day_ix).to_numpy()
    month = nrm['day'].str[5:7].astype(int).to_numpy()
    n = len(nrm)
    starts = np.searchsorted(drow, np.arange(len(days)))
    m23 = yr == 2023

    lab = pd.read_parquet(LAB, columns=['day', 'min_of_day', 'mid'])
    lab = lab[lab.day.isin(days)]
    mids = np.full((len(days), 1440), np.nan, np.float32)
    mids[lab['day'].map(day_ix).to_numpy(), lab['min_of_day'].to_numpy()] = lab['mid'].to_numpy(np.float32)
    FR = {}
    for hname, h in HORIZONS.items():
        fr = np.full(n, np.nan, np.float32)
        ok = mod + h <= 1439
        fr[ok] = mids[drow[ok], mod[ok] + h] / mids[drow[ok], mod[ok]] - 1
        FR[hname] = fr

    qtr = np.char.add(yr.astype(str), np.char.add('Q', ((month - 1) // 3 + 1).astype(str)))
    TRAIN_Q = {'2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2'}
    is_train = np.isin(qtr, list(TRAIN_Q))

    if phase == '1':
        q_mask = (yr >= 2024) & is_train & (mod % stride == 5)
    else:
        q_mask = (yr >= 2024) & (mod % stride == 5)
    qs = np.where(q_mask)[0]
    defs = all_defs()
    if os.environ.get('DEFS'):
        want = os.environ['DEFS'].split(',')
        defs = {k: v for k, v in defs.items() if k in want}
    print(f"[setup] phase={phase} stride={stride} queries={len(qs)} defs={len(defs)}", flush=True)

    summary = []
    for dname, cols in defs.items():
        t0 = _t()
        C = nrm[[f'z_{c}' for c in cols]].to_numpy(np.float32)
        mu = C[m23].mean(0); S = np.cov((C[m23] - mu).T)
        S = np.atleast_2d(S)
        w, V = np.linalg.eigh(S)
        W = (V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-6))) @ V.T).astype(np.float32)
        X = ((C - mu) @ W).astype(np.float32)
        xsq = (X * X).sum(1)
        recs = []
        BLK = 192
        for bi in range(0, len(qs), BLK):
            qb = qs[bi:bi + BLK]
            ends = starts[np.maximum(drow[qb] - EXCL_DAYS, 0)]
            emax = ends.max()
            if emax < 50000: continue
            d2 = xsq[None, :emax] - 2.0 * (X[qb] @ X[:emax].T)
            for j, q in enumerate(qb):
                e = ends[j]
                if e < 50000: continue
                row = d2[j, :e]
                kc = min(K_CAND, e - 1)
                cand = np.argpartition(row, kc)[:kc]
                order = cand[np.argsort(row[cand])]
                od, om = drow[order], mod[order]
                rec = dict(q=int(q), qday=int(drow[q]), quarter=qtr[q], train=bool(is_train[q]))
                for hname, h in HORIZONS.items():
                    sel = greedy_h(od, om, h, N_IND)
                    picks = order[sel]
                    v = FR[hname][picks]; v = v[~np.isnan(v)]; v = v[v != 0]
                    rec[f'{hname}_n'] = len(v)
                    rec[f'{hname}_fup'] = (v > 0).mean() if len(v) else np.nan
                    rec[f'{hname}_frq'] = float(FR[hname][q])
                recs.append(rec)
        R = pd.DataFrame(recs)
        R.to_parquet(f'{OUT}/simdef/def_{dname.replace("+","")}_p{phase}.parquet')
        # 지표 (사전등록): thr65 결합 day-mean gross / thr70 결합 day-mean net
        met = {}
        for thr, fee, key in [(.65, 0, 'g65'), (.70, 11, 'n70')]:
            Ls, rs = [], []
            for hname in HORIZONS:
                ok = (R[f'{hname}_n'] >= MIN_VOTES) & ~R[f'{hname}_frq'].isna() & (R[f'{hname}_frq'] != 0)
                s = R[ok]
                lean = (s[f'{hname}_fup'] >= thr) | (s[f'{hname}_fup'] <= 1 - thr)
                L = s[lean]
                sgn = np.where(L[f'{hname}_fup'] >= .5, 1., -1.)
                Ls.append(L); rs.append(sgn * L[f'{hname}_frq'].to_numpy() * 1e4 - fee)
            Lc = pd.concat(Ls); vals = np.concatenate(rs)
            if len(Lc) < 5:
                met[key] = np.nan; met[key + '_n'] = len(Lc); continue
            dm = pd.Series(vals).groupby(Lc.qday.to_numpy()).mean()
            met[key] = float(dm.mean()); met[key + '_n'] = len(Lc)
        summary.append(dict(definition=dname, dims=len(cols), **met, sec=round(_t() - t0)))
        print(f"  [{dname}] dims={len(cols)} g65={met.get('g65', float('nan')):+.2f}(n={met['g65_n']}) "
              f"n70={met.get('n70', float('nan')):+.2f}(n={met['n70_n']}) {round(_t()-t0)}s", flush=True)
    S = pd.DataFrame(summary).sort_values('g65', ascending=False)
    S.to_csv(f'{OUT}/simdef/phase{phase}_summary.csv', index=False)
    print(S.to_string(index=False))

if __name__ == '__main__':
    os.makedirs(f'{OUT}/simdef', exist_ok=True)
    main()
