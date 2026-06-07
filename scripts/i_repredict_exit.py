#!/usr/bin/env python3
"""
[I] 6-2 — 능동청산 재구현: 재예측 갱신 (사전등록 V1/V2/V3). 익절/손절 X.

- 진입 = 4h thr70 이벤트 (v2, 79건). 진입 leg fee 5.5bp.
- 매분 m: 그 분 末 21차원 상태로 진입과 동일 엔진 kNN → fup240.
  같은방향 thr70 재쏠림 → 청산예정 = m+240 (연장). 반대 thr70 → flip (+2 legs).
  V2: fup ∈ [0.45,0.55] → 즉시 청산. V3: fup 0.5 반대편 → 즉시 청산.
  그 외 → 예정시점 청산. day-end 1439 강제청산 (within-day).
- fee = 5.5bp × legs (방향 유지 = fee 0 — 6-1 의 '매분 fee' 오해 정정).
- lookahead: 결정 at m = m 末 정보만. 가격 = mid[m] (진입/청산 동일 관행).
"""
import os, json
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB = '/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
K_CAND = 1000; N_IND = 100; EXCL_DAYS = 3
LEG = 5.5

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
    nrm = pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(
        ['day', 'min_of_day']).reset_index(drop=True)
    meta = json.load(open(f'{OUT}/reduce_norm_meta.json'))
    reps = [r for r in meta['reps'] if r != 'spread_bp']
    yr = nrm['yr'].astype(int).to_numpy()
    mod = nrm['min_of_day'].to_numpy()
    days = sorted(nrm['day'].unique()); day_ix = {d: i for i, d in enumerate(days)}
    drow = nrm['day'].map(day_ix).to_numpy()
    starts = np.searchsorted(drow, np.arange(len(days)))
    C = nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32)
    m23 = yr == 2023
    mu = C[m23].mean(0); S = np.cov((C[m23] - mu).T)
    w, V = np.linalg.eigh(S)
    W = (V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-6))) @ V.T).astype(np.float32)
    X = ((C - mu) @ W).astype(np.float32)
    xsq = (X * X).sum(1)
    rowmap = {}
    for i, k in enumerate(zip(drow.tolist(), mod.tolist())): rowmap[k] = i

    lab = pd.read_parquet(LAB, columns=['day', 'min_of_day', 'mid'])
    lab = lab[lab.day.isin(days)]
    mids = np.full((len(days), 1440), np.nan, np.float32)
    mids[lab['day'].map(day_ix).to_numpy(), lab['min_of_day'].to_numpy()] = lab['mid'].to_numpy(np.float32)
    n = len(nrm)
    fr240 = np.full(n, np.nan, np.float32)
    okm = mod + 240 <= 1439
    fr240[okm] = mids[drow[okm], mod[okm] + 240] / mids[drow[okm], mod[okm]] - 1

    R = pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
    ok = (R['4h_n'] >= 70) & ~R['4h_frq'].isna() & (R['4h_frq'] != 0)
    s = R[ok]
    lean = (s['4h_fup'] >= .7) | (s['4h_fup'] <= .3)
    EV = s[lean].copy()
    EV['dir'] = np.where(EV['4h_fup'] >= .5, 1, -1)
    print(f"[events] {len(EV)}")

    fup_cache = {}
    def fup_at(d, m, e):
        key = (d, m)
        if key in fup_cache: return fup_cache[key]
        ri = rowmap.get((d, m))
        if ri is None:
            fup_cache[key] = np.nan; return np.nan
        d2 = xsq[:e] - 2.0 * (X[:e] @ X[ri])
        kc = min(K_CAND, e - 1)
        cand = np.argpartition(d2, kc)[:kc]
        order = cand[np.argsort(d2[cand])]
        sel = greedy_h(drow[order], mod[order], 240, N_IND)
        picks = order[sel]
        v = fr240[picks]; v = v[~np.isnan(v)]; v = v[v != 0]
        out = (v > 0).mean() if len(v) >= 70 else np.nan
        fup_cache[key] = out
        return out

    VARIANTS = ['V1', 'V2', 'V3']
    res = {v: [] for v in VARIANTS}
    stats = {v: dict(flips=0, exts=0, dies=0, legs=0, hold=[]) for v in VARIANTS}
    from time import time as _t
    t0 = _t()
    for ei, (_, ev) in enumerate(EV.iterrows()):
        q = int(ev.q); d = drow[q]; m0 = mod[q]; dr0 = int(ev['dir'])
        e = starts[max(d - EXCL_DAYS, 0)]
        for var in VARIANTS:
            pos_dir = dr0; entry_m = m0; entry_p = mids[d, m0]
            planned = min(m0 + 240, 1439)
            legs = 1   # 진입 leg
            pnl = 0.0
            m = m0
            ext = fl = 0
            while True:
                m += 1
                if m >= planned or m >= 1439 or np.isnan(mids[d, m]):
                    xm = min(planned, 1439, m)
                    while np.isnan(mids[d, xm]) and xm > entry_m: xm -= 1
                    pnl += (mids[d, xm] / entry_p - 1) * 1e4 * pos_dir
                    legs += 1
                    break
                f = fup_at(d, m, e)
                if np.isnan(f):
                    continue
                adv = f if pos_dir > 0 else 1 - f
                if adv >= 0.70:
                    planned = min(m + 240, 1439); ext += 1          # 연장 (fee 0)
                elif adv <= 0.30:
                    pnl += (mids[d, m] / entry_p - 1) * 1e4 * pos_dir  # flip
                    legs += 2
                    pos_dir = -pos_dir; entry_p = mids[d, m]; entry_m = m
                    planned = min(m + 240, 1439); fl += 1
                elif var == 'V2' and 0.45 <= f <= 0.55:
                    pnl += (mids[d, m] / entry_p - 1) * 1e4 * pos_dir
                    legs += 1; stats[var]['dies'] += 1
                    break
                elif var == 'V3' and adv < 0.5:
                    pnl += (mids[d, m] / entry_p - 1) * 1e4 * pos_dir
                    legs += 1; stats[var]['dies'] += 1
                    break
            net = pnl - LEG * legs
            res[var].append(dict(qday=d, net=net, legs=legs, flips=fl, exts=ext,
                                 hold=m - m0, quarter=ev.quarter))
            stats[var]['flips'] += fl; stats[var]['exts'] += ext
            stats[var]['legs'] += legs; stats[var]['hold'].append(m - m0)
        if ei % 20 == 0:
            print(f"  ev {ei}/{len(EV)} elapsed={_t()-t0:.0f}s cache={len(fup_cache)}", flush=True)

    # FIXED 비교 (동일 이벤트, 2 legs)
    fixed = []
    for _, ev in EV.iterrows():
        q = int(ev.q); d = drow[q]; m0 = mod[q]; dr0 = int(ev['dir'])
        pnl = (mids[d, m0 + 240] / mids[d, m0] - 1) * 1e4 * dr0
        fixed.append(dict(qday=d, net=pnl - 11.0))
    F = pd.DataFrame(fixed)

    def daymean_ci(df):
        dm = df.groupby('qday')['net'].mean().to_numpy()
        bs = np.random.default_rng(7).choice(dm, (4000, len(dm)), replace=True).mean(axis=1)
        return dm.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print(f"\n===== 재예측 갱신 vs FIXED (n=79 이벤트 체인, leg fee 5.5bp) =====")
    dmean, lo, hi = daymean_ci(F)
    print(f"FIXED  day-mean net {dmean:+7.1f} [{lo:+.1f},{hi:+.1f}] | legs 2.0 | hold 240분")
    rows = []
    tq = ['2025Q3', '2025Q4', '2026Q1', '2026Q2']
    for var in VARIANTS:
        D = pd.DataFrame(res[var])
        dmean, lo, hi = daymean_ci(D)
        st = stats[var]
        oos = D[D.quarter.isin(tq)]
        oosm = oos.groupby('qday')['net'].mean().mean() if len(oos) else np.nan
        print(f"{var}     day-mean net {dmean:+7.1f} [{lo:+.1f},{hi:+.1f}] | legs/evt {st['legs']/len(D):.2f} "
              f"| flips {st['flips']} | 연장 {st['exts']} | 소멸청산 {st['dies']} "
              f"| hold med {np.median(st['hold']):.0f}분 | OOS(2025Q3+) {oosm:+.1f}")
        rows.append(dict(variant=var, net=dmean, lo=lo, hi=hi, legs=st['legs']/len(D),
                         flips=st['flips'], exts=st['exts'], dies=st['dies'],
                         hold_med=float(np.median(st['hold'])), oos=float(oosm)))
        D.to_parquet(f'{OUT}/repredict_{var}.parquet')
    pd.DataFrame(rows).to_csv(f'{OUT}/repredict_summary.csv', index=False)
    # 일수익 (851 쿼리일 기준)
    qdays = R.qday.nunique()
    print(f"\n일수익 (851일): FIXED {F.net.sum()/qdays:+.2f}bp/day | " +
          " | ".join(f"{v} {pd.DataFrame(res[v]).net.sum()/qdays:+.2f}" for v in VARIANTS))

if __name__ == '__main__':
    main()
