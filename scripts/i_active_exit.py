#!/usr/bin/env python3
"""
[I] 6-1 A — 능동청산 백테스트 (4h thr70 이벤트 79건만, 규칙 사전등록 R1~R5).

🚨 trailing 함정 방지: 청산 결정 at 분 m = 분 m 末까지 정보(라벨/vote/가격)만.
   청산가 = mid[m] (진입 관행과 동일 시점 정의 — 동등 취급, 차익 우대 없음).
   재예측(vote)은 그 분의 21차원 상태로 pool(이벤트 day-3 이전)에서 kNN — 미래 0.
fee: 전부 T+T 11bp (조기청산도 taker) — 비교는 순수 gross 경로 차이.
"""
import os, json
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB = '/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
K_CAND = 1000
N_IND = 100
EXCL_DAYS = 3
FEE = 11.0

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
    n = len(nrm)
    starts = np.searchsorted(drow, np.arange(len(days)))
    C = nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32)
    m23 = yr == 2023
    mu = C[m23].mean(0); S = np.cov((C[m23] - mu).T)
    w, V = np.linalg.eigh(S)
    W = (V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-6))) @ V.T).astype(np.float32)
    X = ((C - mu) @ W).astype(np.float32)
    xsq = (X * X).sum(1)
    # (day,minute) -> row
    rowmap = {}
    dm_keys = list(zip(drow.tolist(), mod.tolist()))
    for i, k in enumerate(dm_keys): rowmap[k] = i

    lab = pd.read_parquet(LAB, columns=['day', 'min_of_day', 'mid'])
    lab = lab[lab.day.isin(days)]
    mids = np.full((len(days), 1440), np.nan, np.float32)
    mids[lab['day'].map(day_ix).to_numpy(), lab['min_of_day'].to_numpy()] = lab['mid'].to_numpy(np.float32)
    fr240 = np.full(n, np.nan, np.float32)
    ok = mod + 240 <= 1439
    fr240[ok] = mids[drow[ok], mod[ok] + 240] / mids[drow[ok], mod[ok]] - 1
    fr30 = np.full(n, np.nan, np.float32)
    ok = mod + 30 <= 1439
    fr30[ok] = mids[drow[ok], mod[ok] + 30] / mids[drow[ok], mod[ok]] - 1

    # 4h thr70 이벤트 (v2 와 동일 정의)
    R = pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
    ok = (R['4h_n'] >= 70) & ~R['4h_frq'].isna() & (R['4h_frq'] != 0)
    s = R[ok]
    lean = (s['4h_fup'] >= .7) | (s['4h_fup'] <= .3)
    EV = s[lean].copy()
    EV['dir'] = np.where(EV['4h_fup'] >= .5, 1, -1)
    print(f"[events] {len(EV)} (4h thr70)")

    def vote_at(row_i, e, horizon, fr_arr):
        """분 row_i 시점, pool prefix e 에서 horizon-독립 kNN vote frac_up."""
        d2 = xsq[:e] - 2.0 * (X[:e] @ X[row_i])
        kc = min(K_CAND, e - 1)
        cand = np.argpartition(d2, kc)[:kc]
        order = cand[np.argsort(d2[cand])]
        sel = greedy_h(drow[order], mod[order], horizon, N_IND)
        picks = order[sel]
        v = fr_arr[picks]; v = v[~np.isnan(v)]; v = v[v != 0]
        if len(v) < 70: return np.nan
        return (v > 0).mean()

    RULES = ['FIXED', 'R1', 'R2', 'R3', 'R4', 'R5']
    res = {r: [] for r in RULES}
    exit_t = {r: [] for r in RULES}
    from time import time as _t
    t0 = _t()
    for ei, (_, ev) in enumerate(EV.iterrows()):
        q = int(ev.q); d = drow[q]; m0 = mod[q]; dr = int(ev['dir'])
        e = starts[max(d - EXCL_DAYS, 0)]
        p0 = mids[d, m0]
        # 매분 상태/vote (결정용 — 분 m 末 정보)
        path_exit = {r: None for r in RULES}
        for m in range(m0 + 1, m0 + 240):
            pnl = (mids[d, m] / p0 - 1) * 1e4 * dr
            need_vote = any(path_exit[r] is None for r in ['R1', 'R2', 'R3'])
            f30 = f240 = np.nan
            ri = rowmap.get((d, m))
            if need_vote and ri is not None:
                f30 = vote_at(ri, e, 30, fr30)
                f240 = vote_at(ri, e, 240, fr240)
            adv30 = f30 if dr > 0 else (1 - f30 if not np.isnan(f30) else np.nan)
            adv240 = f240 if dr > 0 else (1 - f240 if not np.isnan(f240) else np.nan)
            if path_exit['R1'] is None and not np.isnan(adv30) and adv30 <= 0.35:
                path_exit['R1'] = (m, pnl)
            if path_exit['R2'] is None and not np.isnan(adv240) and adv240 < 0.5 and pnl > 0:
                path_exit['R2'] = (m, pnl)
            if path_exit['R3'] is None and not np.isnan(adv240) and adv240 <= 0.4:
                path_exit['R3'] = (m, pnl)
            if path_exit['R4'] is None and pnl <= -50:
                path_exit['R4'] = (m, pnl)
            if path_exit['R5'] is None and pnl >= 100:
                path_exit['R5'] = (m, pnl)
        pnl_fix = (mids[d, m0 + 240] / p0 - 1) * 1e4 * dr if m0 + 240 <= 1439 else np.nan
        for r in RULES:
            if r == 'FIXED' or path_exit[r] is None:
                res[r].append(pnl_fix); exit_t[r].append(240)
            else:
                res[r].append(path_exit[r][1]); exit_t[r].append(path_exit[r][0] - m0)
        if ei % 20 == 0:
            print(f"  ev {ei}/{len(EV)} elapsed={_t()-t0:.0f}s", flush=True)

    print(f"\n===== 능동청산 vs 고정 4h hold (n={len(EV)}, net = gross − 11bp) =====")
    qd = EV.qday.to_numpy()
    out_rows = []
    for r in RULES:
        v = np.array(res[r]) - FEE
        m = ~np.isnan(v)
        dm = pd.Series(v[m]).groupby(qd[m]).mean().to_numpy()
        bs = np.random.default_rng(7).choice(dm, (4000, len(dm)), replace=True).mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        et = np.array(exit_t[r])[m]
        trig = (et < 240).mean()
        out_rows.append(dict(rule=r, n=int(m.sum()), net_evt=float(np.nanmean(v)),
                             net_day=float(dm.mean()), lo=lo, hi=hi,
                             trig_pct=float(trig * 100), med_exit_min=float(np.median(et))))
        print(f"{r:6s} net(evt) {np.nanmean(v):+7.1f} | day-mean {dm.mean():+7.1f} [{lo:+.1f},{hi:+.1f}] "
              f"| 발동 {trig*100:.0f}% | 보유 med {np.median(et):.0f}분")
    pd.DataFrame(out_rows).to_csv(f'{OUT}/active_exit_results.csv', index=False)
    # 시기 분해 (전/후반)
    half = np.median(qd)
    print("\n시기 분해 (이벤트 day 중앙값 기준 전/후반, day-mean net):")
    for r in RULES:
        v = np.array(res[r]) - FEE
        m = ~np.isnan(v)
        a = pd.Series(v[m & (qd <= half)]).groupby(qd[m & (qd <= half)]).mean().mean()
        b = pd.Series(v[m & (qd > half)]).groupby(qd[m & (qd > half)]).mean().mean()
        print(f"{r:6s} 전반 {a:+7.1f} | 후반 {b:+7.1f}")

if __name__ == '__main__':
    main()
