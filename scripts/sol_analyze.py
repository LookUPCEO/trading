#!/usr/bin/env python3
"""[SOL] 2단계 분석 — 부호일치 + 70% + hit/net. SOL/ETH182 동일 코드.
whitening = 가장 이른 WHIT_DAYS 일 fit (SOL 2023 없음 → ETH 도 동일 윈도우 = 공정).
env: SIM_OUT (labels_norm_reduced.parquet 위치), LABP (원 라벨), WHIT_DAYS(45), TAG."""
import os, json
import numpy as np
import pandas as pd

SIM_OUT = os.environ['SIM_OUT']
LABP = os.environ['LABP']
WHIT_DAYS = int(os.environ.get('WHIT_DAYS', '45'))
TAG = os.environ.get('TAG', 'sol')
HORIZONS = {'30m': 30, '1h': 60, '4h': 240}
K_CAND = 1000; N_IND = 100; EXCL_DAYS = 3; MIN_VOTES = 70; FEE = 11.0
rng = np.random.default_rng(11)

def greedy_h(od, om, h, nt):
    acc = {}; out = []
    for i in range(len(od)):
        d = od[i]; m = om[i]; lst = acc.get(d)
        if lst is not None:
            if any(abs(m - mm) < h for mm in lst): continue
            lst.append(m)
        else: acc[d] = [m]
        out.append(i)
        if len(out) >= nt: break
    return out

def main():
    nrm = pd.read_parquet(f'{SIM_OUT}/labels_norm_reduced.parquet').sort_values(
        ['day', 'min_of_day']).reset_index(drop=True)
    meta = json.load(open(f'{SIM_OUT}/reduce_norm_meta.json'))
    reps = [r for r in meta['reps'] if r != 'spread_bp']
    mod = nrm['min_of_day'].to_numpy()
    days = sorted(nrm['day'].unique()); day_ix = {d: i for i, d in enumerate(days)}
    drow = nrm['day'].map(day_ix).to_numpy()
    n = len(nrm)
    starts = np.searchsorted(drow, np.arange(len(days)))
    C = nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32)
    # whitening: 가장 이른 WHIT_DAYS 일 fit
    whit = drow < WHIT_DAYS
    mu = C[whit].mean(0); S = np.cov((C[whit] - mu).T)
    w, V = np.linalg.eigh(S)
    W = (V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-6))) @ V.T).astype(np.float32)
    X = ((C - mu) @ W).astype(np.float32)
    xsq = (X * X).sum(1)
    print(f"[{TAG}] DB {n}행 {len(days)}일, whiten fit {WHIT_DAYS}일, reps {len(reps)}")

    lab = pd.read_parquet(LABP, columns=['day', 'min_of_day', 'mid'])
    lab = lab[lab.day.isin(days)]
    mids = np.full((len(days), 1440), np.nan, np.float32)
    mids[lab['day'].map(day_ix).to_numpy(), lab['min_of_day'].to_numpy()] = lab['mid'].to_numpy(np.float32)
    FR = {}
    for hn, h in HORIZONS.items():
        fr = np.full(n, np.nan, np.float32); ok = mod + h <= 1439
        fr[ok] = mids[drow[ok], mod[ok] + h] / mids[drow[ok], mod[ok]] - 1
        FR[hn] = fr

    # 쿼리: whitening 윈도우 이후 + pool>=EXCL 충분 (drow >= max(WHIT,30)), 10분 격자
    q = np.where((drow >= max(WHIT_DAYS, 30)) & (mod % 10 == 5))[0]
    print(f"[{TAG}] queries {len(q)}")
    reps_signed = [r for r in reps if r in meta['signed']]
    sidx = [reps.index(c) for c in reps_signed]

    recs = []; agree_top = []; agree_rnd = []
    BLK = 160
    from time import time as _t
    t0 = _t()
    for bi in range(0, len(q), BLK):
        qb = q[bi:bi + BLK]
        ends = starts[np.maximum(drow[qb] - EXCL_DAYS, 0)]
        emax = ends.max()
        if emax < 5000: continue
        d2 = xsq[None, :emax] - 2.0 * (X[qb] @ X[:emax].T)
        for j, qi in enumerate(qb):
            e = ends[j]
            if e < 5000: continue
            row = d2[j, :e]; kc = min(K_CAND, e - 1)
            cand = np.argpartition(row, kc)[:kc]
            order = cand[np.argsort(row[cand])]
            od, om = drow[order], mod[order]
            rcand = rng.choice(e, min(400, e), replace=False)
            rec = dict(q=int(qi), qday=int(drow[qi]), dayidx=int(drow[qi]))
            # 부호일치 (top100 day-dedupe 무관, 단순 100 최근접 vs random)
            top100 = order[:100]
            qs_ = np.sign(X[qi, sidx]) if False else None
            for hn, h in HORIZONS.items():
                sel = greedy_h(od, om, h, N_IND); picks = order[sel]
                v = FR[hn][picks]; v = v[~np.isnan(v)]; v = v[v != 0]
                rec[f'{hn}_n'] = len(v); rec[f'{hn}_fup'] = (v > 0).mean() if len(v) else np.nan
                rec[f'{hn}_frq'] = float(FR[hn][qi])
            recs.append(rec)
            # 부호일치 (정규화공간 C, signed dims)
            qsig = np.sign(C[qi, sidx])
            agree_top.append((np.sign(C[np.ix_(top100, sidx)]) == qsig).mean())
            rnd100 = rcand[:100]
            agree_rnd.append((np.sign(C[np.ix_(rnd100, sidx)]) == qsig).mean())
        if bi % (BLK * 20) == 0: print(f"  {bi}/{len(q)} {_t()-t0:.0f}s", flush=True)
    R = pd.DataFrame(recs); R.to_parquet(f'{SIM_OUT}/lean_{TAG}.parquet')

    print(f"\n[{TAG}] 부호일치 (signed {len(sidx)}차원): top {np.median(agree_top):.3f} vs random {np.median(agree_rnd):.3f}")

    def events(df, h, thr):
        ok = (df[f'{h}_n'] >= MIN_VOTES) & ~df[f'{h}_frq'].isna() & (df[f'{h}_frq'] != 0)
        s = df[ok]; lean = (s[f'{h}_fup'] >= thr) | (s[f'{h}_fup'] <= 1 - thr)
        L = s[lean]; sgn = np.where(L[f'{h}_fup'] >= .5, 1., -1.)
        return L, sgn * L[f'{h}_frq'].to_numpy() * 1e4 - FEE
    def ci(L, net):
        if len(L) < 5: return np.nan, np.nan, np.nan
        dm = pd.Series(net).groupby(L.qday.to_numpy()).mean().to_numpy()
        bs = np.random.default_rng(7).choice(dm, (4000, len(dm)), replace=True).mean(axis=1)
        return dm.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print(f"\n[{TAG}] thr 곡선 (결합 30m+1h+4h, net=gross-11)")
    nd = R.qday.nunique()
    for thr in [0.62, 0.65, 0.68, 0.70, 0.72]:
        Ls, vs = [], []
        for h in HORIZONS:
            L, net = events(R, h, thr); Ls.append(L); vs.append(net)
        Lc = pd.concat(Ls); vc = np.concatenate(vs)
        if len(Lc) < 5: print(f"  thr{thr}: n={len(Lc)}"); continue
        dm, lo, hi = ci(Lc, vc)
        print(f"  thr{thr:.2f}: n={len(Lc):4d} hit{(vc+FEE>0).mean():.3f} net {vc.mean():+6.1f} "
              f"day {dm:+6.1f} [{lo:+.1f},{hi:+.1f}] 일수익 {vc.sum()/nd:+.2f}bp/day")

    print(f"\n[{TAG}] horizon별 thr0.70 hit/net + audit")
    for h in HORIZONS:
        L, net = events(R, h, 0.70)
        if len(L) < 5: print(f"  {h}: n={len(L)}"); continue
        dm, lo, hi = ci(L, net)
        med = np.median(net); top3 = np.sort(np.abs(net))[-3:]
        net_x = net[np.argsort(np.abs(net))[:-3]] if len(net) > 3 else net
        print(f"  {h}: n={len(L)} hit{(net+FEE>0).mean():.3f} net{net.mean():+.1f} "
              f"day[{lo:+.1f},{hi:+.1f}] med{med:+.1f} top3제외{net_x.mean():+.1f}")

if __name__ == '__main__':
    main()
