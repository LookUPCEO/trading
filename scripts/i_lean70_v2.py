#!/usr/bin/env python3
"""
[I] 3단계 v2 (horizon 기반 독립) + 4단계 (hit rate + 폭 + net).

독립 기준 교정 (사용자 지적): day당 1개 = 과처리.
  올바름 = "미래 창 비겹침": 같은 day 안에서 |Δ분| ≥ horizon 이면 독립
  (다른 day = 자동 독립 — 미래는 day 경계를 안 넘음). horizon 별 별도 선택.
  greedy: 거리 오름차순으로 후보 훑으며 같은 day 의 기수락 매치와 |Δ| < h 면 거부.
  null(random 매치)도 같은 제약 — 공정 대조.

기록 (per query × horizon): n_ind, frac_up, null frac_up, causal base, 쿼리 자신의
미래수익 fr_q (bp 아님, 비율). hit/net/판정은 분석부에서 threshold 적용.
쿼리 미래는 pool(과거 day) 밖 = 구조적 OOS. 진입 가정: 라벨시각 직후 (다음 분).
"""
import os, json
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB = '/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
HORIZONS = {'5m': 5, '10m': 10, '30m': 30, '1h': 60, '4h': 240}
K_CAND = 1000
N_IND = 100
EXCL_DAYS = 3
MIN_VOTES = 70
Q_STRIDE = 10
rng = np.random.default_rng(11)

def greedy_h(order_days, order_mins, h, n_target):
    """거리 오름차순 후보 (day, minute) → 같은 day |Δ|<h 거부 greedy. index 리스트 반환."""
    acc_day = {}
    out = []
    for i in range(len(order_days)):
        d = order_days[i]; m = order_mins[i]
        lst = acc_day.get(d)
        if lst is not None:
            ok = True
            for mm in lst:
                if abs(m - mm) < h:
                    ok = False; break
            if not ok: continue
            lst.append(m)
        else:
            acc_day[d] = [m]
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
    month = nrm['day'].str[5:7].astype(int).to_numpy()
    n = len(nrm)
    print(f"[load] DB {n} rows, {len(days)} days")

    C = nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32)
    m23 = yr == 2023
    mu = C[m23].mean(0); S = np.cov((C[m23] - mu).T)
    w, V = np.linalg.eigh(S)
    W = (V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-6))) @ V.T).astype(np.float32)
    X = ((C - mu) @ W).astype(np.float32)
    xsq = (X * X).sum(1)
    starts = np.searchsorted(drow, np.arange(len(days)))

    lab = pd.read_parquet(LAB, columns=['day', 'min_of_day', 'mid'])
    lab = lab[lab.day.isin(days)]
    mids = np.full((len(days), 1440), np.nan, np.float32)
    mids[lab['day'].map(day_ix).to_numpy(), lab['min_of_day'].to_numpy()] = \
        lab['mid'].to_numpy(np.float32)
    FR = {}
    for hname, h in HORIZONS.items():
        fr = np.full(n, np.nan, np.float32)
        ok = mod + h <= 1439
        fr[ok] = mids[drow[ok], mod[ok] + h] / mids[drow[ok], mod[ok]] - 1
        FR[hname] = fr
    base_cum = {}
    for hname in HORIZONS:
        up = (FR[hname] > 0).astype(np.int64)
        va = (~np.isnan(FR[hname]) & (FR[hname] != 0)).astype(np.int64)
        base_cum[hname] = (np.cumsum(np.bincount(drow, up * va, minlength=len(days))),
                           np.cumsum(np.bincount(drow, va, minlength=len(days))))

    q_mask = (yr >= 2024) & (mod % Q_STRIDE == 5)
    qs = np.where(q_mask)[0]
    qmax = int(os.environ.get('QMAX', '0'))
    if qmax: qs = qs[rng.choice(len(qs), qmax, replace=False)]; qs.sort()
    print(f"[query] {len(qs)} queries")

    recs = []
    BLK = 128
    from time import time as _t
    t0 = _t()
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
            # null 후보 (random 순서, 같은 제약)
            rcand = rng.choice(e, min(3 * N_IND + 200, e), replace=False)
            rd, rm = drow[rcand], mod[rcand]
            rec = dict(q=int(q), qday=int(drow[q]), qyr=int(yr[q]),
                       quarter=f"{yr[q]}Q{(month[q]-1)//3+1}", pool_days=int(drow[q]-EXCL_DAYS))
            for hname, h in HORIZONS.items():
                sel = greedy_h(od, om, h, N_IND)
                picks = order[sel]
                rsel = greedy_h(rd, rm, h, N_IND)
                rpicks = rcand[rsel]
                v = FR[hname][picks]; v = v[~np.isnan(v)]; v = v[v != 0]
                rv_ = FR[hname][rpicks]; rv_ = rv_[~np.isnan(rv_)]; rv_ = rv_[rv_ != 0]
                cu, cv = base_cum[hname]
                di = max(drow[q] - EXCL_DAYS - 1, 0)
                rec[f'{hname}_n'] = len(v)
                rec[f'{hname}_fup'] = (v > 0).mean() if len(v) else np.nan
                rec[f'{hname}_ndays'] = len(np.unique(drow[picks]))
                rec[f'{hname}_rnd_n'] = len(rv_)
                rec[f'{hname}_rnd_fup'] = (rv_ > 0).mean() if len(rv_) else np.nan
                rec[f'{hname}_base'] = cu[di] / max(cv[di], 1)
                rec[f'{hname}_frq'] = float(FR[hname][q])   # 쿼리 자신의 미래 (OOS)
            recs.append(rec)
        if bi % (BLK * 50) == 0:
            print(f"  q {bi}/{len(qs)} elapsed={_t()-t0:.0f}s", flush=True)
    R = pd.DataFrame(recs)
    R.to_parquet(f'{OUT}/lean70_v2_per_query.parquet')
    print(f"[done] {len(R)} queries, elapsed={_t()-t0:.0f}s")
    analyze(R)

def analyze(R):
    from scipy.stats import binom
    H = list(HORIZONS)
    print("\n===== 작업1: horizon 기반 유효 N (day당1개 대비 복원) =====")
    for h in H:
        nn = R[f'{h}_n']; nd = R[f'{h}_ndays']
        print(f"{h:3s}: 독립매치 med {nn.median():.0f} | 고유 day med {nd.median():.0f} "
              f"(이전 day-dedupe 는 매치=고유day=100)")

    print("\n===== 작업2: 쏠림 재측정 (real/null, thr 60/65/70/75) =====")
    for h in H:
        ok = R[f'{h}_n'] >= MIN_VOTES
        f = R.loc[ok, f'{h}_fup']
        fr_ = R.loc[ok & (R[f'{h}_rnd_n'] >= MIN_VOTES), f'{h}_rnd_fup']
        line = f"{h:3s} ({ok.sum()}): "
        for thr in [.60, .65, .70, .75]:
            lr = ((f >= thr) | (f <= 1-thr)).mean()*100
            ln = ((fr_ >= thr) | (fr_ <= 1-thr)).mean()*100
            line += f"t{int(thr*100)} {lr:.2f}/{ln:.2f}%  "
        print(line)
    tr = R.quarter.isin(['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2'])
    print("OOS thr70 (train→test): ", end='')
    for h in H:
        ok = R[f'{h}_n'] >= MIN_VOTES
        a = R[ok & tr]; b = R[ok & ~tr]
        la = (((a[f'{h}_fup'] >= .7) | (a[f'{h}_fup'] <= .3)).mean())*100
        lb = (((b[f'{h}_fup'] >= .7) | (b[f'{h}_fup'] <= .3)).mean())*100
        print(f"{h} {la:.2f}→{lb:.2f}%", end='  ')
    print()

    print("\n===== 작업3+4: hit rate + 폭 + net (쿼리 실제 미래, 구조적 OOS) =====")
    print("(net/trade bp: gross − fee. fee RT: T+T 11 / M+T 7.5 / M+M 4 — non-VIP)")
    rows = []
    for h in H:
        for thr in [.65, .70]:
            ok = (R[f'{h}_n'] >= MIN_VOTES) & ~R[f'{h}_frq'].isna() & (R[f'{h}_frq'] != 0)
            s = R[ok]
            lean = (s[f'{h}_fup'] >= thr) | (s[f'{h}_fup'] <= 1-thr)
            L = s[lean]
            if len(L) < 10:
                rows.append((h, thr, len(L), np.nan, np.nan, np.nan, np.nan, np.nan)); continue
            sgn = np.where(L[f'{h}_fup'] >= .5, 1.0, -1.0)
            ret = sgn * L[f'{h}_frq'].to_numpy() * 1e4   # signed gross bp
            hit = (ret > 0).mean()
            ebase = np.where(sgn > 0, L[f'{h}_base'], 1 - L[f'{h}_base']).mean()
            # cluster bootstrap (day 단위, day-mean 재추출 — day 동일가중 근사)
            bdays = L.qday.to_numpy()
            dm = pd.Series(ret).groupby(bdays).mean().to_numpy()
            rng2 = np.random.default_rng(7)
            bs = rng2.choice(dm, (2000, len(dm)), replace=True).mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append((h, thr, len(L), hit, ebase, ret.mean(), lo, hi))
    T = pd.DataFrame(rows, columns=['h','thr','n_lean','hit','base_hit','gross_bp','ci_lo','ci_hi'])
    T['net_TT'] = T.gross_bp - 11; T['net_MT'] = T.gross_bp - 7.5; T['net_MM'] = T.gross_bp - 4
    print(T.round(3).to_string(index=False))
    T.to_csv(f'{OUT}/lean70_v2_hit_net.csv', index=False)

    print("\n===== 작업5: 시기별 hit/gross (thr70; 최근 기준) =====")
    for h in H:
        ok = (R[f'{h}_n'] >= MIN_VOTES) & ~R[f'{h}_frq'].isna() & (R[f'{h}_frq'] != 0)
        s = R[ok]
        lean = (s[f'{h}_fup'] >= .7) | (s[f'{h}_fup'] <= .3)
        L = s[lean]
        if len(L) < 10: continue
        sgn = np.where(L[f'{h}_fup'] >= .5, 1, -1)
        ret = sgn * L[f'{h}_frq'].to_numpy() * 1e4
        line = f"{h:3s}: "
        for y in [2024, 2025, 2026]:
            m = (L.qyr == y).to_numpy()
            if m.sum() < 5: line += f"{y} n<5  "; continue
            line += f"{y} n={m.sum()} hit {(ret[m]>0).mean():.2f} gross {ret[m].mean():+.1f}bp | "
        print(line)
    print("\n분기별 thr70 gross (bp, n>=8 만):")
    qtrs = sorted(R.quarter.unique())
    for h in H:
        ok = (R[f'{h}_n'] >= MIN_VOTES) & ~R[f'{h}_frq'].isna() & (R[f'{h}_frq'] != 0)
        s = R[ok]
        lean = (s[f'{h}_fup'] >= .7) | (s[f'{h}_fup'] <= .3)
        L = s[lean]
        sgn = np.where(L[f'{h}_fup'] >= .5, 1, -1)
        ret = pd.Series(sgn * L[f'{h}_frq'].to_numpy() * 1e4, index=L.index)
        row = []
        for qt in qtrs:
            m = L.quarter == qt
            row.append(f"{ret[m].mean():+6.1f}({m.sum():3d})" if m.sum() >= 8 else "    -    ")
        print(f"{h:3s}: " + ' '.join(row))
    print("     " + '  '.join(f"{q:>8s}" for q in qtrs))

if __name__ == '__main__':
    if os.environ.get('ANALYZE_ONLY'):
        R = pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
        analyze(R)
    else:
        main()
