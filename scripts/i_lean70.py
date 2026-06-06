#!/usr/bin/env python3
"""
[I] 3단계 — 70% 방향 쏠림 건수. ⚠️ 쏠림 건수까지만 (폭/fee/거래 X).

설계 (승부처 함정 3개 정면):
① 매치 클러스터링: top-K(1000) 후보 → 같은 날 1개(최근접)만 = 독립 표 N=100.
   + 직근 EXCL_DAYS(3일) exclusion zone. 전/후 유효N 분포 보고.
② OOS: pool = 쿼리 day 이전 prefix 만 (구조적 causal). 룰(horizon/threshold)은
   train(2024Q1~2025Q2)에서 보고 test(2025Q3~)에서 고정 검증.
③ 통계: null = 같은 제약(독립일 N, 유효 미래)의 random 매치 — 유사도가 무정보면
   real 쏠림율 == null 쏠림율. + binomial(causal prefix base rate).

미래방향: 매치 day 안의 5m/10m/30m/1h/4h 수익 부호 (day 경계 침범 = NaN, wrap 금지).
votes: 유효(비NaN, 0 제외) ≥ MIN_VOTES 일 때만 판정. frac_up ≥ thr 또는 ≤ 1-thr = 쏠림.
"""
import os, json
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB = '/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
HORIZONS = {'5m': 5, '10m': 10, '30m': 30, '1h': 60, '4h': 240}
K_CAND = 1000      # day-dedupe 전 후보
N_IND = 100        # 독립(일 단위) 매치 수
EXCL_DAYS = 3      # 직근 과거 제외 (비용 ≤1% — I.2+ 측정)
MIN_VOTES = 70
THRS = [0.60, 0.65, 0.70, 0.75]
Q_STRIDE = 10      # 쿼리 격자 (분)
rng = np.random.default_rng(11)

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
    print(f"[load] DB {n} rows, {len(days)} days")

    # whitening (2023 fit — 2024+ 쿼리에 causal)
    C = nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32)
    m23 = yr == 2023
    mu = C[m23].mean(0); S = np.cov((C[m23] - mu).T)
    w, V = np.linalg.eigh(S)
    W = (V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-6))) @ V.T).astype(np.float32)
    X = ((C - mu) @ W).astype(np.float32)
    xsq = (X * X).sum(1)

    # day prefix (pool = X[:start_of(qday - EXCL_DAYS)])
    starts = np.searchsorted(drow, np.arange(len(days)))   # 각 day 첫 row index

    # ---- 미래 수익 (라벨 시각 = 분 마지막 초 끝; 미래 = day 안에서만, wrap 금지) ----
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
    # causal prefix base rate (day 단위 누적): 각 day 까지의 up/valid 누적
    base_cum = {}
    for hname in HORIZONS:
        up = (FR[hname] > 0).astype(np.int64); va = (~np.isnan(FR[hname]) & (FR[hname] != 0)).astype(np.int64)
        up_day = np.bincount(drow, up * va, minlength=len(days))
        va_day = np.bincount(drow, va, minlength=len(days))
        base_cum[hname] = (np.cumsum(up_day), np.cumsum(va_day))

    # ---- 쿼리: 2024+ 매 Q_STRIDE 분 ----
    q_mask = (yr >= 2024) & (mod % Q_STRIDE == 5)
    qs = np.where(q_mask)[0]
    qmax = int(os.environ.get('QMAX', '0'))
    if qmax: qs = qs[rng.choice(len(qs), qmax, replace=False)]; qs.sort()  # smoke
    print(f"[query] {len(qs)} queries ({Q_STRIDE}분 격자, 2024+)")

    def votes(picks, hname):
        v = FR[hname][picks]
        v = v[~np.isnan(v)]; v = v[v != 0]
        return len(v), int((v > 0).sum())

    recs = []
    BLK = 128
    from time import time as _t
    t0 = _t()
    for bi in range(0, len(qs), BLK):
        qb = qs[bi:bi + BLK]
        # 같은 블록 내 쿼리는 day 가까움 — pool 은 블록 최소 day 기준 (보수: 더 작은 pool)
        # 정확성 위해 쿼리별 pool end 사용하되 GEMM 은 블록 최대 end 로 한 번, 이후 마스크.
        ends = starts[np.maximum(drow[qb] - EXCL_DAYS, 0)]
        emax = ends.max()
        if emax < 50000: continue
        Q = X[qb]
        d2 = xsq[None, :emax] - 2.0 * (Q @ X[:emax].T)   # +|q|² 생략 (순위 불변)
        for j, q in enumerate(qb):
            e = ends[j]
            if e < 50000: continue
            row = d2[j, :e]
            kc = min(K_CAND, e - 1)
            cand = np.argpartition(row, kc)[:kc]
            order = cand[np.argsort(row[cand])]
            dd = drow[order]
            _, ui = np.unique(dd, return_index=True)       # day 별 첫 출현(=최근접)
            ui.sort()
            picks = order[ui[:N_IND]]                       # 독립일 매치
            n_uni_raw100 = len(np.unique(dd[:100]))         # 클러스터링 전 진단
            # null: 같은 제약 random 매치 (독립일)
            rnd = rng.choice(e, min(3 * N_IND, e), replace=False)
            _, ri = np.unique(drow[rnd], return_index=True)
            ri.sort(); rpicks = rnd[ri[:N_IND]]
            rec = dict(q=int(q), qday=int(drow[q]), qyr=int(yr[q]),
                       quarter=f"{yr[q]}Q{(int(nrm['day'].iloc[q][5:7])-1)//3+1}",
                       pool_days=int(drow[q] - EXCL_DAYS),
                       n_ind=len(picks), n_uni_raw100=n_uni_raw100)
            for hname in HORIZONS:
                nv, nu = votes(picks, hname)
                rnv, rnu = votes(rpicks, hname)
                cu, cv = base_cum[hname]
                di = max(drow[q] - EXCL_DAYS - 1, 0)
                base = cu[di] / max(cv[di], 1)
                rec[f'{hname}_n'] = nv
                rec[f'{hname}_fup'] = nu / nv if nv else np.nan
                rec[f'{hname}_rnd_n'] = rnv
                rec[f'{hname}_rnd_fup'] = rnu / rnv if rnv else np.nan
                rec[f'{hname}_base'] = base
            recs.append(rec)
        if bi % (BLK * 50) == 0:
            print(f"  q {bi}/{len(qs)} elapsed={_t()-t0:.0f}s", flush=True)
    R = pd.DataFrame(recs)
    R.to_parquet(f'{OUT}/lean70_per_query.parquet')
    print(f"[done] {len(R)} queries, elapsed={_t()-t0:.0f}s")

    # ---- 분석 ----
    from scipy.stats import binom
    print("\n===== 작업1: 클러스터링 전/후 유효 N =====")
    print(f"naive top-100 의 고유 day 수: med {R.n_uni_raw100.median():.0f} "
          f"[p10 {R.n_uni_raw100.quantile(.1):.0f}, p90 {R.n_uni_raw100.quantile(.9):.0f}] / 100")
    print(f"독립일 매치 확보: med {R.n_ind.median():.0f} (목표 {N_IND})")

    print("\n===== 작업2+3: 쏠림 비율 (real vs null random 매치) =====")
    for hname in HORIZONS:
        ok = R[f'{hname}_n'] >= MIN_VOTES
        f = R.loc[ok, f'{hname}_fup']; fr_ = R.loc[ok & (R[f'{hname}_rnd_n'] >= MIN_VOTES), f'{hname}_rnd_fup']
        line = f"{hname:3s} (유효쿼리 {ok.sum()}): "
        for thr in THRS:
            lr = ((f >= thr) | (f <= 1 - thr)).mean()
            ln = ((fr_ >= thr) | (fr_ <= 1 - thr)).mean()
            line += f"thr{int(thr*100)} real {lr*100:.2f}%/null {ln*100:.2f}%  "
        print(line)

    print("\n===== 작업3b: binomial (causal base rate, thr=0.70 쏠림의 유의성) =====")
    for hname in HORIZONS:
        ok = (R[f'{hname}_n'] >= MIN_VOTES)
        sub = R[ok]
        f = sub[f'{hname}_fup']; nv = sub[f'{hname}_n']; base = sub[f'{hname}_base']
        lean = (f >= 0.7) | (f <= 0.3)
        if lean.sum() == 0:
            print(f"{hname:3s}: thr70 쏠림 0건"); continue
        ls = sub[lean]
        k = (ls[f'{hname}_fup'] * ls[f'{hname}_n']).round().astype(int)
        pv = np.where(ls[f'{hname}_fup'] >= .5,
                      binom.sf(k - 1, ls[f'{hname}_n'].astype(int), ls[f'{hname}_base']),
                      binom.cdf(k, ls[f'{hname}_n'].astype(int), ls[f'{hname}_base']))
        print(f"{hname:3s}: 쏠림 {lean.sum()}건 ({lean.mean()*100:.2f}%) | base med {base.median():.3f} | "
              f"binom p<0.01 비율 {(pv < .01).mean()*100:.0f}% | up쏠림 {int((ls[f'{hname}_fup']>=.7).sum())} vs down {int((ls[f'{hname}_fup']<=.3).sum())}")

    print("\n===== 작업4: 분기별 (thr=0.70) + 하루 건수 환산 =====")
    qtrs = sorted(R.quarter.unique())
    for hname in HORIZONS:
        ok = R[f'{hname}_n'] >= MIN_VOTES
        row = []
        for qt in qtrs:
            s = R[ok & (R.quarter == qt)]
            if len(s) < 200: row.append('   -  '); continue
            f = s[f'{hname}_fup']
            row.append(f"{((f>=.7)|(f<=.3)).mean()*100:5.2f}%")
        print(f"{hname:3s}: " + ' '.join(row))
    print("    " + ' '.join(f"{q:>6s}" for q in qtrs))
    # 하루 건수 (10분 격자 → 분 환산은 ×10 아님: 인접 분 동일사건. 격자 기준 건수 그대로 보고)
    for hname in HORIZONS:
        ok = R[f'{hname}_n'] >= MIN_VOTES
        s = R[ok]
        lean = (s[f'{hname}_fup'] >= .7) | (s[f'{hname}_fup'] <= .3)
        perday = lean.groupby(s.qday).sum()
        alldays = s.qday.nunique()
        print(f"{hname:3s}: 10분격자 쏠림 {lean.sum()}건 / {alldays}일 = 일평균 {lean.sum()/alldays:.2f}건 "
              f"(쏠림 있는 날 비율 {(perday>0).mean()*100:.0f}%)")

if __name__ == '__main__':
    main()
