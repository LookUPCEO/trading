#!/usr/bin/env python3
"""
[I] 유사도 거래 — 2단계 후반부: 유사도 검색 검증 (시기 분포 + 진짜 닮음).

⚠️ 거래/예측 X. "닮은 과거를 찾는 동작" 자체만 검증.

조건 (동일 쿼리셋, 동일 pool):
  A raw47-gz   : 47 raw 라벨 + 전체표본 global z  (naive 기준선 — 1단계 경고 무시한 경우)
  B red21-gz   : 21 대표 raw + global z           (축약만)
  C red21-norm : 21 대표 + causal rolling 정규화   (최종 후보)
  C-cos / C-wh : C 의 cosine / whitened(2023 데이터로만 fit) — 거리척도 비교

검증:
  작업3 시기분포 — top-N 매치의 (a) |Δ달력일| 중앙값 / pool 기대값 (recency ratio,
        <1 = 시기 쏠림), (b) 쿼리연도 lift = P(match=qyr)/P(pool=qyr), (c) 연도 히스토그램.
  작업4 진짜 닮음 — 매치의 "과거 90분 가격경로" corr (쿼리 vs 매치, bp 기준).
        기준선: random pool 점들의 경로 corr. top-N 이 random 보다 높아야 진짜 닮음.
        + rank1/10/100 거리, pool 크기 (N 충분성).
lookahead 없음: pool = 쿼리 day 보다 과거 day 만 (sampled days 간격 ≥6 달력일 →
  인접분 leak 구조적 불가). 경로 비교도 과거 90분만 (미래 안 봄).
"""
import os, json
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB = '/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
NRM = f'{OUT}/labels_norm_reduced.parquet'
META = ['yr', 'day', 'sec', 'min_of_day', 'mid']
N_TOP = 100
N_QUERY = 300
PATH_MIN = 90   # 과거 경로 길이(분)
rng = np.random.default_rng(11)

def main():
    raw = pd.read_parquet(LAB)
    nrm = pd.read_parquet(NRM)
    meta = json.load(open(f'{OUT}/reduce_norm_meta.json'))
    reps = [r for r in meta['reps'] if r != 'spread_bp']
    lab47 = [c for c in raw.columns if c not in META]

    # 공통 행: norm 쪽 (day,min) 기준 (norm 은 이미 NaN-free)
    key = ['day', 'min_of_day']
    nrm = nrm.sort_values(key).reset_index(drop=True)
    raw = raw.set_index(key).loc[pd.MultiIndex.from_frame(nrm[key])].reset_index()
    assert len(raw) == len(nrm)
    n = len(nrm)
    days = sorted(nrm['day'].unique())
    day_ix = {d: i for i, d in enumerate(days)}
    drow = nrm['day'].map(day_ix).to_numpy()
    yr = nrm['yr'].astype(int).to_numpy()
    dayord = pd.to_datetime(nrm['day']).map(pd.Timestamp.toordinal).to_numpy()
    print(f"[load] common rows={n}, days={len(days)}")

    # ---- 피처 행렬 ----
    def gz(M):
        return (M - np.nanmean(M, 0)) / (np.nanstd(M, 0) + 1e-12)
    A = gz(raw[lab47].to_numpy(np.float32))
    B = gz(raw[reps].to_numpy(np.float32))
    C = nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32)
    # whitening: 2023 데이터로만 fit (forward 적용 — 검증용 척도비교)
    m23 = yr == 2023
    mu = C[m23].mean(0); S = np.cov((C[m23] - mu).T)
    w, V = np.linalg.eigh(S)
    W = V @ np.diag(1/np.sqrt(np.maximum(w, 1e-6))) @ V.T
    Cw = ((C - mu) @ W).astype(np.float32)

    # ---- 가격 경로 grid (day × minute) ----
    mids = np.full((len(days), 1440), np.nan, np.float32)
    mids[drow, nrm['min_of_day'].to_numpy()] = nrm['mid'].to_numpy(np.float32)

    def past_path(i):
        d, m = drow[i], int(nrm['min_of_day'].iloc[i])
        if m - PATH_MIN < 0: return None
        w_ = mids[d, m-PATH_MIN:m+1]
        if np.isnan(w_).any(): return None
        return (w_ / w_[-1] - 1) * 1e4  # bp, 현재=0

    # ---- 쿼리 선택: 2024+ (pool 충분), 경로 가능, 연도 stratified ----
    elig = (nrm['min_of_day'] >= 240 + PATH_MIN) & (yr >= 2024) & (drow >= 60)
    elig = np.where(elig)[0]
    qs = []
    for y in [2024, 2025, 2026]:
        cand = elig[yr[elig] == y]
        take = min(120 if y < 2026 else 60, len(cand))
        qs += list(rng.choice(cand, take, replace=False))
    qs = np.array(sorted(qs)); print(f"[query] {len(qs)} queries, 연도 {np.unique(yr[qs], return_counts=True)}")

    conds = {'A_raw47': (A, 'euc'), 'B_red21': (B, 'euc'), 'C_norm21': (C, 'euc'),
             'C_cos': (C, 'cos'), 'C_wh': (Cw, 'euc')}
    recs, hist_rows = [], []
    ex_store = {}
    Cn = {k: (M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12) if mt == 'cos' else M)
          for k, (M, mt) in conds.items()}

    for qi, q in enumerate(qs):
        pool = np.where(drow < drow[q])[0]
        qpath = past_path(q)
        # random 기준선 (경로 corr)
        rnd = rng.choice(pool, min(200, len(pool)), replace=False)
        rnd_corr = []
        if qpath is not None:
            for j in rnd[:100]:
                pp = past_path(j)
                if pp is not None:
                    rnd_corr.append(np.corrcoef(qpath, pp)[0, 1])
        pool_dd = np.abs(dayord[pool] - dayord[q])
        for cname, (M, mt) in conds.items():
            Mx = Cn[cname]
            d2 = ((Mx[pool] - Mx[q])**2).sum(1)
            top = pool[np.argsort(d2)[:N_TOP]]
            dsort = np.sort(d2)
            # 시기 분포
            dd = np.abs(dayord[top] - dayord[q])
            qy = yr[q]
            lift = ((yr[top] == qy).mean() / max((yr[pool] == qy).mean(), 1e-9)
                    if (yr[pool] == qy).any() else np.nan)
            # 진짜 닮음: 과거 경로 corr
            tc = []
            if qpath is not None:
                for j in top[:50]:
                    pp = past_path(j)
                    if pp is not None:
                        tc.append(np.corrcoef(qpath, pp)[0, 1])
            recs.append(dict(cond=cname, q=int(q), qyr=int(qy), pool=len(pool),
                             dd_med_top=float(np.median(dd)), dd_med_pool=float(np.median(pool_dd)),
                             recency_ratio=float(np.median(dd)/max(np.median(pool_dd),1e-9)),
                             same_yr_lift=float(lift),
                             d_rank1=float(np.sqrt(dsort[0])), d_rank10=float(np.sqrt(dsort[9])),
                             d_rank100=float(np.sqrt(dsort[N_TOP-1])),
                             d_med_pool=float(np.sqrt(np.median(d2))),
                             path_corr_top=float(np.nanmean(tc)) if tc else np.nan,
                             path_corr_rnd=float(np.nanmean(rnd_corr)) if rnd_corr else np.nan))
            for y2 in np.unique(yr):
                hist_rows.append(dict(cond=cname, qyr=int(qy), myr=int(y2),
                                      frac_top=float((yr[top] == y2).mean()),
                                      frac_pool=float((yr[pool] == y2).mean())))
        if qi % 60 == 0:
            print(f"  q {qi}/{len(qs)}")
        if qi in (len(qs)//3, len(qs)//2) and qpath is not None:
            # 예시 저장 (조건 C 매치 차트용)
            Mx = Cn['C_norm21']
            d2 = ((Mx[pool] - Mx[q])**2).sum(1)
            ex_store[q] = pool[np.argsort(d2)[:5]]

    R = pd.DataFrame(recs); R.to_csv(f'{OUT}/simsearch_per_query.csv', index=False)
    H = pd.DataFrame(hist_rows)

    print("\n===== 작업3: 시기 분포 (중앙값 [p10,p90]) =====")
    for cname in conds:
        s = R[R.cond == cname]
        rr = s.recency_ratio; lf = s.same_yr_lift.dropna()
        print(f"{cname:9s} recency_ratio med {rr.median():.2f} [{rr.quantile(.1):.2f},{rr.quantile(.9):.2f}] | "
              f"same-yr lift med {lf.median():.2f} [{lf.quantile(.1):.2f},{lf.quantile(.9):.2f}]")

    print("\n===== 작업4: 진짜 닮음 (과거 90분 경로 corr, top50 vs random) =====")
    for cname in conds:
        s = R[R.cond == cname]
        t, r0 = s.path_corr_top.dropna(), s.path_corr_rnd.dropna()
        print(f"{cname:9s} top corr med {t.median():.3f} [{t.quantile(.1):.3f},{t.quantile(.9):.3f}] | "
              f"random med {r0.median():.3f} | 쿼리별 top>rnd 비율 {(s.path_corr_top > s.path_corr_rnd).mean():.2f}")

    print("\n===== N 충분성 (조건 C) =====")
    s = R[R.cond == 'C_norm21']
    print(f"pool size med {s['pool'].median():.0f} | d rank1/10/100 med "
          f"{s.d_rank1.median():.2f}/{s.d_rank10.median():.2f}/{s.d_rank100.median():.2f} "
          f"| pool 중앙거리 {s.d_med_pool.median():.2f} (rank100 << pool med 이면 충분)")

    # ---- 시각화 ----
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 1) 매치 연도 분포 (조건별, 쿼리연도별) vs pool
    fig, axes = plt.subplots(3, 5, figsize=(22, 10), sharey=True)
    for j, cname in enumerate(conds):
        for i, qy in enumerate([2024, 2025, 2026]):
            ax = axes[i, j]
            h = H[(H.cond == cname) & (H.qyr == qy)].groupby('myr')[['frac_top', 'frac_pool']].mean()
            x = np.arange(len(h))
            ax.bar(x-0.2, h.frac_top, 0.4, label='top100')
            ax.bar(x+0.2, h.frac_pool, 0.4, label='pool', alpha=0.6)
            ax.set_xticks(x); ax.set_xticklabels(h.index, fontsize=8)
            ax.set_title(f"{cname} | query {qy}", fontsize=9)
            if i == 0 and j == 0: ax.legend(fontsize=8)
    plt.suptitle("top-100 매치 연도 분포 vs pool 구성 (막대 같으면 시기 중립)", fontsize=12)
    plt.tight_layout(); plt.savefig(f'{OUT}/match_year_dist.png', dpi=110); plt.close()

    # 2) recency ratio / path corr 분포
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    names = list(conds)
    axes[0].boxplot([R[R.cond == c].recency_ratio.dropna() for c in names],
                    tick_labels=names, showfliers=False)
    axes[0].axhline(1, color='r', ls='--', lw=1); axes[0].set_title('recency ratio (1=시기중립, <1=쏠림)')
    axes[1].boxplot([R[R.cond == c].path_corr_top.dropna() for c in names] +
                    [R[R.cond == 'C_norm21'].path_corr_rnd.dropna()],
                    tick_labels=names + ['random'], showfliers=False)
    axes[1].set_title('과거 90분 경로 corr (top50 평균)')
    for ax in axes: ax.tick_params(labelsize=8)
    plt.tight_layout(); plt.savefig(f'{OUT}/recency_pathcorr_box.png', dpi=110); plt.close()

    # 3) 예시: 쿼리 vs top5 매치 과거 경로 차트 (조건 C)
    if ex_store:
        fig, axes = plt.subplots(1, len(ex_store), figsize=(8*len(ex_store), 4.5))
        axes = np.atleast_1d(axes)
        for ax, (q, tops) in zip(axes, ex_store.items()):
            qp = past_path(q)
            ax.plot(np.arange(-PATH_MIN, 1), qp, 'k-', lw=2.2,
                    label=f"query {nrm['day'].iloc[q]} {int(nrm['min_of_day'].iloc[q])//60:02d}:{int(nrm['min_of_day'].iloc[q])%60:02d}")
            for j in tops:
                pp = past_path(j)
                if pp is not None:
                    ax.plot(np.arange(-PATH_MIN, 1), pp, lw=1,
                            label=f"{nrm['day'].iloc[j]} (r={np.corrcoef(qp,pp)[0,1]:.2f})")
            ax.legend(fontsize=7); ax.set_xlabel('분 (0=시점)'); ax.set_ylabel('bp')
            ax.set_title('과거 90분 경로: 쿼리(굵은선) vs top5 매치 (C_norm21)', fontsize=9)
        plt.tight_layout(); plt.savefig(f'{OUT}/example_match_paths.png', dpi=110); plt.close()

    print(f"\n[save] {OUT}/simsearch_per_query.csv, match_year_dist.png, "
          f"recency_pathcorr_box.png, example_match_paths.png")

if __name__ == '__main__':
    main()
