#!/usr/bin/env python3
"""
[I] 유사도 거래 — 2단계 전반부: 정규화(시기 드리프트 제거) + 중복 축약.

⚠️ 거래/예측 X. stage1 labels.parquet (203일 243k분행 47라벨) 입력.

정규화 (작업2, 전부 causal):
  - rolling 통계 = "과거 day 들의 per-day robust 통계"만 사용. 현재 day 제외
    → 구조적으로 lookahead 불가능 (t 시점 정규화에 t 이후 정보 0).
  - 부호(방향) 라벨: scale-only  x / scale_past   (방향·0점 보존)
  - 크기 라벨(rv/atr/width/spread/range/rv_ratio): full robust-z  (x - med_past)/scale_past
    (vol-regime "수준" 자체가 드리프트이므로 location 제거가 목적)
  - scale = median(과거 D일 per-day IQR)/1.349, med = median(과거 D일 per-day median)
  - D=15 sampled days (STEP=6 → 달력 ~90일), min 5일. 첫 5 sampled days 는 NaN→제외.

축약 (작업1):
  - 정규화 후 spearman |r| 계층군집(average linkage), threshold 로 잘라 군집별
    medoid(군집 내 평균|r| 최대) 대표 선택.
  - 정보손실: 탈락 라벨마다 retained 셋으로 ridge 복원 R² + max|corr| 보고.
    R²<0.5 인 탈락 라벨은 다시 살림 (정보 버리지 않기).
  - 방향 정보: 부호 라벨은 scale-only 정규화라 sign 보존 자명 + 부호군별 대표 존재 확인.
  - PCA EVR 은 참조(차원수 sanity)로만 — 부호 섞임 때문에 검색공간은 대표선택.
"""
import os, json
import numpy as np
import pandas as pd

OUT = os.environ.get('SIM_OUT', '/Users/mark/Desktop/Mark/mark19/research/i_similarity')
os.makedirs(OUT, exist_ok=True)
LAB = os.environ.get('SIM_LAB', '/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet')

D_WIN = int(os.environ.get('D_WIN', '15'))   # rolling 과거 day 수. 의도 = 달력 ~90일
D_MIN = int(os.environ.get('D_MIN', '5'))    # (STEP=6 표본: 15/5, STEP=1 전수: 90/15)
CLUST_TH = 0.7  # |spearman| > 0.7 이면 같은 군집

META = ['yr', 'day', 'sec', 'min_of_day', 'mid']

# 부호(방향) 라벨 — scale-only (location 보존). 나머지 라벨 분류는 아래서 자동.
SIGNED = [c for c in [
    'ma_dev_5','ma_dev_15','ma_dev_30','ma_dev_60','ma_dev_120','ma_dev_240',
    'ma_slope_5','ma_slope_15','ma_slope_30','ma_slope_60','ma_slope_120','ma_slope_240',
    'boll_pos','macd','macd_hist','di_diff',
    'obi1','obi5','obi20','obi50','obi_wtd','dobi5_30','dobi5_60',
    'flow_30','flow_1m','flow_5m','bigflow_5m','bigflow_norm',
    'body_ratio','vol_z',
]]
# 크기-드리프트 라벨 — full robust-z (stage1 에서 시기 드리프트 확인된 군)
MAG_DRIFT = ['rv_60','rv_300','rv_900','rv_1800','rv_3600','atr_14',
             'boll_width','range_bp','rv_ratio','spread_bp']
# bounded/stationary 0~100 류 — 중심 알려짐: 고정 affine 후 scale-only 와 동일 취급
# (rsi/stoch 는 50 중심, adx/wick 는 크기지만 stationary) → scale-only 그룹에 합류
CENTERED50 = ['rsi_14','rsi_30','stoch_k','stoch_d']
STAT_MAG = ['adx_14','upper_wick','lower_wick']

def main():
    df = pd.read_parquet(LAB)
    labels = [c for c in df.columns if c not in META]
    assert len(labels) == 47, len(labels)
    days = sorted(df['day'].unique())
    print(f"[load] {df.shape}, days={len(days)}")

    # rsi/stoch 50 중심화 (고정 상수 — 데이터 비의존, lookahead 아님)
    for c in CENTERED50:
        df[c] = df[c] - 50.0

    # ---- 작업2: causal rolling per-day robust 정규화 ----
    # per-day median / IQR
    g = df.groupby('day')[labels]
    day_med = g.median()
    day_iqr = g.quantile(0.75) - g.quantile(0.25)
    day_med = day_med.reindex(days); day_iqr = day_iqr.reindex(days)

    # 과거 D_WIN 일 (현재 day 제외: shift(1)) 의 median-of-medians / median-of-IQRs
    past_med = day_med.shift(1).rolling(D_WIN, min_periods=D_MIN).median()
    past_scale = (day_iqr.shift(1).rolling(D_WIN, min_periods=D_MIN).median()) / 1.349
    eps = 1e-12

    Z = pd.DataFrame(index=df.index)
    med_map = past_med.reindex(df['day'].values).to_numpy()
    scl_map = past_scale.reindex(df['day'].values).to_numpy()
    X = df[labels].to_numpy(float)
    li = {c: i for i, c in enumerate(labels)}
    Zv = np.full_like(X, np.nan)
    for c in labels:
        i = li[c]
        s = scl_map[:, i]
        if c in MAG_DRIFT:
            Zv[:, i] = (X[:, i] - med_map[:, i]) / (s + eps)
        else:  # SIGNED + CENTERED50 + STAT_MAG: scale-only (방향/0점 보존)
            Zv[:, i] = X[:, i] / (s + eps)
    Z = pd.DataFrame(Zv, columns=labels, index=df.index)

    # 방향 보존 sanity: scale-only 라벨은 sign 동일해야 (scale>0)
    sgn_ok = {}
    for c in SIGNED:
        a, b = np.sign(X[:, li[c]]), np.sign(Zv[:, li[c]])
        m = ~np.isnan(b)
        sgn_ok[c] = float((a[m] == b[m]).mean())
    bad_sign = {c: v for c, v in sgn_ok.items() if v < 0.9999}
    print(f"[sign] scale-only 부호보존 위반 라벨: {bad_sign if bad_sign else '없음 (전부 100%)'}")

    # 유효행: 정규화 가능 + ma_slope_240 NaN 제외
    valid = ~Z.isna().any(axis=1)
    print(f"[valid] {valid.sum()}/{len(df)} rows ({df.loc[valid,'day'].nunique()} days, "
          f"{df.loc[~valid,'day'].nunique()} days 부분/전체 제외)")

    # ---- 시기 안정성: 정규화 전/후 연도별 중앙값/IQR ----
    rows = []
    for c in labels:
        for yr, sub in df[valid].groupby('yr'):
            zc = Z.loc[sub.index, c]
            rows.append(dict(label=c, yr=int(yr),
                             raw_med=float(sub[c].median()), raw_iqr=float(sub[c].quantile(.75)-sub[c].quantile(.25)),
                             z_med=float(zc.median()), z_iqr=float(zc.quantile(.75)-zc.quantile(.25))))
    ts = pd.DataFrame(rows)
    ts.to_csv(f"{OUT}/norm_temporal_stats.csv", index=False)
    # 드리프트 지표: 연도 IQR 최대/최소 비
    drift = ts.groupby('label').apply(
        lambda t: pd.Series(dict(raw_iqr_ratio=t.raw_iqr.max()/max(t.raw_iqr.min(),1e-12),
                                 z_iqr_ratio=t.z_iqr.max()/max(t.z_iqr.min(),1e-12))), include_groups=False)
    drift = drift.sort_values('raw_iqr_ratio', ascending=False)
    drift.to_csv(f"{OUT}/norm_drift_ratio.csv")
    print("[drift] 연도 IQR ratio (raw→z), 상위 12:")
    print(drift.head(12).round(2).to_string())

    # ---- 작업1: 축약 (정규화 공간에서) ----
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    Zs = Z[valid].sample(min(60000, valid.sum()), random_state=7)
    C = Zs.corr('spearman').to_numpy()
    Cd = 1 - np.abs(C); np.fill_diagonal(Cd, 0)
    lk = linkage(squareform(Cd, checks=False), method='average')
    for th in [0.6, 0.7, 0.8]:
        ncl = len(set(fcluster(lk, 1-th, criterion='distance')))
        print(f"[cluster] |r|>{th} → {ncl} clusters")
    cl = fcluster(lk, 1-CLUST_TH, criterion='distance')
    groups = {}
    for c, k in zip(labels, cl):
        groups.setdefault(int(k), []).append(c)

    # medoid 대표 (군집 내 평균 |r| 최대)
    reps = []
    for k, mem in groups.items():
        if len(mem) == 1:
            reps.append(mem[0]); continue
        idx = [li2 for li2, c in enumerate(labels) if c in mem]
        sub = np.abs(C[np.ix_(idx, idx)])
        reps.append(mem[int(np.argmax(sub.mean(0)))])
    reps = sorted(reps, key=labels.index)

    # 정보손실: 탈락 라벨 ridge 복원 R² (정규화 공간, 충분표본)
    from sklearn.linear_model import Ridge
    dropped = [c for c in labels if c not in reps]
    Xr = Zs[reps].to_numpy()
    rec = []
    for c in dropped:
        y = Zs[c].to_numpy()
        r = Ridge(alpha=1.0).fit(Xr, y)
        r2 = r.score(Xr, y)
        mx = float(np.abs(C[labels.index(c), [labels.index(p) for p in reps]]).max())
        rec.append(dict(label=c, recon_r2=float(r2), max_abs_corr_to_reps=mx))
    rec = pd.DataFrame(rec).sort_values('recon_r2')
    # R² < 0.5 → 정보 버려짐 → 다시 살림
    revive = rec[rec.recon_r2 < 0.5].label.tolist()
    if revive:
        print(f"[revive] 복원 R²<0.5 → 대표에 추가: {revive}")
        reps = sorted(set(reps) | set(revive), key=labels.index)
        dropped = [c for c in labels if c not in reps]
        Xr = Zs[reps].to_numpy()
        rec = []
        for c in dropped:
            y = Zs[c].to_numpy()
            r2 = Ridge(alpha=1.0).fit(Xr, y).score(Xr, y)
            mx = float(np.abs(C[labels.index(c), [labels.index(p) for p in reps]]).max())
            rec.append(dict(label=c, recon_r2=float(r2), max_abs_corr_to_reps=mx))
        rec = pd.DataFrame(rec).sort_values('recon_r2')
    rec.to_csv(f"{OUT}/reduction_recon_r2.csv", index=False)
    print(f"[reduce] 47 → {len(reps)} 대표: {reps}")
    print(f"[recon] 탈락 {len(dropped)}개 복원 R²: min={rec.recon_r2.min():.3f} "
          f"med={rec.recon_r2.median():.3f}")
    print(rec.head(8).round(3).to_string(index=False))

    # PCA 참조 (유효 차원수 sanity)
    from sklearn.decomposition import PCA
    Zw = (Zs - Zs.mean()) / (Zs.std() + 1e-12)
    p = PCA().fit(Zw.to_numpy())
    cum = np.cumsum(p.explained_variance_ratio_)
    n90, n95 = int(np.searchsorted(cum, .90) + 1), int(np.searchsorted(cum, .95) + 1)
    print(f"[pca] EVR 90%→{n90}dim, 95%→{n95}dim (참조)")

    # 부호군 대표 존재 확인 (방향 정보)
    sgn_groups = dict(trend=['ma_dev','ma_slope','boll_pos','macd','di_diff','rsi','stoch'],
                      obi=['obi','dobi'], flow=['flow','bigflow'], candle=['body'])
    for gname, pref in sgn_groups.items():
        has = [r for r in reps if any(r.startswith(p) for p in pref)]
        print(f"[direction] {gname}: 대표 {has if has else '⚠️ 없음!'}")

    # spread_bp 제외: 1틱 양자화 → per-day IQR≈0 → scale 나눗셈 폭발 (z_iqr 20~45,
    # 거리 지배 + 드리프트 잔존 2.26). 정보량 최소(스프레드≈상수 1틱). 별도 게이트로 재방문 가능.
    if 'spread_bp' in reps:
        reps = [r for r in reps if r != 'spread_bp']
        print(f"[exclude] spread_bp 제외 (양자화/scale 폭발) → {len(reps)} dims")

    # 저장 (±10 winsorize — 단일 차원 폭주가 거리 지배하는 것 방지; 고정상수, causal)
    out = df[META].copy()
    for c in reps:
        out[f"z_{c}"] = Z[c].clip(-10, 10)
    out = out[valid].reset_index(drop=True)
    out.to_parquet(f"{OUT}/labels_norm_reduced.parquet")
    meta = dict(reps=reps, n_reps=len(reps), clust_th=CLUST_TH, d_win=D_WIN,
                signed=SIGNED, mag_drift=MAG_DRIFT, pca_n90=n90, pca_n95=n95,
                rows=int(valid.sum()))
    json.dump(meta, open(f"{OUT}/reduce_norm_meta.json", 'w'), indent=1, ensure_ascii=False)

    # 정규화 전/후 시기 박스플롯 (드리프트 라벨 위주)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = [c for c in ['rv_300','atr_14','boll_width','range_bp','ma_dev_60','macd',
                        'spread_bp','rsi_14','obi5','flow_5m'] ]
    fig, axes = plt.subplots(2, len(show), figsize=(3.2*len(show), 7), sharex=True)
    dv = df[valid]
    for j, c in enumerate(show):
        for i, (src, ttl) in enumerate([(dv[c], 'raw'), (Z[valid][c], 'norm')]):
            data = [src[dv.yr == y].dropna() for y in sorted(dv.yr.unique())]
            axes[i, j].boxplot(data, tick_labels=[str(y) for y in sorted(dv.yr.unique())],
                               showfliers=False)
            axes[i, j].set_title(f"{c} ({ttl})", fontsize=8)
            axes[i, j].tick_params(labelsize=7)
    plt.tight_layout(); plt.savefig(f"{OUT}/norm_before_after_boxplots.png", dpi=110)
    print(f"[save] {OUT}/labels_norm_reduced.parquet ({len(reps)} dims), boxplots, csvs")

if __name__ == '__main__':
    main()
