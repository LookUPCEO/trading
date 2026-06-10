#!/usr/bin/env python3
"""[I] 8단계 — 경로 모양 유사도: 과거길이 L × 미래 horizon H 공간.
모양 요약 5특징 (raw 점 X), 정규화(크기 제거), whitening per-L 2023 fit. 전부 t≤0.
출력: (L,H) 84 히트맵 train/test, 순간 21차원 대비, 겹침률."""
import os, json
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB = '/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
LS = [3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240]      # 과거길이(분)
HS = [5, 10, 15, 30, 60, 120, 240]                           # 미래 horizon(분)
K_CAND = 1000; N_IND = 100; EXCL_DAYS = 3; MIN_VOTES = 70; FEE = 11.0; THR = 0.70
TRAIN_Q = ['2024Q1', '2024Q2', '2024Q3', '2024Q4', '2025Q1', '2025Q2']
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
    lab = pd.read_parquet(LAB, columns=['yr', 'day', 'min_of_day', 'mid'])
    days = sorted(lab['day'].unique()); day_ix = {d: i for i, d in enumerate(days)}
    ND = len(days)
    mids = np.full((ND, 1440), np.nan, np.float32)
    di = lab['day'].map(day_ix).to_numpy(); mo = lab['min_of_day'].to_numpy()
    mids[di, mo] = lab['mid'].to_numpy(np.float32)
    yr_of_day = np.array([int(d[:4]) for d in days])
    qtr_of_day = np.array([f"{d[:4]}Q{(int(d[5:7])-1)//3+1}" for d in days])
    logm = np.log(mids)
    print(f"[load] {ND}일 grid", flush=True)

    # 미래 수익 (모든 minute, H별) — grid 에서
    FR = {}
    for h in HS:
        fr = np.full((ND, 1440), np.nan, np.float32)
        fr[:, :1440 - h] = mids[:, h:] / mids[:, :1440 - h] - 1
        FR[h] = fr

    # ---- 경로 모양 5특징 × L (벡터화: sliding_window_view, t≤0, day 내) ----
    from numpy.lib.stride_tricks import sliding_window_view
    feats = {}   # L -> (ND,1440,5) — 모양 요약, raw 점 아님
    nmin = np.arange(1440)
    for L in LS:
        F = np.full((ND, 1440, 5), np.nan, np.float32)
        x = np.arange(L + 1, dtype=np.float64); xc = x - x.mean()
        denom1 = (xc * xc).sum()
        q = xc * xc; q = q - q.mean(); q = q - (q @ xc) / denom1 * xc   # xc 와 직교화
        denomq = (q * q).sum()
        third = max(L // 3, 1)
        # window: (ND, 1440-L, L+1)
        win = sliding_window_view(logm, L + 1, axis=1).astype(np.float64)
        nanw = np.isnan(win).any(-1)
        w0 = win[..., 0]; wL = win[..., -1]
        ret = wL - w0
        slope = np.einsum('dmk,k->dm', win, xc) / denom1
        curv = np.einsum('dmk,k->dm', win, q) / denomq
        dif = np.diff(win, axis=-1)
        vol = dif.std(-1) + 1e-12
        rev = (np.diff(np.sign(dif), axis=-1) != 0).sum(-1)
        end_mom = win[..., -third:].mean(-1) - win[..., :third].mean(-1)
        sc = vol * np.sqrt(L)
        out = np.stack([ret / sc, slope * L / sc, curv * L * L / sc,
                        rev / L, end_mom / sc], axis=-1).astype(np.float32)
        out[nanw] = np.nan
        F[:, L:, :] = out      # minute m 의 특징 = [m-L..m] (m>=L, day 내)
        feats[L] = F
        del win, out; print(f"  shape L={L} done", flush=True)

    # ---- 쿼리: 2024+, 30분 격자 (탐색용), 최장 L+burn 만족 ----
    qmask = (np.zeros((ND, 1440), bool))
    for d in range(ND):
        if yr_of_day[d] < 2024: continue
        ok = (nmin % 60 == 30) & (nmin >= 240)
        qmask[d, ok] = True
    qd, qm = np.where(qmask)
    print(f"[query] {len(qd)}", flush=True)

    # flatten index helper
    def flat(d, m): return d * 1440 + m
    flat_day = np.repeat(np.arange(ND), 1440)
    flat_min = np.tile(nmin, ND)

    results = {}   # (L,H) -> list of (qday, net, fup)
    overlap = []   # 순간 21차원 top100 vs 경로 top100 겹침 (대표 L=30)
    # 순간 21차원 X 로드 (whitened)
    nrm = pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    meta = json.load(open(f'{OUT}/reduce_norm_meta.json'))
    reps = [r for r in meta['reps'] if r != 'spread_bp']
    nyr = nrm['yr'].astype(int).to_numpy()
    Cmom = nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32)
    mu0 = Cmom[nyr==2023].mean(0); S0 = np.cov((Cmom[nyr==2023]-mu0).T)
    w0,V0 = np.linalg.eigh(S0); W0 = (V0@np.diag(1/np.sqrt(np.maximum(w0,1e-6)))@V0.T).astype(np.float32)
    Xmom = ((Cmom-mu0)@W0).astype(np.float32)
    mom_key = {(day_ix[d], m): i for i, (d, m) in enumerate(zip(nrm['day'], nrm['min_of_day']))}

    from time import time as _t
    t0 = _t()
    for L in LS:
        F = feats[L].reshape(ND * 1440, 5)
        valid = ~np.isnan(F).any(1)
        # whitening 2023 fit
        is23 = (flat_day < np.searchsorted(yr_of_day, 2024)) if False else (np.repeat(yr_of_day, 1440) == 2023)
        fit = valid & is23
        mu = F[fit].mean(0); S = np.cov((F[fit] - mu).T)
        w, V = np.linalg.eigh(S); W = (V @ np.diag(1/np.sqrt(np.maximum(w,1e-6))) @ V.T).astype(np.float32)
        X = np.where(valid[:, None], (F - mu) @ W, np.nan).astype(np.float32)
        # pool 인덱스 (valid only), day 정렬
        pool_idx = np.where(valid)[0][::2]   # 2분 stride 서브샘플 (속도; 독립일 매치 영향 미미)
        pday = flat_day[pool_idx]; pmin = flat_min[pool_idx]
        Xp = X[pool_idx]; xsqp = (Xp * Xp).sum(1)
        order_pool = np.argsort(pday, kind='stable')
        pday = pday[order_pool]; pmin = pmin[order_pool]; Xp = Xp[order_pool]; xsqp = xsqp[order_pool]
        day_start = np.searchsorted(pday, np.arange(ND))
        # 쿼리 루프
        qf = flat(qd, qm)
        qvalid = valid[qf]
        QI = np.where(qvalid)[0]
        Xq_all = X[qf[QI]]
        qd_v = qd[QI]; qm_v = qm[QI]
        BLK = 96
        for bi in range(0, len(QI), BLK):
            sl = slice(bi, bi + BLK)
            qdb = qd_v[sl]; qmb = qm_v[sl]; Xqb = Xq_all[sl]
            ends = day_start[np.maximum(qdb - EXCL_DAYS, 0)]
            emax = int(ends.max())
            if emax < 5000: continue
            D2 = xsqp[None, :emax] - 2.0 * (Xqb @ Xp[:emax].T)   # 배치 GEMM
            for j in range(len(qdb)):
                e = int(ends[j])
                if e < 5000: continue
                dd = D2[j, :e]
                kc = min(K_CAND, e - 1)
                cand = np.argpartition(dd, kc)[:kc]
                ordr = cand[np.argsort(dd[cand])]
                sel = greedy_h(pday[ordr], pmin[ordr], 240, N_IND)
                picks = ordr[sel]
                pdays = pday[picks]; pmins = pmin[picks]
                d = int(qdb[j]); m = int(qmb[j])
                for h in HS:
                    fr = FR[h][pdays, pmins]
                    v = fr[~np.isnan(fr)]; v = v[v != 0]
                    if len(v) < MIN_VOTES: continue
                    fup = (v > 0).mean()
                    if fup >= THR or fup <= 1 - THR:
                        sgn = 1. if fup >= .5 else -1.
                        frq = FR[h][d, m]
                        if np.isnan(frq) or frq == 0: continue
                        results.setdefault((L, h), []).append((d, sgn * frq * 1e4 - FEE))
        print(f"  L={L} kNN done ({_t()-t0:.0f}s)", flush=True)

    # ---- 집계: (L,H) 히트맵 train/test ----
    rows = []
    for (L, h), lst in results.items():
        df = pd.DataFrame(lst, columns=['dayidx', 'net'])
        df['qtr'] = qtr_of_day[df.dayidx.to_numpy()]
        tr = df[df.qtr.isin(TRAIN_Q)]; te = df[~df.qtr.isin(TRAIN_Q)]
        def dm(x): return x.groupby('dayidx')['net'].mean().mean() if len(x) >= 5 else np.nan
        rows.append(dict(L=L, H=h, n=len(df), n_tr=len(tr), n_te=len(te),
                         net_tr=dm(tr), net_te=dm(te), hit=float((df.net+FEE>0).mean())))
    H = pd.DataFrame(rows).sort_values(['L', 'H'])
    H.to_csv(f'{OUT}/pathshape_LH.csv', index=False)
    print("\n===== (L,H) train net day-mean 히트맵 =====")
    pv = H.pivot(index='L', columns='H', values='net_tr')
    print(pv.round(0).to_string())
    print("\n===== test net (train net>0 인 짝만, Bonferroni 분모 =", (H.net_tr.notna()).sum(), ") =====")
    cand = H[(H.net_tr > 0) & (H.n_tr >= 30)].sort_values('net_tr', ascending=False)
    for _, r in cand.head(15).iterrows():
        print(f"  L={r.L:3.0f} H={r.H:3.0f}: train {r.net_tr:+.1f}(n{r.n_tr:.0f}) → test {r.net_te:+.1f}(n{r.n_te:.0f}) hit{r.hit:.3f}")
    print(f"\n베이스라인 순간 21차원 4h thr0.70 = hit 0.68, net+90(full), 일수익 15bp")

if __name__ == '__main__':
    main()
