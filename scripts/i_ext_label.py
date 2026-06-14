#!/usr/bin/env python3
"""[I] 14단계 ① — 외부 데이터(funding) 라벨 추가 → 유사도 재계산 → hit/net.
funding 만 전체기간 보유 (liquidation/OI 미수집). funding rate + 30일 z (causal, 8h 설정값).
21차원 + funding 2차원(자체 whitening) → kNN → 4h thr0.70 hit/net vs baseline."""
import os, json
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB = '/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
FUND = '/Users/mark/mark19_data/funding_ETHUSDT.parquet'
K_CAND = 1000; N_IND = 100; EXCL_DAYS = 3; MIN_VOTES = 70; FEE = 11.0; THR = 0.70
TRAIN_Q = ['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']

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
    nrm = pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    meta = json.load(open(f'{OUT}/reduce_norm_meta.json'))
    reps = [r for r in meta['reps'] if r != 'spread_bp']
    yr = nrm['yr'].astype(int).to_numpy(); mod = nrm['min_of_day'].to_numpy()
    days = sorted(nrm['day'].unique()); dix = {d:i for i,d in enumerate(days)}
    drow = nrm['day'].map(dix).to_numpy(); n = len(nrm)
    starts = np.searchsorted(drow, np.arange(len(days)))

    # ---- funding 라벨: rate (ffill to minute) + 30일 rolling z (causal) ----
    fund = pd.read_parquet(FUND).sort_values('ts').reset_index(drop=True)
    fund['fz'] = (fund.funding_rate - fund.funding_rate.rolling(90, min_periods=10).mean()) / \
                 (fund.funding_rate.rolling(90, min_periods=10).std() + 1e-9)   # 90×8h=30일 z
    # 각 (day,minute) 에 가장 최근 funding (ts <= 그 분 UTC) — ffill, causal
    fund['d'] = fund.ts.dt.strftime('%Y-%m-%d')
    # day별 마지막 적용 rate/z: 분 단위는 8h 블록 내 상수 → day+block. 단순화: 그 분 직전 funding.
    f_ts = fund.ts.values.astype('datetime64[s]').astype(np.int64)
    f_rate = fund.funding_rate.to_numpy(); f_z = fund.fz.to_numpy()
    # 각 nrm 행의 UTC epoch
    base = pd.to_datetime(nrm['day']).values.astype('datetime64[s]').astype(np.int64) + mod * 60
    idx = np.searchsorted(f_ts, base, side='right') - 1   # 직전 funding (causal)
    idx = np.clip(idx, 0, len(f_ts)-1)
    rate_lab = f_rate[idx]; z_lab = np.nan_to_num(f_z[idx])

    # ---- 기존 21차원 whitened (baseline) ----
    C = nrm[[f'z_{c}' for c in reps]].to_numpy(np.float32)
    m23 = yr == 2023
    def whiten(M, fitmask):
        mu = M[fitmask].mean(0); S = np.cov((M[fitmask]-mu).T); S = np.atleast_2d(S)
        w,V = np.linalg.eigh(S); W = (V@np.diag(1/np.sqrt(np.maximum(w,1e-6)))@V.T).astype(np.float32)
        return ((M-mu)@W).astype(np.float32)
    X21 = whiten(C, m23)
    # funding 2차원: robust z (2023 통계 기준, 특이행렬 회피 위해 whiten 대신 독립 표준화)
    Fraw = np.column_stack([rate_lab, z_lab]).astype(np.float64)
    med = np.median(Fraw[m23], 0); iqr = (np.percentile(Fraw[m23],75,0)-np.percentile(Fraw[m23],25,0))/1.349
    F = ((Fraw - med) / (iqr + 1e-12)).clip(-10, 10).astype(np.float32)
    Xext = np.column_stack([X21, F]).astype(np.float32)   # 23차원
    print(f"[funding] rate_lab std {rate_lab.std():.6f}, z_lab std {z_lab.std():.3f}, F std {F.std(0)}")

    # ---- 미래 4h ----
    lab = pd.read_parquet(LAB, columns=['day','min_of_day','mid']); lab = lab[lab.day.isin(days)]
    mids = np.full((len(days),1440), np.nan, np.float32)
    mids[lab['day'].map(dix).to_numpy(), lab['min_of_day'].to_numpy()] = lab['mid'].to_numpy(np.float32)
    fr = np.full(n, np.nan, np.float32); ok = mod+240<=1439
    fr[ok] = mids[drow[ok], mod[ok]+240]/mids[drow[ok], mod[ok]]-1

    qm = (yr>=2024) & (mod%10==5)
    qs = np.where(qm)[0]
    qtr = np.char.add(yr.astype(str), np.char.add('Q', (((nrm['day'].str[5:7].astype(int)-1)//3+1)).astype(str).to_numpy()))
    print(f"[load] {n} rows, queries {len(qs)}, funding z 범위 [{np.nanmin(z_lab):.1f},{np.nanmax(z_lab):.1f}]", flush=True)

    def run(X, tag):
        xsq = (X*X).sum(1); recs=[]
        BLK=128
        from time import time as _t; t0=_t()
        for bi in range(0,len(qs),BLK):
            qb = qs[bi:bi+BLK]; ends = starts[np.maximum(drow[qb]-EXCL_DAYS,0)]; emax=int(ends.max())
            if emax<50000: continue
            d2 = xsq[None,:emax] - 2.0*(X[qb]@X[:emax].T)
            for j,q in enumerate(qb):
                e=int(ends[j])
                if e<50000: continue
                row=d2[j,:e]; kc=min(K_CAND,e-1)
                cand=np.argpartition(row,kc)[:kc]; order=cand[np.argsort(row[cand])]
                sel=greedy_h(drow[order],mod[order],240,N_IND); picks=order[sel]
                v=fr[picks]; v=v[~np.isnan(v)]; v=v[v!=0]
                if len(v)<MIN_VOTES: continue
                fup=(v>0).mean()
                if fup>=THR or fup<=1-THR:
                    frq=fr[q]
                    if np.isnan(frq) or frq==0: continue
                    recs.append((int(drow[q]), qtr[q], (1 if fup>=.5 else -1)*frq*1e4-FEE))
        R=pd.DataFrame(recs, columns=['qday','qtr','net'])
        R.to_parquet(f'{OUT}/ext_{tag}.parquet')
        te=~R.qtr.isin(TRAIN_Q)
        nd=len(set(drow[qs])); ndte=len(set(drow[qs][~np.isin(qtr[qs],TRAIN_Q)]))
        def dm(s):
            d=s.groupby('qday')['net'].mean(); return d.mean()
        print(f"[{tag}] n={len(R)} hit{(R.net+FEE>0).mean():.3f} per-trade {R.net.mean():+.1f} "
              f"day {dm(R):+.1f} 일수익 {R.net.sum()/nd:+.2f} | test n={te.sum()} 일수익 {R[te].net.sum()/ndte:+.2f} "
              f"({_t()-t0:.0f}s)", flush=True)
        return R

    print("\n===== ① 외부(funding) 라벨 추가 vs baseline (4h thr0.70) =====")
    Rb = run(X21, 'base21')
    Re = run(Xext, 'ext23')
    print(f"\n외부가 새 정보면 ext 가 base 초과해야. baseline 단일4h 일수익 ~8.4bp(0.084%/day).")

if __name__ == '__main__':
    main()
