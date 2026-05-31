#!/usr/bin/env python3
"""
G 분기점 형제 탐색 — oracle gross 실현 경로 (lookahead 없이).

핵심 원칙 (CLAUDE.md):
  - side 결정 정보 시점 ≤ 진입 시점 (t=0). PnL 시작 = 진입 시점.
  - 30s 덫 회피: "좋아진 후" 진입 금지. 모든 방향 신호는 t<=0 정보.
  - 분포 + 시기분해 + fill 현실 (maker/taker 시나리오).

형제:
  1. 부분 capture / 스케일 — net 비율 (fee 고정비라 부분일수록 불리?)
  2. 실현가능 oracle (causal side rule, hedge+stop straddle)
  3. vol 게이트 × 진입전 방향신호 (OBI, momentum)
  4. 정보-진입 간극 천장 (oracle |.| vs best causal rule)

비-overlap 5m(300s) 윈도우. 진입 = 윈도우 시작 t=0, 청산 = t=300.
방향 규칙은 전부 t<=0 정보:
  R_mom5  = sign(mid[0]-mid[-300])
  R_rev5  = -R_mom5
  R_mom15 = sign(mid[0]-mid[-900])
  R_obi   = sign(bidsize0_4 - asksize0_4) at t=0
출력: 각 규칙 net 분포 + 시기(연/regime)별 + vol게이트 조건부.
"""
import os, sys, json
import numpy as np
import pandas as pd

RAW = '/Users/mark/mark19_data/ETHUSDT'
REG = '/Users/mark/mark19_data/regime_labels.parquet'
OUT = '/Users/mark/Desktop/Mark/mark19/research/g_sibling'
os.makedirs(OUT, exist_ok=True)

# ── 일자 샘플: 전 구간 고르게 (시기분해용) ──
all_days = sorted([d[:-8] for d in os.listdir(RAW) if d.endswith('.parquet')])
# 전 구간 균등 샘플 ~ every Nth
STEP = int(os.environ.get('STEP', '24'))   # 1198/24 ≈ 50일
sample_days = all_days[::STEP]
print(f'[setup] total days {len(all_days)}, sampling {len(sample_days)} (step {STEP})')
print(f'        range {sample_days[0]} -> {sample_days[-1]}')

_rdf = pd.read_parquet(REG)
_rdf['date'] = pd.to_datetime(_rdf['date']).dt.strftime('%Y-%m-%d')
reg = _rdf.set_index('date')['hmm_label'].to_dict()

# 필요 컬럼만 (top5 OB + l0)
LV = 5
cols = ['timestamp']
for i in range(LV):
    cols += [f'bid_{i}_price', f'bid_{i}_size', f'ask_{i}_price', f'ask_{i}_size']

WIN = 300       # 5m
HBACK1 = 300    # 5m 모멘텀
HBACK2 = 900    # 15m 모멘텀

rows = []
for di, day in enumerate(sample_days):
    fp = f'{RAW}/{day}.parquet'
    try:
        df = pd.read_parquet(fp, columns=cols)
    except Exception as e:
        print(f'  skip {day}: {e}'); continue
    # day filter (wrap-around 방지): 해당 day 만
    ts = pd.to_datetime(df['timestamp'], utc=True)
    df = df[ts.dt.date == pd.Timestamp(day).date()].copy()
    if len(df) < 1000:
        continue
    ts = pd.to_datetime(df['timestamp'], utc=True)
    # second offset from day start (pandas3: total_seconds, astype int64 unreliable)
    so_f = (ts - ts.iloc[0]).dt.total_seconds().values
    df['so'] = np.round(so_f).astype(int)
    mid = ((df['bid_0_price'] + df['ask_0_price']) / 2.0).values
    # top5 size imbalance at each snapshot
    bsz = df[[f'bid_{i}_size' for i in range(LV)]].sum(axis=1).values
    asz = df[[f'ask_{i}_size' for i in range(LV)]].sum(axis=1).values
    obi = (bsz - asz) / (bsz + asz + 1e-9)
    so = df['so'].values
    # build 1s grid: last snapshot per second, ffill
    maxs = int(so[-1])
    grid_mid = np.full(maxs + 1, np.nan)
    grid_obi = np.full(maxs + 1, np.nan)
    grid_mid[so] = mid    # later assignment wins = last per sec (so monotone)
    grid_obi[so] = obi
    # vectorized ffill (then bfill the leading gap)
    def ffill(a):
        mask = ~np.isnan(a)
        if not mask.any(): return a
        idx = np.where(mask, np.arange(len(a)), 0)
        np.maximum.accumulate(idx, out=idx)
        a = a[idx]
        first = np.argmax(mask)
        a[:first] = a[first]
        return a
    grid_mid = ffill(grid_mid)
    grid_obi = ffill(grid_obi)

    yr = day[:4]
    rg = reg.get(day, -1)
    # non-overlap 5m windows; start s requires history HBACK2 and future WIN
    for s in range(HBACK2, maxs - WIN, WIN):
        p_b2 = grid_mid[s - HBACK2]
        p_b1 = grid_mid[s - HBACK1]
        p0   = grid_mid[s]
        p1   = grid_mid[s + WIN]
        if not np.isfinite([p_b2, p_b1, p0, p1]).all() or p0 <= 0:
            continue
        ret_fwd = (p1 - p0) / p0 * 1e4   # bp, signed (long pov)
        oracle  = abs(ret_fwd)           # 사후 one-way 상한
        # 진입전 vol (직전 5m 1s 로그수익 std) → 게이트
        seg = grid_mid[s - HBACK1: s + 1]
        rr = np.diff(seg) / seg[:-1]
        vol_prior = np.std(rr) * 1e4 if len(rr) > 5 else np.nan
        # causal side rules (정보 시점 <= 0)
        mom5  = np.sign(p0 - p_b1)
        mom15 = np.sign(p0 - p_b2)
        obi0  = grid_obi[s]
        sobi  = np.sign(obi0)
        rows.append(dict(
            day=day, yr=yr, rg=rg, s=s,
            ret_fwd=ret_fwd, oracle=oracle, vol_prior=vol_prior, obi0=obi0,
            net_mom5  = mom5  * ret_fwd,
            net_rev5  = -mom5 * ret_fwd,
            net_mom15 = mom15 * ret_fwd,
            net_obi   = sobi  * ret_fwd,
        ))
    if di % 10 == 0:
        print(f'  [{di}/{len(sample_days)}] {day} rg={rg} cum_rows={len(rows)}')

R = pd.DataFrame(rows)
R.to_parquet(f'{OUT}/windows.parquet')
print(f'\n[done] {len(R)} windows -> {OUT}/windows.parquet')

# ── 요약: 형제 4 (간극 천장) ──
RULES = ['net_mom5', 'net_rev5', 'net_mom15', 'net_obi']
def summ(s):
    return dict(mean=s.mean(), p10=s.quantile(.1), p50=s.median(),
                p90=s.quantile(.9), winrate=(s > 0).mean())

print('\n=== oracle one-way gross (사후 상한) ===')
print(f"  mean {R.oracle.mean():.2f}bp  p50 {R.oracle.median():.2f}  p90 {R.oracle.quantile(.9):.2f}")
print('\n=== causal rule GROSS net (fee 차감 전, signed) ===')
for r in RULES:
    d = summ(R[r])
    print(f"  {r:10s} mean {d['mean']:+.3f}bp  win {d['winrate']:.3f}  p10 {d['p10']:+.2f} p50 {d['p50']:+.2f} p90 {d['p90']:+.2f}")

print('\n=== fee 차감 후 (시나리오) ===')
for fee, lab in [(4.0,'M+M'), (7.5,'M+T'), (11.0,'T+T')]:
    best = None
    for r in RULES:
        net = R[r].mean() - fee
        if best is None or net > best[1]: best = (r, net)
    print(f"  fee {fee:4.1f}bp ({lab}): best rule {best[0]} net {best[1]:+.3f}bp/trade")

print('\n=== 시기분해 (연도별 best causal gross) ===')
for yr, g in R.groupby('yr'):
    line = f'  {yr} (n={len(g)}): oracle {g.oracle.mean():5.2f} | '
    line += ' '.join(f'{r.split("_")[1]} {g[r].mean():+.2f}' for r in RULES)
    print(line)

print('\n=== regime별 (hmm) best causal gross ===')
for rg, g in R.groupby('rg'):
    line = f'  rg{rg} (n={len(g)}): oracle {g.oracle.mean():5.2f} | '
    line += ' '.join(f'{r.split("_")[1]} {g[r].mean():+.2f}' for r in RULES)
    print(line)

# ── 형제 3: vol 게이트 조건부 (high-vol = 크기 큼, 방향도 개선되나?) ──
print('\n=== 형제3: vol 게이트 조건부 (vol_prior 분위) ===')
R['volq'] = pd.qcut(R.vol_prior.rank(method='first'), 5, labels=[1,2,3,4,5])
for vq, g in R.groupby('volq', observed=True):
    line = f'  volq{vq} (n={len(g)}): oracle {g.oracle.mean():5.2f} | '
    line += ' '.join(f'{r.split("_")[1]} {g[r].mean():+.3f}' for r in RULES)
    print(line)

# ── 형제 3b: OBI 강도 게이트 ──
print('\n=== 형제3b: |OBI| 게이트 (강한 imbalance 일 때 방향 맞나) ===')
R['obiabs'] = R.obi0.abs()
R['obiq'] = pd.qcut(R.obiabs.rank(method='first'), 5, labels=[1,2,3,4,5])
for oq, g in R.groupby('obiq', observed=True):
    print(f'  obiq{oq} (n={len(g)}): net_obi {g.net_obi.mean():+.3f}bp  win {(g.net_obi>0).mean():.3f}')

# 저장
summary = {
    'n_windows': int(len(R)),
    'oracle_mean_bp': float(R.oracle.mean()),
    'causal_gross': {r: float(R[r].mean()) for r in RULES},
    'best_causal_gross_bp': float(max(R[r].mean() for r in RULES)),
    'fee_M+M_bp': 4.0,
    'gap_oracle_vs_bestcausal': float(R.oracle.mean() - max(R[r].mean() for r in RULES)),
}
with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\n[summary] {json.dumps(summary, indent=2)}')
