#!/usr/bin/env python3
"""[I] 10단계 — 시간대 선택적 보완 (합의 X, 각자 강점/조건부 라우팅).
핵심: A 틀리는 상황 ≠ B 틀리는 상황 인가 (오답 상관). horizon 5개."""
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
FEE = 11.0
TRAIN_Q = ['2024Q1', '2024Q2', '2024Q3', '2024Q4', '2025Q1', '2025Q2']
H5 = ['30m', '45m', '1h', '2h', '4h']

v2 = pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
hf = pd.read_parquet(f'{OUT}/lean70_v2_per_query_hfine.parquet')
R = v2.merge(hf[['q'] + [c for c in hf.columns if c.split('_')[0] in ('45m', '2h')]], on='q', how='inner')
nrm = pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
R['mod'] = nrm['min_of_day'].iloc[R.q].to_numpy()
R['z_atr'] = nrm['z_atr_14'].iloc[R.q].to_numpy()
R['slope120'] = nrm['z_ma_slope_120'].iloc[R.q].to_numpy()
R['is_tr'] = R.quarter.isin(TRAIN_Q)
nd = R.qday.nunique()
print(f"[load] {len(R)} 쿼리 ({nd}일)")

def lean(df, h, thr=0.70):
    """thr0.70 lean: dir (+1/-1/0), net(4h-self hold), correct(미래맞음)."""
    f = df[f'{h}_fup'].to_numpy(); n = df[f'{h}_n'].to_numpy()
    st = np.where(n >= 70, np.maximum(f, 1 - f), np.nan)
    d = np.where(f >= .5, 1, -1)
    fire = (st >= thr) & ~np.isnan(st)
    dirn = np.where(fire, d, 0)
    frq = df[f'{h}_frq'].to_numpy()
    net = dirn * frq * 1e4 - FEE
    correct = (dirn * frq > 0)
    return dirn, net, correct, fire

def ci(qday, net):
    if len(net) < 5: return np.nan, np.nan, np.nan
    dm = pd.Series(net).groupby(qday).mean().to_numpy()
    bs = np.random.default_rng(7).choice(dm, (5000, len(dm)), replace=True).mean(axis=1)
    return dm.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

# ===== 작업1: 시간대/상황별 조건부 강점 (hit) =====
print("\n===== 작업1: horizon × 상황 hit (thr0.70 lean, 발화 시) =====")
sits = [('early(mod<=1199)', R['mod'] <= 1199), ('late(mod>1199)', R['mod'] > 1199),
        ('hivol(atr>0)', R.z_atr > 0), ('lovol(atr<=0)', R.z_atr <= 0),
        ('trend(|slope120|>0.5)', R.slope120.abs() > 0.5), ('range(|slope120|<=0.5)', R.slope120.abs() <= 0.5)]
print(f"{'horizon':8s} " + " ".join(f"{nm.split('(')[0]:>8s}" for nm, _ in sits) + "   전체")
for h in H5:
    dirn, net, corr, fire = lean(R, h)
    row = f"{h:8s} "
    for nm, sm in sits:
        m = fire & sm.to_numpy()
        row += f"{corr[m].mean():8.2f} " if m.sum() >= 20 else f"{'n<20':>8s} "
    row += f"  {corr[fire].mean():.2f}({fire.sum()})"
    print(row)

# ===== 작업2: 오답 상관 (A 틀릴 때 B 맞나) =====
print("\n===== 작업2: 오답 상관 — 두 horizon 다 발화 시 분할표 + 방향 오답 corr =====")
D = {h: lean(R, h) for h in H5}
for a, b in [('30m', '4h'), ('30m', '1h'), ('1h', '4h'), ('45m', '2h')]:
    fa, fb = D[a][3], D[b][3]
    both = fa & fb
    if both.sum() < 10: print(f"  {a}&{b}: 동시발화 n={both.sum()}"); continue
    ca, cb = D[a][2], D[b][2]
    bb = both
    n11 = (ca & cb & bb).sum(); n10 = (ca & ~cb & bb).sum()
    n01 = (~ca & cb & bb).sum(); n00 = (~ca & ~cb & bb).sum()
    # 보완 여지 = A틀릴때 B맞는 비율
    a_wrong = (~ca & bb); compl = cb[a_wrong].mean() if a_wrong.sum() else np.nan
    print(f"  {a}&{b} (동시 {bb.sum()}): 둘맞 {n11} A만 {n10} B만 {n01} 둘틀 {n00} | "
          f"A틀릴때 B맞을 확률 {compl:.2f} (>0.5=보완)")

# 전체 영역 방향 오답 상관 (발화 무관, 전 쿼리 방향 sign(fup-.5) vs 실제)
print("\n  전체영역 방향예측 오답 상관 (모든 쿼리, 발화 무관):")
err = {}
for h in H5:
    f = R[f'{h}_fup'].to_numpy(); frq = R[f'{h}_frq'].to_numpy()
    d = np.where(f >= .5, 1, -1)
    e = (d * frq < 0).astype(float)  # 1=틀림
    e[np.isnan(frq) | (R[f'{h}_n'].to_numpy() < 70)] = np.nan
    err[h] = e
for a, b in [('30m', '4h'), ('1h', '4h')]:
    m = ~np.isnan(err[a]) & ~np.isnan(err[b])
    print(f"    {a}-{b} 오답 corr: {np.corrcoef(err[a][m], err[b][m])[0,1]:.3f} (낮을수록 보완 여지)")

# ===== 작업3+4: 보완 룰 (쿼리 단위 dedupe — 동시발화=1포지션, 자본전액 함정 회피) =====
print("\n===== 작업3+4: 보완 룰 vs 단일 4h (4h 실현 hold) =====")
print("(신호 89~113x 동시발화=같은 노출 → 쿼리 단위 1포지션. 자본 1/k 분산도 병기)")
# 각 horizon 발화 마스크/방향
FIRE = {h: lean(R, h) for h in H5}
frq4 = R['4h_frq'].to_numpy()
valid4 = ~np.isnan(frq4) & (frq4 != 0)
qday = R.qday.to_numpy(); quarter = R.quarter.to_numpy(); te = ~R.is_tr.to_numpy()
any_fire = np.zeros(len(R), bool); dirsum = np.zeros(len(R))
for h in H5:
    dirn, net, corr, fire = FIRE[h]
    any_fire |= fire; dirsum += dirn
cdir = np.sign(dirsum)
# R-union-dedup: 쿼리당 1포지션 (합의방향), 4h hold 실현
def report(name, mask):
    m = mask & valid4
    if m.sum() < 5: print(f"  {name}: n={m.sum()}"); return
    net = cdir[m] * frq4[m] * 1e4 - FEE
    dm, lo, hi = ci(qday[m], net)
    mte = m & te
    nette = cdir[mte] * frq4[mte] * 1e4 - FEE
    nd_te = R[te].qday.nunique()
    print(f"  {name}: n={m.sum()} hit {(net>0).mean():.3f} per-trade {net.mean():+.1f} day{dm:+.1f}[{lo:+.0f},{hi:+.0f}] "
          f"| 일수익 full {net.sum()/nd:+.2f} test {nette.sum()/nd_te:+.2f}")
report("R-union(쿼리dedupe 합집합)", any_fire)
report("  단일 4h", FIRE['4h'][3])
report("  단일 30m", FIRE['30m'][3])
# 자본 1/평균동시발화 분산 보정
avg_conc = np.array([sum(FIRE[h][3][i] for h in H5) for i in np.where(any_fire & valid4)[0]]).mean()
netu = cdir[any_fire & valid4] * frq4[any_fire & valid4] * 1e4 - FEE
print(f"  → R-union 동시발화 평균 {avg_conc:.1f}개 → 자본 1/{avg_conc:.1f} 분산 시 일수익 ≈ {netu.sum()/nd/avg_conc:+.2f}bp/day")


# R-route: 시간대(early/late) × train 최고 hit horizon
print("\n  R-route (시간대별 train 최고 hit horizon → OOS):")
for sit_nm, sit in [('early', R['mod'] <= 1199), ('late', R['mod'] > 1199)]:
    # train 최고 hit
    best_h, best_hit = None, -1
    for h in H5:
        dirn, net, corr, fire = lean(R, h)
        m = fire & sit.to_numpy() & R.is_tr.to_numpy()
        if m.sum() >= 20 and corr[m].mean() > best_hit:
            best_hit = corr[m].mean(); best_h = h
    # OOS
    dirn, net, corr, fire = lean(R, best_h)
    te = fire & sit.to_numpy() & ~R.is_tr.to_numpy()
    qd = R.qday.to_numpy()[te]
    print(f"    {sit_nm}: train 최고 = {best_h} (hit {best_hit:.2f}) → test n={te.sum()} hit {corr[te].mean():.3f} net {net[te].mean():+.1f} 일수익 {net[te].sum()/R[~R.is_tr].qday.nunique():+.2f}")

print(f"\n현행 단일4h thr0.70: full 일수익 +8.40bp/day (4h만), 결합 15.1. test 8.1")
