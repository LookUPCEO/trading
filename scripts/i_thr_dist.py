#!/usr/bin/env python3
"""[I] 6-4 — thr 분포 정밀 (평균 X) + 부분집합 train→OOS. 사전등록 48셀."""
import json
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
H = ['30m', '1h', '4h']
BANDS = [(0.60, 0.62), (0.62, 0.64), (0.64, 0.66), (0.66, 0.68), (0.68, 0.70), (0.70, 1.01)]
LOW_BANDS = [(0.62, 0.64), (0.64, 0.66), (0.66, 0.68), (0.68, 0.70)]
TRAIN_Q = ['2024Q1', '2024Q2', '2024Q3', '2024Q4', '2025Q1', '2025Q2']
FEE = 11.0

R = pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
nrm = pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(
    ['day', 'min_of_day']).reset_index(drop=True)
R['mod'] = nrm['min_of_day'].iloc[R.q].to_numpy()
R['z_atr'] = nrm['z_atr_14'].iloc[R.q].to_numpy()
R['z_obi'] = nrm['z_obi_wtd'].iloc[R.q].to_numpy()
R['is_train'] = R.quarter.isin(TRAIN_Q)

# 이벤트 테이블 (horizon 별, fup 밴드 포함)
evs = []
for h in H:
    ok = (R[f'{h}_n'] >= 70) & ~R[f'{h}_frq'].isna() & (R[f'{h}_frq'] != 0)
    s = R[ok]
    strength = np.maximum(s[f'{h}_fup'], 1 - s[f'{h}_fup'])
    m = strength >= 0.60
    s = s[m]; strength = strength[m]
    sgn = np.where(s[f'{h}_fup'] >= .5, 1, -1)
    evs.append(pd.DataFrame(dict(
        h=h, qday=s.qday.to_numpy(), quarter=s.quarter.to_numpy(), train=s.is_train.to_numpy(),
        strength=strength.to_numpy(), dir=sgn,
        net=sgn * s[f'{h}_frq'].to_numpy() * 1e4 - FEE,
        mod=s['mod'].to_numpy(), z_atr=s['z_atr'].to_numpy(),
        obi_agree=(np.sign(s['z_obi'].to_numpy()) == sgn))))
E = pd.concat(evs, ignore_index=True)
E['band'] = pd.cut(E.strength, [b[0] for b in BANDS] + [1.01], right=False,
                   labels=[f"[{a:.2f},{b:.2f})" for a, b in BANDS])
print(f"[events] {len(E)} (전 밴드, 3 horizon)")

print("\n===== 작업1: 밴드별 net 분포 (평균 한 숫자 X) =====")
print("band         |     n  | win%(net>0) | p10    p25    med    p75    p90  | mean | fee넘는비율")
for b in E.band.cat.categories:
    s = E[E.band == b]
    if len(s) < 10: continue
    q = np.percentile(s.net, [10, 25, 50, 75, 90])
    print(f"{b:12s} | {len(s):6d} | {(s.net>0).mean()*100:5.1f}%      | "
          f"{q[0]:+6.1f} {q[1]:+6.1f} {q[2]:+6.1f} {q[3]:+6.1f} {q[4]:+6.1f} | {s.net.mean():+5.1f} | "
          f"{(s.net>0).mean()*100:.0f}%")
print("→ '0.66 평균 음수' 안의 이익거래 비율:",
      f"{(E[E.band=='[0.66,0.68)'].net>0).mean()*100:.1f}%")

print("\n===== 작업2+3: 부분집합 48셀 — train 선택 → OOS 판정 =====")
def daymean_ci(s, alphas=(2.5, 0.104)):   # 0.104% ≈ Bonferroni-48 양측
    dm = s.groupby('qday')['net'].mean().to_numpy()
    if len(dm) < 4: return np.nan, np.nan, np.nan, np.nan, np.nan
    bs = np.random.default_rng(7).choice(dm, (6000, len(dm)), replace=True).mean(axis=1)
    return (dm.mean(), np.percentile(bs, alphas[0]), np.percentile(bs, 100 - alphas[0]),
            np.percentile(bs, alphas[1]), np.percentile(bs, 100 - alphas[1]))

SPLITS = [('all', lambda s: np.ones(len(s), bool))]
SPLITS += [(f'h={h}', (lambda hh: (lambda s: (s.h == hh).to_numpy()))(h)) for h in H]
SPLITS += [('early', lambda s: (s['mod'] <= 1199).to_numpy()),
           ('late', lambda s: (s['mod'] > 1199).to_numpy()),
           ('up', lambda s: (s['dir'] > 0).to_numpy()),
           ('down', lambda s: (s['dir'] < 0).to_numpy()),
           ('hivol', lambda s: (s.z_atr > 0).to_numpy()),
           ('lovol', lambda s: (s.z_atr <= 0).to_numpy()),
           ('obi_agree', lambda s: s.obi_agree.to_numpy()),
           ('obi_contra', lambda s: (~s.obi_agree).to_numpy())]
n_tried = 0
cands = []
rows = []
for a, b in LOW_BANDS:
    bname = f"[{a:.2f},{b:.2f})"
    sb = E[E.band == bname]
    for sname, fn in SPLITS:
        n_tried += 1
        sub = sb[fn(sb)]
        tr_ = sub[sub.train]; te_ = sub[~sub.train]
        if len(tr_) < 50:
            continue
        tr_dm = tr_.groupby('qday')['net'].mean().mean()
        rows.append(dict(band=bname, split=sname, n_tr=len(tr_), tr_net=tr_dm, n_te=len(te_)))
        if tr_dm > 0:
            cands.append((bname, sname, sub, tr_, te_))
T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/thr_dist_cells.csv', index=False)
print(f"시도 {n_tried}셀 (사전등록 48), n_tr>=50 인 셀 {len(rows)}, train net>0 후보 {len(cands)}")
print("\ntrain 양수 후보 → OOS:")
print("band×split          | n_tr  tr_net | n_te  te_net | te 95% CI | te Bonf-48 CI")
surv95 = []
for bname, sname, sub, tr_, te_ in sorted(cands, key=lambda x: -x[3].groupby('qday')['net'].mean().mean()):
    trm = tr_.groupby('qday')['net'].mean().mean()
    if len(te_) < 10:
        print(f"{bname}×{sname:11s} | {len(tr_):4d} {trm:+7.1f} | te n<10 판정불가"); continue
    tem, lo95, hi95, loB, hiB = daymean_ci(te_)
    tag = '✓' if lo95 > 0 else '✗'
    tagB = '✓' if loB > 0 else '✗'
    print(f"{bname}×{sname:11s} | {len(tr_):4d} {trm:+7.1f} | {len(te_):4d} {tem:+7.1f} | "
          f"[{lo95:+6.1f},{hi95:+6.1f}]{tag} | [{loB:+6.1f},{hiB:+6.1f}]{tagB}")
    if lo95 > 0: surv95.append((bname, sname, sub, te_))

print("\n===== 작업4: 생존 셀 일수익 (현행 결합 thr0.70 = 15.1bp/day 대비) =====")
nd_all = R.qday.nunique(); nd_te = R[~R.is_train].qday.nunique()
if not surv95:
    print("OOS 95% 생존 부분집합 없음 — 평균에 가린 이익 구간은 train 환상이거나 미미.")
for bname, sname, sub, te_ in surv95:
    add_all = sub.net.sum() / nd_all
    add_te = te_.net.sum() / nd_te
    print(f"{bname}×{sname}: 전체 +{add_all:.2f}bp/day, test +{add_te:.2f}bp/day (가산분 — 0.70 미만 밴드는 현행과 서로소)")
