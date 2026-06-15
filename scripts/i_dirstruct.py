#!/usr/bin/env python3
"""[I] 17단계 — thr 낮춤 + 방향 구조 패턴 필터. 약신호 사전식별 3번째 (6-4 부분집합·10-2 보완동의 ❌ 후).
구조 = (5m,10m,30m,1h,4h) lean 부호 조합 (t≤0). 강도(9단계) 아닌 구조 (다른 축)."""
import itertools
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
FEE = 11.0
TRAIN_Q = ['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
H5 = ['5m','10m','30m','1h','4h']   # 5m/10m 은 v2 parquet 에 있음

v2 = pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
R = v2.copy()
R['is_tr'] = R.quarter.isin(TRAIN_Q)
nd = R.qday.nunique(); nd_te = R[~R.is_tr].qday.nunique()
frq4 = R['4h_frq'].to_numpy(); v4 = ~np.isnan(frq4) & (frq4 != 0)

def struct_code(df, thr_low):
    """각 horizon lean 부호 (strength>=thr_low 발화 시 ±1, 미발화 0). 구조 = 5튜플."""
    codes = np.zeros((len(df), len(H5)), np.int8)
    for j,h in enumerate(H5):
        f=df[f'{h}_fup'].to_numpy(); nn=df[f'{h}_n'].to_numpy()
        st=np.where(nn>=70, np.maximum(f,1-f), np.nan); sd=np.where(f>=.5,1,-1)
        codes[:,j]=np.where((st>=thr_low)&~np.isnan(st), sd, 0)
    return codes

def ci(qd, net):
    if len(net)<5: return np.nan,np.nan,np.nan
    dm=pd.Series(net).groupby(qd).mean().to_numpy()
    bs=np.random.default_rng(7).choice(dm,(4000,len(dm)),replace=True).mean(axis=1)
    return dm.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)

print(f"[load] {len(R)} 쿼리. 방향 구조 = (5m,10m,30m,1h,4h) lean 부호.")
print(f"baseline 단일4h thr0.70: 일수익 +8.40 (full) / test +2.90\n")

rows=[]; ntests=0
for thr_low in [0.55, 0.60, 0.65]:
    codes = struct_code(R, thr_low)
    # 구조 = 부호 튜플. 4h 방향 = 발화 horizon 다수결(합의방향). 거래 = 그 방향 4h hold.
    cdir = np.sign(codes.sum(1))
    # 구조 식별: 부호 패턴 문자열 (발화한 것만; 0 포함 32^... → 실제 등장 구조만 집계)
    nz = (codes != 0).sum(1)
    active = (nz >= 1) & (cdir != 0) & v4
    code_str = np.array(['|'.join(map(str, row)) for row in codes])
    s = R[active].copy()
    s['code']=code_str[active]; s['cdir']=cdir[active]
    s['net']=cdir[active]*frq4[active]*1e4-FEE
    # 구조별 train/test
    for code, g in s.groupby('code'):
        if len(g)<30: continue   # 충분 표본만
        ntests+=1
        tr=g[g.is_tr]; te=g[~g.is_tr]
        if len(tr)<15: continue
        rows.append(dict(thr=thr_low, code=code, n=len(g), n_tr=len(tr), n_te=len(te),
                         hit=float((g.net+FEE>0).mean()),
                         net_tr=tr.net.mean(), net_te=te.net.mean() if len(te) else np.nan,
                         daily_full=g.net.sum()/nd, daily_te=te.net.sum()/nd_te if len(te) else np.nan,
                         freq=len(g)/nd))
T=pd.DataFrame(rows)
T.to_csv(f'{OUT}/dirstruct.csv', index=False)
print(f"===== 작업1+2: 구조×thr (n>=30) {len(T)}개 (Bonferroni 분모 ~{ntests}) =====")
print("train net 상위 12 (예측력 구조 후보):")
top = T.sort_values('net_tr', ascending=False).head(12)
for _,r in top.iterrows():
    print(f"  thr{r.thr} [{r.code}] n={r.n}(빈도{r.freq:.3f}/d) hit{r.hit:.3f} train{r.net_tr:+.1f} test{r.net_te:+.1f} 일수익_te{r.daily_te:+.2f}")

print(f"\n===== 작업3+4: train 양수 구조 → OOS 생존 (현행 test +2.90 대비) =====")
cand = T[(T.net_tr>0)&(T.n_tr>=15)].sort_values('net_tr',ascending=False)
print(f"train net>0 구조 {len(cand)}개 → OOS:")
surv=0
for _,r in cand.head(10).iterrows():
    thr_low=r.thr; code=r.code
    codes=struct_code(R,thr_low); cdir=np.sign(codes.sum(1))
    cs=np.array(['|'.join(map(str,row)) for row in codes])
    m=(cs==code)&v4&~R.is_tr.to_numpy()
    if m.sum()<5: print(f"  thr{thr_low}[{code}]: test n<5"); continue
    net=cdir[m]*frq4[m]*1e4-FEE
    dm,lo,hi=ci(R.qday.to_numpy()[m], net)
    ok='✓' if lo>0 else '✗'; surv+= (lo>0)
    print(f"  thr{thr_low}[{code}]: train{r.net_tr:+.1f}(n{r.n_tr}) → test n={m.sum()} hit{(net>0).mean():.3f} net{net.mean():+.1f} day[{lo:+.0f},{hi:+.0f}]{ok} 일수익{net.sum()/nd_te:+.2f}")
print(f"\nOOS 95% 생존 구조: {surv}개. Bonferroni 분모 ~{ntests}.")
# '눌림목' 류 명시 확인: 단기 음 장기 양 (5m,10m <0, 1h,4h >0)
print("\n눌림목류 구조 (5m,10m 음 / 1h,4h 양) 직접 확인:")
for thr_low in [0.60]:
    codes=struct_code(R,thr_low); cdir=np.sign(codes.sum(1))
    pull=(codes[:,0]<=0)&(codes[:,1]<=0)&(codes[:,3]>0)&(codes[:,4]>0)&v4
    if pull.sum()>=10:
        net=np.sign(codes[pull].sum(1))*frq4[pull]*1e4-FEE
        tr_m=pull&R.is_tr.to_numpy(); te_m=pull&~R.is_tr.to_numpy()
        print(f"  thr{thr_low} 눌림목: n={pull.sum()} hit{(net>0).mean():.3f} "
              f"train{(np.sign(codes[tr_m].sum(1))*frq4[tr_m]*1e4-FEE).mean():+.1f} "
              f"test{(np.sign(codes[te_m].sum(1))*frq4[te_m]*1e4-FEE).mean() if te_m.sum() else float('nan'):+.1f}")
    else:
        print(f"  thr{thr_low} 눌림목: n={pull.sum()} (희소)")
