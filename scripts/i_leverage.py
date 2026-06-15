#!/usr/bin/env python3
"""[I] 16단계 — '한 방 최대화': 4h thr0.70 단독 Kelly + 레버리지 파산확률 + 합의선별.
파산확률이 진짜 질문 (기대값 아님). 몬테카를로 부트스트랩 경로. $180 자본."""
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
FEE = 11.0
TRAIN_Q = ['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
H5 = ['30m','45m','1h','2h','4h']
CAP = 180.0; LOSS_HALT_USD = 60.0   # -33% = 손실한도

v2 = pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
hf = pd.read_parquet(f'{OUT}/lean70_v2_per_query_hfine.parquet')
R = v2.merge(hf[['q']+[c for c in hf.columns if c.split('_')[0] in ('45m','2h')]], on='q', how='inner')
R['is_tr'] = R.quarter.isin(TRAIN_Q)

# 4h thr0.70 단독 거래당 수익률(비율, fee 차감)
def ev4(df):
    ok=(df['4h_n']>=70)&~df['4h_frq'].isna()&(df['4h_frq']!=0)
    s=df[ok]; lean=(s['4h_fup']>=.7)|(s['4h_fup']<=.3); L=s[lean]
    sgn=np.where(L['4h_fup']>=.5,1.,-1.)
    ret=sgn*L['4h_frq'].to_numpy() - FEE/1e4   # 비율 (자본대비, 노출 1x 기준)
    return L, ret
L4, ret4 = ev4(R)
tr = L4.is_tr.to_numpy()
print(f"[load] 4h thr0.70 단독 {len(ret4)}건 (train {tr.sum()} / OOS {(~tr).sum()})")

# ===== 작업1: Kelly f* (4h 단독) train vs OOS 안정성 =====
def kelly(r):
    mu=r.mean(); var=r.var(); return mu/var if var>0 else 0
print("\n===== 작업1: 4h 단독 Kelly f* =====")
print(f"  전체: mean {ret4.mean()*1e4:+.0f}bp std {ret4.std()*1e4:.0f} hit {(ret4>0).mean():.3f} f*={kelly(ret4):.1f}x")
print(f"  train: mean {ret4[tr].mean()*1e4:+.0f}bp f*={kelly(ret4[tr]):.1f}x | OOS: mean {ret4[~tr].mean()*1e4:+.0f}bp f*={kelly(ret4[~tr]):.1f}x")
print(f"  → f* train vs OOS 안정성 (14단계 합의 n=9 24x 불안정과 대조)")

# ===== 작업2: 레버리지별 몬테카를로 파산확률 =====
# 신호 빈도: 79건 / 851일 = 0.093/day. 1년(365일) ≈ 34거래.
N_PER_YEAR = int(round(len(ret4)/851*365))
print(f"\n===== 작업2: 레버리지별 몬테카를로 (1년 ≈ {N_PER_YEAR}거래, 부트스트랩 {len(ret4)}건 분포) =====")
print(f"파산 = 자본 -{LOSS_HALT_USD/CAP*100:.0f}%(손실한도) 도달. 청산 = -100%. NSIM=20000")
rng = np.random.default_rng(7)
NSIM = 20000
def montecarlo(pool, lev, n_trades, halt=-LOSS_HALT_USD/CAP):
    # 각 sim: n_trades 부트스트랩, 복리, halt 도달 시 중단
    draws = rng.choice(pool, (NSIM, n_trades), replace=True)
    finals=[]; halts=0; liqs=0; maxdds=[]
    for s in range(NSIM):
        bank=1.0; peak=1.0; dd=0; halted=False
        for r in draws[s]:
            bank *= (1 + lev*r)
            if bank<=0.0: bank=1e-9; liqs+=1; halted=True; break
            peak=max(peak,bank); dd=max(dd,(peak-bank)/peak)
            if bank-1 <= halt: halts+=1; halted=True; break
        finals.append(bank); maxdds.append(dd)
    finals=np.array(finals); maxdds=np.array(maxdds)
    ann=(np.median(finals)-1)*100
    daily=ann/365  # 거래일 아닌 달력일 근사
    return dict(lev=lev, med_final=np.median(finals), ann_med=ann, daily=daily,
                ruin=(halts+liqs)/NSIM*100, liq=liqs/NSIM*100, maxdd_med=np.median(maxdds)*100,
                p10=(np.percentile(finals,10)-1)*100)
print(f"{'lev':>4} | 연수익(중앙) | 일수익 | 파산확률 | 청산확률 | maxDD중앙 | p10수익")
for lev in [1,2,3,5,8]:
    m=montecarlo(ret4, lev, N_PER_YEAR)
    print(f"{lev:>3}x | {m['ann_med']:+10.1f}% | {m['daily']:+.3f}% | {m['ruin']:6.1f}% | {m['liq']:6.1f}% | {m['maxdd_med']:6.0f}% | {m['p10']:+.0f}%")

# OOS 분포로도 (감쇠 반영 — OOS per-trade 약함)
print(f"\n  [OOS 거래당 분포로 (최근, 감쇠 반영) mean {ret4[~tr].mean()*1e4:+.0f}bp]:")
for lev in [1,2,3,5]:
    m=montecarlo(ret4[~tr], lev, N_PER_YEAR)
    print(f"{lev:>3}x | {m['ann_med']:+10.1f}% | {m['daily']:+.3f}% | 파산 {m['ruin']:.1f}% | 청산 {m['liq']:.1f}% | maxDD {m['maxdd_med']:.0f}%")

# ===== 작업3: 합의선별 k3/4/5 거래당 분포 =====
def consensus(df, thr=0.70):
    sig=np.zeros((len(df),len(H5)),np.int8)
    for j,h in enumerate(H5):
        f=df[f'{h}_fup'].to_numpy(); nn=df[f'{h}_n'].to_numpy()
        st=np.where(nn>=70,np.maximum(f,1-f),np.nan); sd=np.where(f>=.5,1,-1)
        sig[:,j]=np.where((st>=thr)&~np.isnan(st),sd,0)
    cdir=np.sign(sig.sum(1)); k=np.where(cdir>0,(sig>0).sum(1),np.where(cdir<0,(sig<0).sum(1),0))
    return cdir,k.astype(int)
cd,k=consensus(R); frq4=R['4h_frq'].to_numpy(); v=~np.isnan(frq4)&(frq4!=0)
print("\n===== 작업3: 합의선별 k별 거래당 (4h 실현) — 빈도 vs 거래당 =====")
for kk in [1,3,4,5]:
    m=(k>=kk)&v
    ret=cd[m]*frq4[m]-FEE/1e4
    if m.sum()<5: print(f"k>={kk}: n={m.sum()}"); continue
    freq=m.sum()/851
    print(f"k>={kk}: n={m.sum()} (빈도 {freq:.3f}/day, {1/freq:.0f}일 1건) mean {ret.mean()*1e4:+.0f}bp hit {(ret>0).mean():.3f} f*={kelly(ret):.1f}x")

print("\n현행 단일4h 1x = ~0.084%/day, 목표 0.5%. 레버리지=증폭(변동도×), 파산확률이 판정.")
print("⚠️ OOS f* 불안정/감쇠 시 레버리지는 독. 합의 k4/5 n<20 — 방향성만.")

# ===== 작업4 보강: 감쇠 시나리오 (per-trade mean 스케일) — 레버리지가 독 되는 지점 =====
print("\n===== 작업4: 감쇠 시나리오 — mean 을 x배 스케일 (1.0=현재, 0=edge소멸) =====")
print("(분포 모양 유지, 평균만 이동 → 감쇠 시 레버리지 파산확률)")
def scale_mean(r, f):
    return (r - r.mean()) + r.mean()*f   # 분산 유지, 평균만 f배
print(f"{'감쇠':>6} | 3x 일수익/파산 | 5x 일수익/파산")
for f in [1.0, 0.7, 0.5, 0.3, 0.0, -0.3]:
    rs = scale_mean(ret4, f)
    m3 = montecarlo(rs, 3, N_PER_YEAR); m5 = montecarlo(rs, 5, N_PER_YEAR)
    print(f"x{f:>4.1f} | {m3['daily']:+.3f}%/{m3['ruin']:4.1f}% | {m5['daily']:+.3f}%/{m5['ruin']:5.1f}%")
print("→ 감쇠 0.5 (절반)면 5x 파산↑, edge 소멸(0)이면 레버리지=순손실. shadow 가 감쇠 판정 = 레버리지 결정 선결.")
