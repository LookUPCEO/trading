#!/usr/bin/env python3
"""[I] 24단계 — hold 길이 정밀화: A 분단위 최적(3h~5h) + B 신호별 조건부.
4h thr0.70 진입(동결) 고정, hold 만 변경. cross-day 가격 불필요(4h~5h 는 day 내 대부분).
과적합 위험 — OOS(train/test)+Bonferroni. 4h 정각 baseline 대비."""
import numpy as np, pandas as pd, json, datetime as dt
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB='/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
FEE=11.0; TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
HOLDS=list(range(180,301,10))   # 3h00~5h00, 10분 간격 (13개)

def main():
    # 4h thr0.70 진입 이벤트 (동결): lean70_v2 의 4h 신호 — 진입 시점(q)·방향
    R=pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
    ok=(R['4h_n']>=70)&~R['4h_frq'].isna()&(R['4h_frq']!=0)
    s=R[ok]; lean=(s['4h_fup']>=.7)|(s['4h_fup']<=.3); EV=s[lean].copy()
    EV['dir']=np.where(EV['4h_fup']>=.5,1,-1)
    # 진입 시점 (day,minute) — nrm 로 복원, 가격은 cross-day 연속(1d gap 무관, 5h<1d)
    nrm=pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
    qmod=nrm['min_of_day'].to_numpy()
    full=pd.read_parquet(LAB,columns=['day','min_of_day','mid'])
    fdays=sorted(full['day'].unique()); dixf={d:i for i,d in enumerate(fdays)}
    Pf=np.full(len(fdays)*1440,np.nan,np.float32)
    Pf[full['day'].map(dixf).to_numpy()*1440+full['min_of_day'].to_numpy()]=full['mid'].to_numpy(np.float32)
    NG=len(Pf)
    ndays=sorted(nrm['day'].unique());
    EV=EV.copy(); EV['mod']=qmod[EV['q'].to_numpy()]
    EV['gf']=nrm['day'].iloc[EV['q'].to_numpy()].map(dixf).to_numpy()*1440+EV['mod'].to_numpy()
    EV['z_atr']=nrm['z_atr_14'].iloc[EV['q'].to_numpy()].to_numpy()
    EV['strength']=np.maximum(EV['4h_fup'],1-EV['4h_fup'])
    EV['is_tr']=EV.quarter.isin(TRAIN_Q)
    qday=EV.qday.to_numpy()
    nd=EV.qday.nunique(); ndte=EV[~EV.is_tr].qday.nunique()
    print(f"[load] 4h thr0.70 진입 {len(EV)}건 (train {EV.is_tr.sum()}/test {(~EV.is_tr).sum()})")

    def net_at(hold, mask):
        gf=EV['gf'].to_numpy(); tgt=gf+hold; ok=tgt<NG
        m=mask&ok
        ret=EV['dir'].to_numpy()[m]*(Pf[tgt[m]]/Pf[gf[m]]-1)*1e4-FEE
        return EV.qday.to_numpy()[m], ret
    def dayci(qd,net):
        if len(net)<5: return np.nan,np.nan,np.nan
        dm=pd.Series(net).groupby(qd).mean().to_numpy()
        bs=np.random.default_rng(7).choice(dm,(4000,len(dm)),replace=True).mean(axis=1)
        return dm.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)

    tr=EV.is_tr.to_numpy(); te=~tr
    print("\n===== A: 분단위 hold 스캔 (3h~5h), train hit/net vs OOS =====")
    print(f"{'hold':>6} | train hit/net day | test net day")
    rows=[]
    for h in HOLDS:
        qd,net=net_at(h, np.ones(len(EV),bool))
        qdt,nt=net_at(h, tr); qde,ne=net_at(h, te)
        tr_dm=pd.Series(nt).groupby(qdt).mean().mean() if len(nt) else np.nan
        te_dm=pd.Series(ne).groupby(qde).mean().mean() if len(ne) else np.nan
        rows.append((h,tr_dm,te_dm,net.mean(),len(net)))
        mark='←4h' if h==240 else ''
        print(f"{h//60}h{h%60:02d} | {tr_dm:+7.1f} {(nt>0).mean() if len(nt) else 0:.3f} | {te_dm:+7.1f} {mark}")
    A=pd.DataFrame(rows,columns=['hold','tr','te','pt','n'])
    best_tr=A.loc[A.tr.idxmax()]
    print(f"\nA OOS 판정 (Bonferroni {len(HOLDS)} hold):")
    print(f"  train 최적 hold = {int(best_tr.hold)}m (train day {best_tr.tr:+.1f}) → test day {best_tr.te:+.1f}")
    print(f"  4h(240m): train {A[A.hold==240].tr.iloc[0]:+.1f} → test {A[A.hold==240].te.iloc[0]:+.1f}")
    print(f"  → train 최적이 test 서 4h 넘나: {'YES' if best_tr.te>A[A.hold==240].te.iloc[0] else 'NO(과적합)'}")

    print("\n===== B: 신호 특성 → 최적 hold 관계 (사전 식별 정보) =====")
    # 강신호(strength>=0.72) vs 약신호(0.70~0.72), 고변동(z_atr>0) vs 저변동 — 각 group 최적 hold
    for gname, gmask in [('강신호(str≥.72)',EV.strength>=.72),('약(.70-.72)',EV.strength<.72),
                         ('고변동(atr>0)',EV.z_atr>0),('저변동(atr≤0)',EV.z_atr<=0)]:
        gm=gmask.to_numpy()
        best_h=None;best_v=-1e9
        for h in HOLDS:
            qd,net=net_at(h,gm&tr)
            if len(net)<10: continue
            v=pd.Series(net).groupby(qd).mean().mean()
            if v>best_v: best_v=v;best_h=h
        print(f"  {gname} (n={gm.sum()}): train 최적 hold = {best_h}m (day {best_v:+.1f})")
    print("  → '강신호 길게/약신호 짧게' 류 단조 관계 있나 (있으면 B 룰 후보)")

    # B 룰: 사전등록 — strength>=.72 → 긴 hold(train 강신호 최적), else 4h. OOS.
    print("\n===== B 조건부 룰 OOS (사전등록: 강신호=강신호train최적hold, 약=4h) =====")
    # train 강신호 최적 hold 도출
    bh=240
    bv=-1e9
    for h in HOLDS:
        qd,net=net_at(h,(EV.strength>=.72).to_numpy()&tr)
        if len(net)<10: continue
        v=pd.Series(net).groupby(qd).mean().mean()
        if v>bv: bv=v; bh=h
    # test 적용: 강신호→bh, 약→240
    gf=EV['gf'].to_numpy()
    holds_per=np.where(EV.strength.to_numpy()>=.72, bh, 240)
    tgt=gf+holds_per; ok=tgt<NG; m=te&ok
    net_cond=EV['dir'].to_numpy()[m]*(Pf[tgt[m]]/Pf[gf[m]]-1)*1e4-FEE
    qd_cond=EV.qday.to_numpy()[m]
    dm,lo,hi=dayci(qd_cond,net_cond)
    qd4,net4=net_at(240,te); dm4,lo4,hi4=dayci(qd4,net4)
    print(f"  조건부(강신호 hold={bh}m): test day {dm:+.1f} [{lo:+.0f},{hi:+.0f}] 일수익 {net_cond.sum()/ndte:+.2f}")
    print(f"  고정 4h: test day {dm4:+.1f} [{lo4:+.0f},{hi4:+.0f}] 일수익 {net4.sum()/ndte:+.2f}")
    print(f"\n현행 4h 정각 0.084%/day. A/B 가 OOS 4h 넘으면 후보(라이브 즉시교체 X).")

if __name__=='__main__':
    main()
