#!/usr/bin/env python3
"""[I] 25단계 — 방향별 최적 horizon (롱/숏 비대칭).
가설: 상승=계단식(느림)→긴 horizon, 하락=엘리베이터(빠름)→짧은 horizon.
20단계 단서: 1d up hit 0.64(n237) vs down 0.458(n24) → 하락이 긴 horizon서 약함.
  본인 질문 = "하락이 짧은 horizon서 강한가" (미탐).
방법: 기존 per_query parquet 2개 병합(추가 검색 X) — 각 쿼리×horizon 의
  fup(합의)·n(표수)·frq(쿼리 실제 미래, 구조적 OOS) 사용.
  up-lean = fup>=thr & n>=70 (롱), down-lean = fup<=1-thr & n>=70 (숏).
  net/trade = dir*frq*1e4 - FEE. hit = dir*frq>0. day-cluster bootstrap CI.
  train(2024Q1~2025Q2) 에서 방향별 최적 horizon 선택 → test(2025Q3~) OOS 판정.
  Bonferroni 분모 = 방향(2) × horizon 수. 하락 n 작으면 방향성만(단정 X).
lookahead 0: frq = 쿼리 자신의 미래(pool=과거 prefix 밖). 진입=라벨직후 다음분."""
import numpy as np, pandas as pd
OUT='/Users/mark/Desktop/Mark/mark19/research/i_similarity'
FEE=11.0; THR=0.70; MINV=70
TRAIN_Q=['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
# 두 파일 병합: 전체 horizon 격자
HZ=['5m','10m','15m','20m','30m','45m','1h','2h','3h','4h','6h','8h']
HMIN={'5m':5,'10m':10,'15m':15,'20m':20,'30m':30,'45m':45,'1h':60,'2h':120,'3h':180,'4h':240,'6h':360,'8h':480}

def load():
    A=pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
    B=pd.read_parquet(f'{OUT}/lean70_v2_per_query_hfine.parquet')
    keep=['q','qday','qyr','quarter']
    bcols=[c for c in B.columns if c not in A.columns or c=='q']
    R=A.merge(B[bcols],on='q',how='inner')
    return R

def dayci(qd,net,seed=7):
    if len(net)<5: return np.nan,np.nan,np.nan
    dm=pd.Series(net).groupby(qd).mean().to_numpy()
    bs=np.random.default_rng(seed).choice(dm,(4000,len(dm)),replace=True).mean(axis=1)
    return dm.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)

def events(R, h, direction, thr=THR):
    """direction: 'up'|'down'. 반환 DataFrame(qday,quarter,net,hit,frq)."""
    ok=(R[f'{h}_n']>=MINV)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
    s=R[ok]
    if direction=='up':
        m=s[f'{h}_fup']>=thr; dirn=1
    else:
        m=s[f'{h}_fup']<=1-thr; dirn=-1
    L=s[m]
    if len(L)==0: return L.assign(net=[],hit=[])
    frq=L[f'{h}_frq'].to_numpy()
    net=dirn*frq*1e4-FEE
    return pd.DataFrame(dict(qday=L.qday.to_numpy(),quarter=L.quarter.to_numpy(),
                            qyr=L.qyr.to_numpy(),frq=frq,net=net,hit=(dirn*frq>0)))

def main():
    R=load()
    print(f"[load] {len(R)} queries (병합 horizon {len(HZ)})")
    tr_mask=lambda E: E.quarter.isin(TRAIN_Q)

    print("\n===== 작업1: 방향별 horizon hit/net (전체 / train / test) =====")
    print(f"{'hz':>4} | {'UP n':>5} {'hit':>5} {'gross':>7} {'net':>7} | {'DN n':>5} {'hit':>5} {'gross':>7} {'net':>7}")
    grid={}
    for h in HZ:
        line=f"{h:>4} | "
        for d in ['up','down']:
            E=events(R,h,d); grid[(h,d)]=E
            if len(E)<3: line+=f"{len(E):>5} {'--':>5} {'--':>7} {'--':>7} | "; continue
            line+=f"{len(E):>5} {E.hit.mean():>5.3f} {(E.net+FEE).mean():>+7.1f} {E.net.mean():>+7.1f} | "
        print(line)

    print("\n===== 작업2: 비대칭 — up 최적 vs down 최적 horizon (train 기준) =====")
    for d in ['up','down']:
        best_h=None;best_v=-1e9;tab=[]
        for h in HZ:
            E=grid[(h,d)]; Etr=E[tr_mask(E)]
            if len(Etr)<5: tab.append((h,len(Etr),np.nan,np.nan)); continue
            dm=pd.Series(Etr.net.to_numpy()).groupby(Etr.qday.to_numpy()).mean().mean()
            hitr=Etr.hit.mean()
            tab.append((h,len(Etr),hitr,dm))
            if dm>best_v: best_v=dm;best_h=h
        print(f"\n  [{d}] train day-net by horizon (n>=5):")
        for h,nn,hr,dm in tab:
            mk=' ←train최적' if h==best_h else ''
            print(f"    {h:>4}: n={nn:>4} hit {hr if not np.isnan(hr) else 0:.3f} day-net {dm if not np.isnan(dm) else 0:+7.1f}{mk}")
        print(f"  → [{d}] train 최적 horizon = {best_h}")

    print("\n  하락이 hit>0.5 되는 horizon (전체, 엘리베이터 가설):")
    for h in HZ:
        E=grid[(h,'down')]
        if len(E)<3: continue
        print(f"    {h:>4}: down n={len(E):>3} hit {E.hit.mean():.3f} {'(>0.5!)' if E.hit.mean()>0.5 else ''}")

    print("\n===== 작업3+4: 방향별 최적 horizon OOS (Bonferroni) =====")
    # train 최적 horizon 사전선택 → test 평가
    ndte={}  # test 일수 per direction
    for d in ['up','down']:
        bh=None;bv=-1e9
        for h in HZ:
            E=grid[(h,d)];Etr=E[tr_mask(E)]
            if len(Etr)<5:continue
            v=pd.Series(Etr.net.to_numpy()).groupby(Etr.qday.to_numpy()).mean().mean()
            if v>bv:bv=v;bh=h
        if bh is None: print(f"  [{d}] train 표본 부족 — skip"); continue
        E=grid[(bh,d)];Ete=E[~tr_mask(E)]
        dm,lo,hi=dayci(Ete.qday.to_numpy(),Ete.net.to_numpy())
        ndte[d]=Ete.qday.nunique()
        # 4h 동방향 baseline test
        E4=grid[('4h',d)];E4te=E4[~tr_mask(E4)]
        dm4,lo4,hi4=dayci(E4te.qday.to_numpy(),E4te.net.to_numpy())
        print(f"  [{d}] train최적 {bh}: test n={len(Ete)} hit {Ete.hit.mean() if len(Ete) else 0:.3f} "
              f"day-net {dm:+.1f} [{lo:+.0f},{hi:+.0f}]")
        print(f"       동방향 4h baseline: test n={len(E4te)} hit {E4te.hit.mean() if len(E4te) else 0:.3f} "
              f"day-net {dm4:+.1f} [{lo4:+.0f},{hi4:+.0f}]")

    nbonf=2*len([h for h in HZ])  # 방향×horizon
    print(f"\n  Bonferroni 분모 = {nbonf} (방향2 × horizon{len(HZ)})")

    print("\n===== 작업3: 방향별 운용 — one-way 겹침 + 빈도 + 일수익 =====")
    # 운영안: 롱=up최적h, 숏=down최적h. 같은 시각 동시 발생(one-way 충돌) 점검.
    # 단순화: 4h 합산(현행) vs 방향분리(up=up최적, down=down최적) 일수익 비교 (test).
    def updown_best(use_train_opt=True):
        res={}
        for d in ['up','down']:
            bh=None;bv=-1e9
            for h in HZ:
                E=grid[(h,d)];Etr=E[tr_mask(E)]
                if len(Etr)<5:continue
                v=pd.Series(Etr.net.to_numpy()).groupby(Etr.qday.to_numpy()).mean().mean()
                if v>bv:bv=v;bh=h
            res[d]=bh
        return res
    opt=updown_best()
    print(f"  방향분리 운영안: 롱={opt['up']} / 숏={opt['down']}")
    # 일수익 = 모든 test 거래 net 합 / test 거래일수 (단일포지션 1x 가정)
    # 현행: 4h 양방향 합산
    E4u=grid[('4h','up')];E4d=grid[('4h','down')]
    E4=pd.concat([E4u,E4d]);E4te=E4[~E4.quarter.isin(TRAIN_Q)]
    nd4=E4te.qday.nunique()
    print(f"  현행 4h 양방향: test 거래 {len(E4te)} (up {len(E4u[~E4u.quarter.isin(TRAIN_Q)])}/dn {len(E4d[~E4d.quarter.isin(TRAIN_Q)])}) "
          f"일수익 {E4te.net.sum()/max(nd4,1):+.2f}bp/day (거래일 {nd4})")
    Eu=grid[(opt['up'],'up')];Ed=grid[(opt['down'],'down')]
    Esep=pd.concat([Eu,Ed]);Esepte=Esep[~Esep.quarter.isin(TRAIN_Q)]
    # one-way 충돌: 같은 qday 에 up·down 둘 다 신호? (대략 — 분단위 충돌은 보수적으로 day 단위)
    ndsep=Esepte.qday.nunique()
    udays=set(Eu[~Eu.quarter.isin(TRAIN_Q)].qday);ddays=set(Ed[~Ed.quarter.isin(TRAIN_Q)].qday)
    overlap=udays&ddays
    print(f"  방향분리: test 거래 {len(Esepte)} (롱 {len(Eu[~Eu.quarter.isin(TRAIN_Q)])}/숏 {len(Ed[~Ed.quarter.isin(TRAIN_Q)])}) "
          f"일수익 {Esepte.net.sum()/max(ndsep,1):+.2f}bp/day (거래일 {ndsep})")
    print(f"  one-way 충돌(같은날 롱&숏 둘다): {len(overlap)}일 / 롱{len(udays)} 숏{len(ddays)} — {'겹침 적음(실행가능)' if len(overlap)<=2 else '겹침 주의'}")
    print(f"\n  현행 단일 4h 합산 일수익 ≈ 8.1bp/day(test) 기준. 방향분리가 넘나 = 위 비교.")

    print("\n===== 작업4-심화: down 단기 horizon 정밀 (엘리베이터 가설 핵심) =====")
    print("  down 단기서 hit>0.5 — 방향은 맞나 net 은? gross vs fee? OOS 유지?")
    for h in ['10m','15m','20m','30m','1h']:
        E=grid[(h,'down')]
        Etr=E[tr_mask(E)];Ete=E[~tr_mask(E)]
        g=(E.net+FEE).mean()
        dm,lo,hi=dayci(E.qday.to_numpy(),E.net.to_numpy())
        htr=Etr.hit.mean() if len(Etr) else np.nan
        hte=Ete.hit.mean() if len(Ete) else np.nan
        print(f"  down {h:>3}: n={len(E):>3} hit{E.hit.mean():.3f}(tr{htr if not np.isnan(htr) else 0:.2f}/te{hte if not np.isnan(hte) else 0:.2f}) "
              f"gross{g:+.1f} net{E.net.mean():+.1f} dayCI[{lo:+.0f},{hi:+.0f}] {'fee초과' if g>FEE else 'fee미달'}")
    print("\n  ↑ down 단기 hit>0.5(방향 맞음=엘리베이터 일부 진실) but gross<fee(작은 움직임) → 거래불가?")

    print("\n  down 2~3h 붕괴 (hit<0.5, 방향 역전?):")
    for h in ['2h','3h']:
        E=grid[(h,'down')]
        print(f"  down {h}: n={len(E)} hit {E.hit.mean():.3f} gross {(E.net+FEE).mean():+.1f} (역신호?)")

    print("\n  up vs down 방향정확도 비교 (같은 horizon, 단기서 down 이 더 정확?):")
    for h in ['10m','15m','20m','30m','1h']:
        u=grid[(h,'up')];d=grid[(h,'down')]
        print(f"  {h:>3}: up hit {u.hit.mean():.3f}(n{len(u)}) | down hit {d.hit.mean():.3f}(n{len(d)}) "
              f"{'← down 우세' if d.hit.mean()>u.hit.mean() else ''}")

if __name__=='__main__':
    main()
