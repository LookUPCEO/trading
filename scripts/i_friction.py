#!/usr/bin/env python3
"""[I] 11단계 — 실거래 마찰 정밀: 백테스트 net 에 지연/슬리피지/funding 추가 반영.
백테스트 net = gross - 11bp(taker RT 이미 포함). 추가 마찰을 실데이터로 측정. ETH 중심."""
import numpy as np
import pandas as pd

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
LAB = '/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
FUND = '/Users/mark/mark19_data/funding_ETHUSDT.parquet'
HOR = {'30m': 30, '1h': 60, '4h': 240}
TRAIN_Q = ['2024Q1','2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
FEE_TAKER = 11.0   # RT (5.5×2)
FEE_MAKER = 4.0    # RT (2×2)

# day 인덱싱 = nrm(labels_norm_reduced) 기준 (lean70_v2 의 qday 와 동일해야 정렬 일치)
nrm = pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(['day','min_of_day']).reset_index(drop=True)
days = sorted(nrm['day'].unique()); dix={d:i for i,d in enumerate(days)}
ND=len(days)
qmod = nrm['min_of_day'].to_numpy()
lab = pd.read_parquet(LAB, columns=['day','min_of_day','mid','spread_bp'])
lab = lab[lab.day.isin(days)]
mid = np.full((ND,1440), np.nan, np.float32)
spr = np.full((ND,1440), np.nan, np.float32)
di=lab['day'].map(dix).to_numpy(); mo=lab['min_of_day'].to_numpy()
mid[di,mo]=lab['mid'].to_numpy(np.float32); spr[di,mo]=lab['spread_bp'].to_numpy(np.float32)

# funding: 분 grid 에 8h 이벤트 누적 (UTC). day=date, minute=UTC분.
fund=pd.read_parquet(FUND)
fund['d']=fund.ts.dt.strftime('%Y-%m-%d'); fund['m']=fund.ts.dt.hour*60+fund.ts.dt.minute
frate=np.zeros((ND,1440),np.float32)
for _,r in fund.iterrows():
    if r['d'] in dix: frate[dix[r['d']], int(r['m'])]=r['funding_rate']
cum_f=np.cumsum(frate,axis=1)   # day 내 누적 (4h hold 는 day 내)

R=pd.read_parquet(f'{OUT}/lean70_v2_per_query.parquet')
qtr=R.quarter.to_numpy(); is_te=~R.quarter.isin(TRAIN_Q).to_numpy()
nd_all=R.qday.nunique()

def events(h, thr=0.70):
    ok=(R[f'{h}_n']>=70)&~R[f'{h}_frq'].isna()&(R[f'{h}_frq']!=0)
    s=R[ok]; lean=(s[f'{h}_fup']>=thr)|(s[f'{h}_fup']<=1-thr)
    L=s[lean]; sgn=np.where(L[f'{h}_fup']>=.5,1.,-1.)
    return L, sgn

def measure(h):
    L,sgn=events(h); H=HOR[h]
    dd=L.qday.to_numpy();
    # 쿼리 분 (q index → min_of_day) — nrm 정렬과 동일하게 lean parquet 에 mod 없음 → labels grid 로 복원
    # lean70_v2 에 min 없음: q 는 nrm 행인덱스. nrm 로드해 mod 매핑.
    return L,sgn,dd,H

results={}
for h,H in HOR.items():
    L,sgn=events(h)
    q=L['q'].to_numpy(); d=L.qday.to_numpy(); m0=qmod[q]
    te=is_te[L.index.to_numpy()] if False else L.quarter.isin(TRAIN_Q).to_numpy()==False
    # 백테스트: entry mid[m0], exit mid[m0+H]
    bt_gross=sgn*L[f'{h}_frq'].to_numpy()*1e4   # stage5 와 동일 (frq 직접)
    bt_net=bt_gross-FEE_TAKER
    # 지연: entry mid[m0+1], exit mid[m0+H+1]
    m1=np.minimum(m0+1,1439); mh1=np.minimum(m0+H+1,1439)
    e1=mid[d,m1]; x1=mid[d,mh1]
    dl_gross=sgn*(x1/e1-1)*1e4
    # 슬리피지 (taker: 진입 half-spread + 청산 half-spread = 1 spread RT). spread label at entry/exit.
    slip=(np.nan_to_num(spr[d,m1])+np.nan_to_num(spr[d,mh1]))/2*2/2  # ≈ (spr_in+spr_out)/2 (half each ×2legs)
    slip=(np.nan_to_num(spr[d,m1])+np.nan_to_num(spr[d,mh1]))/2     # half-spread per leg ×2 = avg spread
    # funding: hold (m1, mh1] 내 funding 합 × dir (롱이 +rate 지불 → PnL -dir*rate)
    fwin=(cum_f[d,mh1]-cum_f[d,m1])  # 비율
    fund_bp=-sgn*fwin*1e4
    net_taker=dl_gross - FEE_TAKER - slip + fund_bp
    net_maker=dl_gross - FEE_MAKER - slip + fund_bp   # maker (fill 위험 별도)
    valid=~np.isnan(net_taker)
    results[h]=dict(L=L,d=d,te=te,valid=valid,bt_net=bt_net,dl_gross=dl_gross,
                    slip=slip,fund_bp=fund_bp,net_taker=net_taker,net_maker=net_maker)

print("===== 작업1~4: 마찰 요소별 + 마찰 후 net (4h thr0.70 중심, ETH) =====")
print(f"{'h':4s} n  | bt_net | 지연Δ  슬립   funding | taker net | maker net | bt일수익→taker")
for h in HOR:
    r=results[h]; v=r['valid']
    bt=r['bt_net'][v]; dl=r['dl_gross'][v]-FEE_TAKER  # 지연만 반영 (slip/fund 전)
    delay_delta=(dl-bt).mean()
    print(f"{h:4s} {v.sum():3d} | {bt.mean():+6.1f} | {delay_delta:+5.1f}  {-r['slip'][v].mean():+5.2f}  {r['fund_bp'][v].mean():+6.2f} | "
          f"{r['net_taker'][v].mean():+7.1f} | {r['net_maker'][v].mean():+7.1f} | {bt.sum()/nd_all:+.2f}→{r['net_taker'][v].sum()/nd_all:+.2f}")

print("\n===== 결합 (30m+1h+4h) 마찰 후 일수익 (자본전액=상한, 단일4h=현실) =====")
for scen in ['net_taker','net_maker']:
    allnet=[]; alld=[]; te_net=[]
    for h in HOR:
        r=results[h]; v=r['valid']
        allnet.append(r[scen][v]); alld.append(r['d'][v]); te_net.append(r[scen][v & r['te']])
    an=np.concatenate(allnet);
    nd_te=R[is_te].qday.nunique()
    tn=np.concatenate(te_net)
    print(f"  {scen}: 결합 full {an.sum()/nd_all:+.2f}bp/day | test {tn.sum()/nd_te:+.2f} | 단일4h full {results['4h'][scen][results['4h']['valid']].sum()/nd_all:+.2f}")

# day-cluster CI (4h taker)
r=results['4h']; v=r['valid']
dm=pd.Series(r['net_taker'][v]).groupby(r['d'][v]).mean().to_numpy()
bs=np.random.default_rng(7).choice(dm,(5000,len(dm)),replace=True).mean(axis=1)
print(f"\n4h taker 마찰후: per-trade {r['net_taker'][v].mean():+.1f} day-mean {dm.mean():+.1f} CI[{np.percentile(bs,2.5):+.0f},{np.percentile(bs,97.5):+.0f}]")
print(f"funding 부호: 평균 {results['4h']['fund_bp'][v].mean():+.2f}bp (롱숏 평균), hold당 funding 이벤트 빈도 4h=0.5회")
print(f"\n현행 백테스트: 4h +90.5/trade, 결합 15.1bp/day. SOL: 스프레드 23x(슬립 1.66bp RT)지만 edge 없어 무의미.")
