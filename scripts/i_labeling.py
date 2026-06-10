#!/usr/bin/env python3
"""
[I] 유사도 기반 거래 — 1단계: 라벨링(지표 계산)만.

⚠️ 이번엔 거래/예측 X. 매 1분 시점마다 t<=0 정보로만 지표(라벨)를 계산하고,
   그 분포/정상성/중복/시기안정성을 검증하는 것이 목적. forward return/target 없음.

데이터 (전부 t<=0):
  OB: ETHUSDT 50레벨 1Hz   trades: trades_perp/ETHUSDT (aggressor side+size, 초 단위)
  day-boundary wrap 제거: ts.dt.date==day (다음날 첫 snapshot 버림, CLAUDE.md 규율)

라벨 그룹 (넓게 — 닫힌 목록 X):
  가격/추세: MA이격(여러기간), MA기울기, 볼린저(위치/폭), RSI, 스토캐스틱, MACD, ADX
  변동성: rv(여러창), ATR, 밴드폭, vol surge비
  오더북: OBI(1/5/20/50/wtd), 호가동역학(dOBI), spread
  체결: aggressor flow(30s/1m/5m), 큰체결 net, 거래량 z(급증/압축)
  캔들: 몸통/꼬리 비율, range

가능한 한 scale-free(비율·z·bp·bounded) — 가격레벨 1000~4000 변동 + 시기 정규화 대비.
각 지표 causal: 현재 분(또는 그 마지막 초)까지의 정보만.
"""
import os, glob, json, time
import numpy as np
import pandas as pd

SYMBOL=os.environ.get('SYMBOL','ETHUSDT')   # SOL 재사용: SYMBOL=SOLUSDT
OB=f'/Users/mark/mark19_data/{SYMBOL}'
TR=f'/Users/mark/mark19_data/trades_perp/{SYMBOL}'
OUT=os.environ.get('LABEL_OUT','/Users/mark/Desktop/Mark/mark19/research/i_labeling')
os.makedirs(OUT,exist_ok=True)

STEP=int(os.environ.get('STEP','6'))   # 일 subsample (시기 전체 고르게). STEP=6 ~200일
LV=50
BURN_MIN=240                            # 분 burn-in (최장 SMA 240분 = 4h 정의 위해)
SAMPLE_DAYS_VIZ=['2023-06-15','2024-06-15','2025-06-15','2026-01-15']  # 시각화용 전체분 저장

obcols=['timestamp']
for i in range(LV):
    obcols+=[f'bid_{i}_price',f'bid_{i}_size',f'ask_{i}_price',f'ask_{i}_size']

def ffill_idx(so, n):
    """second -> source row index (last snapshot <= sec)."""
    idx=np.full(n,-1,dtype=int); idx[so]=np.arange(len(so))
    mask=idx>=0
    pos=np.where(mask,np.arange(n),0); np.maximum.accumulate(pos,out=pos)
    idx=idx[pos]; first=np.argmax(mask); idx[:first]=idx[first]
    return idx

# ---- TA helpers (전부 causal: pandas rolling/ewm 은 backward-looking) ----
def sma(s,n): return s.rolling(n,min_periods=n).mean()
def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def rsi(c,n=14):
    d=c.diff()
    up=d.clip(lower=0); dn=(-d).clip(lower=0)
    au=up.ewm(alpha=1/n,adjust=False).mean()
    ad=dn.ewm(alpha=1/n,adjust=False).mean()
    # 100-100/(1+au/ad) == 100*au/(au+ad). flat(au=ad=0) 은 이론상 중립 50
    # (구버전은 0 반환 — accuracy audit 에서 수정. 실데이터 발생 0건이었음)
    den=au+ad
    out=100*au/(den+1e-12)
    return out.where(den>1e-12, 50.0)

def stoch(c,h,l,n=14,d=3):
    ll=l.rolling(n,min_periods=n).min(); hh=h.rolling(n,min_periods=n).max()
    rngw=hh-ll
    # flat 14분(hh==ll) 은 '범위 내 위치' 미정의 → 이론상 중립 50
    # (구버전/TA-Lib 은 0 반환 = '바닥' 오신호. accuracy audit 에서 수정 — 정의 명시)
    k=(100*(c-ll)/(rngw+1e-12)).where(rngw>1e-12, 50.0)
    return k, k.rolling(d,min_periods=d).mean()

def macd(c,f=12,sl=26,sig=9):
    m=ema(c,f)-ema(c,sl); s=ema(m,sig)
    return m, s, m-s

def adx(h,l,c,n=14):
    pc=c.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    up=h.diff(); dn=-l.diff()
    pdm=((up>dn)&(up>0))*up
    ndm=((dn>up)&(dn>0))*dn
    atr=tr.ewm(alpha=1/n,adjust=False).mean()
    pdi=100*pdm.ewm(alpha=1/n,adjust=False).mean()/(atr+1e-12)
    ndi=100*ndm.ewm(alpha=1/n,adjust=False).mean()/(atr+1e-12)
    dx=100*(pdi-ndi).abs()/((pdi+ndi)+1e-12)
    return dx.ewm(alpha=1/n,adjust=False).mean(), pdi, ndi, atr

MA_PERIODS=[5,15,30,60,120,240]   # minutes
RV_WINS=[60,300,900,1800,3600]    # seconds

def process_day(day, trunc_sec=None, big_thr=None):
    """trunc_sec: 검증용 — 그 초 이후 데이터를 자른 것처럼 처리 (truncation invariance 테스트).
    big_thr: '큰 체결' 임계 (이전 처리일의 q95 — causal). None 이면 bigflow 라벨 NaN.
       구버전은 당일 전체 q95 사용 = 일중 lookahead (truncation 테스트로 검출, 수정)."""
    try:
        ob=pd.read_parquet(f'{OB}/{day}.parquet',columns=obcols)
    except Exception as e:
        return None,None
    ts=pd.to_datetime(ob['timestamp'],utc=True)
    keep=(ts.dt.date==pd.Timestamp(day).date()).values   # day-boundary wrap 제거
    ob=ob[keep].reset_index(drop=True); ts=ts[keep]
    if len(ob)<5000: return None,None
    so=np.round((ts-ts.iloc[0]).dt.total_seconds().values).astype(int)
    if trunc_sec is not None:
        tm=so<=trunc_sec
        ob=ob[tm].reset_index(drop=True); ts=ts[:tm.sum()]; so=so[tm]
    t0=ts.iloc[0]; n=int(so[-1])+1
    if n<BURN_MIN*60+120: return None,None

    bp=ob['bid_0_price'].values; ap=ob['ask_0_price'].values
    mid=(bp+ap)/2.0
    BS=ob[[f'bid_{i}_size' for i in range(LV)]].values
    AS=ob[[f'ask_{i}_size' for i in range(LV)]].values
    BP=ob[[f'bid_{i}_price' for i in range(LV)]].values
    AP=ob[[f'ask_{i}_price' for i in range(LV)]].values
    def obid(d):
        b=BS[:,:d].sum(1); a=AS[:,:d].sum(1); return (b-a)/(b+a+1e-9)
    o1,o5,o20,o50=obid(1),obid(5),obid(20),obid(50)
    mv=mid[:,None]
    wb=1.0/(1.0+np.abs(BP-mv)/0.01); wa=1.0/(1.0+np.abs(AP-mv)/0.01)
    owt=((BS*wb).sum(1)-(AS*wa).sum(1))/((BS*wb).sum(1)+(AS*wa).sum(1)+1e-9)
    spread=(ap-bp)/mid*1e4   # bp

    idx=ffill_idx(so,n)
    g_mid=mid[idx]; g_bp=bp[idx]; g_ap=ap[idx]
    g_o1,g_o5,g_o20,g_o50,g_owt=o1[idx],o5[idx],o20[idx],o50[idx],owt[idx]
    g_spr=spread[idx]

    # trades -> per-second
    try:
        tr=pd.read_parquet(f'{TR}/{day}.parquet',columns=['timestamp','side','size'])
    except Exception:
        return None,None
    tts=pd.to_datetime(tr['timestamp'],unit='s',utc=True)
    tsec=np.round((tts-t0).dt.total_seconds().values).astype(int)
    m=(tsec>=0)&(tsec<n)
    tsec=tsec[m]; tside=tr['side'].values[m]; tsz=tr['size'].values[m]
    buyv=np.zeros(n); sellv=np.zeros(n); bign=np.zeros(n); vols=np.zeros(n)
    isbuy=(tside=='Buy')
    np.add.at(buyv,tsec[isbuy],tsz[isbuy])
    np.add.at(sellv,tsec[~isbuy],tsz[~isbuy])
    np.add.at(vols,tsec,tsz)
    day_q95=float(np.quantile(tsz,0.95)) if len(tsz)>100 else None   # 다음 처리일에 전달 (causal)
    if big_thr is not None:
        big=tsz>=big_thr
        np.add.at(bign,tsec[big],np.where(isbuy[big],tsz[big],-tsz[big]))
    cbuy=np.cumsum(buyv); csell=np.cumsum(sellv); cbig=np.cumsum(bign); cvol=np.cumsum(vols)
    def flow(a,b):
        bb=cbuy[b]-cbuy[max(a,0)]; ss=csell[b]-csell[max(a,0)]; return (bb-ss)/(bb+ss+1e-9)

    # rv per-second log returns
    lr=np.zeros(n); lr[1:]=np.log(g_mid[1:]/g_mid[:-1]+1e-12)
    clr2=np.cumsum(lr*lr)
    def rv(s,w):
        a=max(s-w,0); k=s-a
        return np.sqrt((clr2[s]-clr2[a])/max(k,1))*1e4   # bp per-sec std

    # ---- minute bars from second grid ----
    nmin=n//60
    secs_end=np.arange(1,nmin+1)*60-1           # 각 분의 마지막 초 index (분 i: [i*60, i*60+59])
    secs_end=secs_end[secs_end<n]
    nmin=len(secs_end)
    C=g_mid[secs_end]
    O=g_mid[np.maximum(secs_end-59,0)]
    H=np.array([g_mid[max(e-59,0):e+1].max() for e in secs_end])
    L=np.array([g_mid[max(e-59,0):e+1].min() for e in secs_end])
    Vm=np.array([cvol[e]-cvol[max(e-60,0)] for e in secs_end])
    cs=pd.Series(C); hs=pd.Series(H); ls=pd.Series(L); os_=pd.Series(O)

    feat={}
    for k in MA_PERIODS:
        ma=sma(cs,k)
        feat[f'ma_dev_{k}']=((cs-ma)/(ma+1e-12)).values
        feat[f'ma_slope_{k}']=((ma-ma.shift(k))/(ma.shift(k)+1e-12)).values
    # ddof=0 (모집단 std): TA-Lib/pandas-ta/차트 표준과 일치 (accuracy audit — 구버전 ddof=1 은 ×1.026 상수배)
    bm=sma(cs,20); bsd=cs.rolling(20,min_periods=20).std(ddof=0)
    feat['boll_pos']=((cs-bm)/(bsd+1e-12)).values
    feat['boll_width']=(bsd/(bm+1e-12)).values
    feat['rsi_14']=rsi(cs,14).values
    feat['rsi_30']=rsi(cs,30).values
    sk,sd=stoch(cs,hs,ls,14,3); feat['stoch_k']=sk.values; feat['stoch_d']=sd.values
    mc,ms,mh=macd(cs); feat['macd']=(mc/(cs+1e-12)).values*1e4; feat['macd_hist']=(mh/(cs+1e-12)).values*1e4
    ax,pdi,ndi,atr=adx(hs,ls,cs,14)
    feat['adx_14']=ax.values; feat['di_diff']=(pdi-ndi).values
    feat['atr_14']=(atr/(cs+1e-12)).values*1e4
    feat['range_bp']=((hs-ls)/(os_+1e-12)).values*1e4
    hl=(H-L)+1e-12
    feat['body_ratio']=(C-O)/hl
    feat['upper_wick']=(H-np.maximum(O,C))/hl
    feat['lower_wick']=(np.minimum(O,C)-L)/hl

    # second-grid snapshot features at each minute-end
    e=secs_end
    feat['obi1']=g_o1[e]; feat['obi5']=g_o5[e]; feat['obi20']=g_o20[e]; feat['obi50']=g_o50[e]
    feat['obi_wtd']=g_owt[e]; feat['spread_bp']=g_spr[e]
    feat['dobi5_30']=g_o5[e]-g_o5[np.maximum(e-30,0)]
    feat['dobi5_60']=g_o5[e]-g_o5[np.maximum(e-60,0)]
    for w in RV_WINS:
        feat[f'rv_{w}']=np.array([rv(s,w) for s in e])
    feat['rv_ratio']=feat['rv_60']/(feat['rv_900']+1e-9)   # vol surge (단기/장기)
    feat['flow_30']=np.array([flow(s-30,s) for s in e])
    feat['flow_1m']=np.array([flow(s-60,s) for s in e])
    feat['flow_5m']=np.array([flow(s-300,s) for s in e])
    if big_thr is not None:
        feat['bigflow_5m']=np.array([(cbig[s]-cbig[max(s-300,0)]) for s in e])
    else:
        feat['bigflow_5m']=np.full(nmin,np.nan)   # 첫 처리일: causal 임계 없음
    # 거래량 z (급증/압축): 분거래량 log / 240분 rolling median
    vser=pd.Series(np.log1p(Vm))
    feat['vol_z']=((vser-vser.rolling(240,min_periods=30).median())/
                   (vser.rolling(240,min_periods=30).std()+1e-9)).values
    feat['bigflow_norm']=feat['bigflow_5m']/(np.array([cvol[s]-cvol[max(s-300,0)] for s in e])+1e-9)

    df=pd.DataFrame(feat)
    df.insert(0,'min_of_day',np.arange(nmin))
    df.insert(0,'sec',e)
    df.insert(0,'day',day)
    df.insert(0,'yr',day[:4])
    df['mid']=C   # 참고용 (라벨 아님)
    df=df.iloc[BURN_MIN:].reset_index(drop=True)   # burn-in 제거
    return df, day_q95

def main():
    all_days=sorted([d[:-8] for d in os.listdir(OB) if d.endswith('.parquet')])
    days=all_days[::STEP]
    # 시각화 sample day 포함 보장
    for sd in SAMPLE_DAYS_VIZ:
        if sd in all_days and sd not in days: days.append(sd)
    days=sorted(set(days))
    print(f'[setup] STEP={STEP} -> {len(days)} days, {days[0]}~{days[-1]}, BURN={BURN_MIN}min',flush=True)
    parts=[]; t0=time.time(); prev_q95=None
    for di,day in enumerate(days):
        df,q95=process_day(day,big_thr=prev_q95)   # 큰체결 임계 = 이전 처리일 q95 (causal)
        if q95 is not None: prev_q95=q95
        if df is not None:
            parts.append(df)
            if day in SAMPLE_DAYS_VIZ:
                df.to_parquet(f'{OUT}/vizday_{day}.parquet')
        if di%20==0:
            tot=sum(len(p) for p in parts)
            print(f'  [{di}/{len(days)}] {day} rows={tot} elapsed={time.time()-t0:.0f}s',flush=True)
    R=pd.concat(parts,ignore_index=True)
    R.to_parquet(f'{OUT}/labels.parquet')
    print(f'\n[done] {len(R)} rows, {R.day.nunique()} days, {len([c for c in R.columns if c not in ["yr","day","sec","min_of_day","mid"]])} labels')
    print('cols:',[c for c in R.columns])
    print(f'elapsed {time.time()-t0:.0f}s')

if __name__=='__main__':
    main()
