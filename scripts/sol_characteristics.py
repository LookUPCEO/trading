#!/usr/bin/env python3
"""[SOL] 1단계 작업3 — SOL vs ETH 기초 특성 (겹치는 기간 동일 일자). 라벨/거래 X.
변동성(rv bp)/스프레드(bp)/거래량(notional)/유동성(OB depth) — 유리·불리 사전 단서."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from i_labeling import obcols, ffill_idx, LV

DATA = '/Users/mark/mark19_data'
# 겹치는 182일에서 시기 고르게 12일 샘플
import datetime as dt
sol_days = sorted(f[:-8] for f in os.listdir(f'{DATA}/SOLUSDT') if f.endswith('.parquet'))
SAMPLE = sol_days[::len(sol_days)//12][:12]

def day_stats(sym, day):
    try:
        ob = pd.read_parquet(f'{DATA}/{sym}/{day}.parquet', columns=obcols)
    except Exception:
        return None
    ts = pd.to_datetime(ob['timestamp'], utc=True)
    keep = (ts.dt.date == pd.Timestamp(day).date()).values
    ob = ob[keep].reset_index(drop=True); ts = ts[keep]
    if len(ob) < 5000: return None
    so = np.round((ts - ts.iloc[0]).dt.total_seconds().values).astype(int)
    n = int(so[-1]) + 1
    bp = ob['bid_0_price'].values; ap = ob['ask_0_price'].values
    mid = (bp + ap) / 2.0
    spread_bp = (ap - bp) / mid * 1e4
    idx = ffill_idx(so, n); g_mid = mid[idx]
    lr = np.diff(np.log(g_mid))
    rv_sec_bp = np.std(lr) * 1e4                  # per-sec log-ret std (bp)
    # top-5 depth (notional) — 유동성
    b5 = ob[[f'bid_{i}_size' for i in range(5)]].values.sum(1)
    a5 = ob[[f'ask_{i}_size' for i in range(5)]].values.sum(1)
    depth5_usd = np.nanmedian((b5 + a5) * mid)    # 양측 top5 notional
    tick = np.nanmedian(np.diff(np.sort(np.unique(bp))))  # 가격 틱 추정
    try:
        tr = pd.read_parquet(f'{DATA}/trades_perp/{sym}/{day}.parquet', columns=['size', 'price'])
        vol_notional = float((tr['size'] * tr['price']).sum())
        ntr = len(tr)
    except Exception:
        vol_notional = np.nan; ntr = 0
    return dict(sym=sym, day=day, price=float(np.nanmedian(mid)),
                rv_sec_bp=float(rv_sec_bp), spread_bp=float(np.nanmedian(spread_bp)),
                tick=float(tick), depth5_usd=float(depth5_usd),
                vol_musd=vol_notional / 1e6, ntrades=ntr)

rows = []
for day in SAMPLE:
    for sym in ['ETHUSDT', 'SOLUSDT']:
        r = day_stats(sym, day)
        if r: rows.append(r)
        print('.', end='', flush=True)
print()
D = pd.DataFrame(rows)
D.to_csv('/Users/mark/Desktop/Mark/mark19/research/sol/characteristics.csv', index=False)
print("\n===== SOL vs ETH 기초 특성 (겹치는 12일 중앙값) =====")
agg = D.groupby('sym').agg(price=('price', 'median'), rv_sec_bp=('rv_sec_bp', 'median'),
                           spread_bp=('spread_bp', 'median'), tick=('tick', 'median'),
                           depth5_kusd=('depth5_usd', lambda x: np.median(x) / 1e3),
                           vol_musd=('vol_musd', 'median'), ntrades=('ntrades', 'median'))
print(agg.round(3).T.to_string())
e = agg.loc['ETHUSDT']; s = agg.loc['SOLUSDT']
print("\n===== 비율 (SOL / ETH) — 유리·불리 단서 =====")
print(f"변동성 rv_sec:  SOL/ETH = {s.rv_sec_bp/e.rv_sec_bp:.2f}x  ({'SOL 변동성 큼=폭 유리' if s.rv_sec_bp>e.rv_sec_bp else 'SOL 작음'})")
print(f"스프레드 bp:    SOL/ETH = {s.spread_bp/e.spread_bp:.2f}x  ({'SOL 스프레드 큼=fill 불리' if s.spread_bp>e.spread_bp else 'SOL 작음'})")
print(f"  → rv/spread (신호여지/마찰): ETH {e.rv_sec_bp/e.spread_bp:.1f}, SOL {s.rv_sec_bp/s.spread_bp:.1f}")
print(f"depth5 유동성:  SOL/ETH = {s.depth5_kusd/e.depth5_kusd:.2f}x  ({'SOL 얇음=충격 큼' if s.depth5_kusd<e.depth5_kusd else 'SOL 두꺼움'})")
print(f"거래량 notional:SOL/ETH = {s.vol_musd/e.vol_musd:.2f}x")
print(f"\nfee 환경: non-VIP taker 5.5bp/leg 는 코인 무관 (% 기준 동일) — 단 SOL rv {s.rv_sec_bp:.2f}bp/sec 가 ETH 대비 큰지가 핵심")
