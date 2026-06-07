#!/usr/bin/env python3
"""
[I] 6-1 B — 4h thr70 shadow 전향 검증 데몬 (돈 X, 기록만).

서브커맨드:
  build [--asof DAY] : 동결 artifact 생성 (DB 행렬·whitening·정규화 통계·big_thr)
                       asof 주면 그 날 기준 (replay 일치검증용), 없으면 최신.
  replay DAY         : 과거 날 raw 1Hz parquet 을 "실시간처럼" 분 단위 incremental 로
                       흘려 라벨→정규화→kNN fup 계산, 배치 파이프라인(labels_norm_reduced
                       + 동일 vote)과 대조. z max|Δ|, fup max|Δ| 보고. (백테스트=실시간 점검)
  run                : WS 라이브 데몬 — 매분 fup240 로그, thr70 신호 + 4h 후 결과 기록.
                       당일 1Hz OB/trades 도 기존 스키마로 persist (수집 재개 겸용).

정직 보장:
  - 라벨 = i_labeling 의 검증된 함수/로직 그대로 (truncation invariance 증명됨
    → day-so-far 재계산 = 배치와 동일). 신호는 min_of_day>=480 만 (배치와 동일 분포).
  - 정규화 = 과거 90 가용일 (수집 공백 시 stale — artifact 에 기준일 기록).
  - big_thr = 마지막 가용일 trades q95 (causal).
  - 신호 시점 = 분 末. 진입가 = 그 분 close mid (백테스트와 동일 관행).
"""
import os, sys, json, time, threading, logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i_labeling as IL
from i_labeling import sma, ema, rsi, stoch, macd, adx, ffill_idx
from i_reduce_norm import SIGNED, MAG_DRIFT, CENTERED50, STAT_MAG  # 검증된 분류 그대로

OUT = '/Users/mark/Desktop/Mark/mark19/research/i_similarity'
SHD = f'{OUT}/shadow'
LABP = '/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet'
OBD = IL.OB; TRD = IL.TR
K_CAND = 1000; N_IND = 100; THR = 0.70; MIN_VOTES = 70
D_WIN, D_MIN = 90, 15

# ─────────────────── 라벨 (i_labeling 의 분봉/지표 블록과 동일 — 함수 재사용) ───────────────────
def labels_from_seconds(g_mid, g_o1, g_o5, g_o20, g_o50, g_owt, g_spr,
                        cbuy, csell, cbig, cvol, n, big_thr):
    """day-so-far 초 그리드 → 분 라벨 DataFrame (i_labeling.process_day 의 라벨 블록 그대로)."""
    nmin = n // 60
    secs_end = np.arange(1, nmin + 1) * 60 - 1
    secs_end = secs_end[secs_end < n]; nmin = len(secs_end)
    C = g_mid[secs_end]
    O = g_mid[np.maximum(secs_end - 59, 0)]
    H = np.array([g_mid[max(e - 59, 0):e + 1].max() for e in secs_end])
    L = np.array([g_mid[max(e - 59, 0):e + 1].min() for e in secs_end])
    Vm = np.array([cvol[e] - cvol[max(e - 60, 0)] for e in secs_end])
    cs = pd.Series(C); hs = pd.Series(H); ls = pd.Series(L); os_ = pd.Series(O)
    lr = np.zeros(n); lr[1:] = np.log(g_mid[1:] / g_mid[:-1] + 1e-12)
    clr2 = np.cumsum(lr * lr)
    def rv(s, w_):
        a = max(s - w_, 0); k = s - a
        return np.sqrt((clr2[s] - clr2[a]) / max(k, 1)) * 1e4
    def flow(a, b):
        bb = cbuy[b] - cbuy[max(a, 0)]; ss = csell[b] - csell[max(a, 0)]
        return (bb - ss) / (bb + ss + 1e-9)
    feat = {}
    for k in IL.MA_PERIODS:
        ma = sma(cs, k)
        feat[f'ma_dev_{k}'] = ((cs - ma) / (ma + 1e-12)).values
        feat[f'ma_slope_{k}'] = ((ma - ma.shift(k)) / (ma.shift(k) + 1e-12)).values
    bm = sma(cs, 20); bsd = cs.rolling(20, min_periods=20).std(ddof=0)
    feat['boll_pos'] = ((cs - bm) / (bsd + 1e-12)).values
    feat['boll_width'] = (bsd / (bm + 1e-12)).values
    feat['rsi_14'] = rsi(cs, 14).values
    feat['rsi_30'] = rsi(cs, 30).values
    sk, sd_ = stoch(cs, hs, ls, 14, 3); feat['stoch_k'] = sk.values; feat['stoch_d'] = sd_.values
    mc, ms_, mh = macd(cs)
    feat['macd'] = (mc / (cs + 1e-12)).values * 1e4
    feat['macd_hist'] = (mh / (cs + 1e-12)).values * 1e4
    ax, pdi, ndi, atr = adx(hs, ls, cs, 14)
    feat['adx_14'] = ax.values; feat['di_diff'] = (pdi - ndi).values
    feat['atr_14'] = (atr / (cs + 1e-12)).values * 1e4
    feat['range_bp'] = ((hs - ls) / (os_ + 1e-12)).values * 1e4
    hl = (H - L) + 1e-12
    feat['body_ratio'] = (C - O) / hl
    feat['upper_wick'] = (H - np.maximum(O, C)) / hl
    feat['lower_wick'] = (np.minimum(O, C) - L) / hl
    e = secs_end
    feat['obi1'] = g_o1[e]; feat['obi5'] = g_o5[e]; feat['obi20'] = g_o20[e]; feat['obi50'] = g_o50[e]
    feat['obi_wtd'] = g_owt[e]; feat['spread_bp'] = g_spr[e]
    feat['dobi5_30'] = g_o5[e] - g_o5[np.maximum(e - 30, 0)]
    feat['dobi5_60'] = g_o5[e] - g_o5[np.maximum(e - 60, 0)]
    for w_ in IL.RV_WINS:
        feat[f'rv_{w_}'] = np.array([rv(s, w_) for s in e])
    feat['rv_ratio'] = feat['rv_60'] / (feat['rv_900'] + 1e-9)
    feat['flow_30'] = np.array([flow(s - 30, s) for s in e])
    feat['flow_1m'] = np.array([flow(s - 60, s) for s in e])
    feat['flow_5m'] = np.array([flow(s - 300, s) for s in e])
    if big_thr is not None:
        feat['bigflow_5m'] = np.array([(cbig[s] - cbig[max(s - 300, 0)]) for s in e])
    else:
        feat['bigflow_5m'] = np.full(nmin, np.nan)
    vser = pd.Series(np.log1p(Vm))
    feat['vol_z'] = ((vser - vser.rolling(240, min_periods=30).median()) /
                     (vser.rolling(240, min_periods=30).std() + 1e-9)).values
    feat['bigflow_norm'] = feat['bigflow_5m'] / (np.array([cvol[s] - cvol[max(s - 300, 0)] for s in e]) + 1e-9)
    df = pd.DataFrame(feat)
    df.insert(0, 'min_of_day', np.arange(nmin))
    df['mid'] = C
    return df

# ─────────────────── artifact ───────────────────
def build(asof=None):
    os.makedirs(SHD, exist_ok=True)
    nrm = pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet').sort_values(
        ['day', 'min_of_day']).reset_index(drop=True)
    meta = json.load(open(f'{OUT}/reduce_norm_meta.json'))
    reps = [r for r in meta['reps'] if r != 'spread_bp']
    yr = nrm['yr'].astype(int).to_numpy()
    days = sorted(nrm['day'].unique())
    if asof:
        days_db = [d for d in days if d < asof]
    else:
        days_db = days
    day_ix = {d: i for i, d in enumerate(days_db)}
    sel = nrm['day'].isin(days_db).to_numpy()
    sub = nrm[sel].reset_index(drop=True)
    drow = sub['day'].map(day_ix).to_numpy()
    mod = sub['min_of_day'].to_numpy()
    C = sub[[f'z_{c}' for c in reps]].to_numpy(np.float32)
    m23 = sub['yr'].astype(int).to_numpy() == 2023
    mu = C[m23].mean(0); S = np.cov((C[m23] - mu).T)
    w, V = np.linalg.eigh(S)
    W = (V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-6))) @ V.T).astype(np.float32)
    X = ((C - mu) @ W).astype(np.float32)
    # FR 4h
    lab = pd.read_parquet(LABP, columns=['day', 'min_of_day', 'mid'])
    lab = lab[lab.day.isin(days_db)]
    mids = np.full((len(days_db), 1440), np.nan, np.float32)
    mids[lab['day'].map(day_ix).to_numpy(), lab['min_of_day'].to_numpy()] = lab['mid'].to_numpy(np.float32)
    fr240 = np.full(len(sub), np.nan, np.float32)
    okm = mod + 240 <= 1439
    fr240[okm] = mids[drow[okm], mod[okm] + 240] / mids[drow[okm], mod[okm]] - 1
    # 정규화 통계: asof(또는 최신) 이전 D_WIN 가용일 per-day median/IQR (i_reduce_norm 동일)
    full = pd.read_parquet(LABP)
    lab47 = [c for c in full.columns if c not in ['yr', 'day', 'sec', 'min_of_day', 'mid']]
    for c in CENTERED50:
        full[c] = full[c] - 50.0
    fdays = sorted(full['day'].unique())
    use_days = [d for d in fdays if (asof is None or d < asof)][-D_WIN:]
    assert len(use_days) >= D_MIN, len(use_days)
    g = full[full.day.isin(use_days)].groupby('day')[lab47]
    med = g.median().median()
    scale = (g.quantile(0.75) - g.quantile(0.25)).median() / 1.349
    # big_thr: 마지막 가용일 trades q95
    last_day = use_days[-1]
    tr = pd.read_parquet(f'{TRD}/{last_day}.parquet', columns=['size'])
    big_thr = float(np.quantile(tr['size'].values, 0.95))
    np.savez_compressed(f'{SHD}/artifact{"_"+asof if asof else ""}.npz',
                        X=X, drow=drow, mod=mod, fr240=fr240, mu=mu, W=W,
                        med=med[lab47].to_numpy(), scale=scale[lab47].to_numpy())
    jmeta = dict(reps=reps, lab47=lab47, days_db=days_db, asof=asof or 'latest',
                 norm_window=[use_days[0], use_days[-1]], big_thr=big_thr,
                 signed=SIGNED, mag_drift=MAG_DRIFT, centered50=CENTERED50, built=datetime.now(timezone.utc).isoformat())
    json.dump(jmeta, open(f'{SHD}/artifact{"_"+asof if asof else ""}.json', 'w'))
    print(f"[build] DB {X.shape}, days={len(days_db)}, norm window {use_days[0]}~{use_days[-1]} "
          f"(stale 주의: asof={asof or 'latest'}), big_thr={big_thr:.3f}")

class Engine:
    def __init__(self, asof=None):
        tag = '_' + asof if asof else ''
        z = np.load(f'{SHD}/artifact{tag}.npz')
        self.meta = json.load(open(f'{SHD}/artifact{tag}.json'))
        self.X = z['X']; self.drow = z['drow']; self.mod = z['mod']; self.fr240 = z['fr240']
        self.mu = z['mu']; self.W = z['W']
        self.med = pd.Series(z['med'], index=self.meta['lab47'])
        self.scale = pd.Series(z['scale'], index=self.meta['lab47'])
        self.xsq = (self.X * self.X).sum(1)
    def normalize(self, row47):
        out = {}
        for c in self.meta['lab47']:
            x = row47[c]
            if c in self.meta['centered50']: x = x - 50.0   # rsi/stoch 중심화 (build 와 동일)
            s = self.scale[c] + 1e-12
            out[c] = (x - self.med[c]) / s if c in self.meta['mag_drift'] else x / s
        z = np.array([out[c] for c in self.meta['reps']], np.float32)
        return np.clip(z, -10, 10)
    def fup240(self, z21):
        q = (z21 - self.mu) @ self.W
        d2 = self.xsq - 2.0 * (self.X @ q.astype(np.float32))
        kc = min(K_CAND, len(d2) - 1)
        cand = np.argpartition(d2, kc)[:kc]
        order = cand[np.argsort(d2[cand])]
        acc = {}; picks = []
        for i in order:
            d = self.drow[i]; m = self.mod[i]
            lst = acc.get(d)
            if lst is not None:
                if any(abs(m - mm) < 240 for mm in lst): continue
                lst.append(m)
            else:
                acc[d] = [m]
            picks.append(i)
            if len(picks) >= N_IND: break
        v = self.fr240[picks]; v = v[~np.isnan(v)]; v = v[v != 0]
        if len(v) < MIN_VOTES: return np.nan, len(v)
        return float((v > 0).mean()), len(v)

# ─────────────────── replay (백테스트 = 실시간 일치 점검) ───────────────────
def replay(day):
    eng = Engine(asof=day)
    # raw 1Hz → 초 그리드 (process_day 와 동일 전처리)
    ob = pd.read_parquet(f'{OBD}/{day}.parquet', columns=IL.obcols)
    ts = pd.to_datetime(ob['timestamp'], utc=True)
    keep = (ts.dt.date == pd.Timestamp(day).date()).values
    ob = ob[keep].reset_index(drop=True); ts = ts[keep]
    so = np.round((ts - ts.iloc[0]).dt.total_seconds().values).astype(int)
    t0 = ts.iloc[0]; n = int(so[-1]) + 1
    bp = ob['bid_0_price'].values; ap = ob['ask_0_price'].values
    mid = (bp + ap) / 2.0
    BS = ob[[f'bid_{i}_size' for i in range(IL.LV)]].values
    AS = ob[[f'ask_{i}_size' for i in range(IL.LV)]].values
    BP = ob[[f'bid_{i}_price' for i in range(IL.LV)]].values
    AP = ob[[f'ask_{i}_price' for i in range(IL.LV)]].values
    def obid(d):
        b = BS[:, :d].sum(1); a = AS[:, :d].sum(1); return (b - a) / (b + a + 1e-9)
    o1, o5, o20, o50 = obid(1), obid(5), obid(20), obid(50)
    mv = mid[:, None]
    wb = 1.0 / (1.0 + np.abs(BP - mv) / 0.01); wa = 1.0 / (1.0 + np.abs(AP - mv) / 0.01)
    owt = ((BS * wb).sum(1) - (AS * wa).sum(1)) / ((BS * wb).sum(1) + (AS * wa).sum(1) + 1e-9)
    spread = (ap - bp) / mid * 1e4
    idx = ffill_idx(so, n)
    g = dict(g_mid=mid[idx], g_o1=o1[idx], g_o5=o5[idx], g_o20=o20[idx], g_o50=o50[idx],
             g_owt=owt[idx], g_spr=spread[idx])
    tr = pd.read_parquet(f'{TRD}/{day}.parquet', columns=['timestamp', 'side', 'size'])
    tts = pd.to_datetime(tr['timestamp'], unit='s', utc=True)
    tsec = np.round((tts - t0).dt.total_seconds().values).astype(int)
    m = (tsec >= 0) & (tsec < n)
    tsec = tsec[m]; tside = tr['side'].values[m]; tsz = tr['size'].values[m]
    buyv = np.zeros(n); sellv = np.zeros(n); bign = np.zeros(n); vols = np.zeros(n)
    isbuy = (tside == 'Buy')
    np.add.at(buyv, tsec[isbuy], tsz[isbuy])
    np.add.at(sellv, tsec[~isbuy], tsz[~isbuy])
    np.add.at(vols, tsec, tsz)
    big_thr = eng.meta['big_thr']
    big = tsz >= big_thr
    np.add.at(bign, tsec[big], np.where(isbuy[big], tsz[big], -tsz[big]))
    cb, cs_, cg, cv = np.cumsum(buyv), np.cumsum(sellv), np.cumsum(bign), np.cumsum(vols)

    # incremental: 매 30분마다 day-so-far 로 라벨 재계산해 마지막 분들 검증 (전수는 비싸서
    # 끝분 기준 — truncation invariance 가 이미 증명되었으므로 spot 검증 + 최종 전수)
    checks = []
    for upto_min in range(480, n // 60, 37):   # 비균일 spot
        T = upto_min * 60 + 59
        if T >= n: break
        df = labels_from_seconds(g['g_mid'][:T+1], g['g_o1'][:T+1], g['g_o5'][:T+1],
                                 g['g_o20'][:T+1], g['g_o50'][:T+1], g['g_owt'][:T+1],
                                 g['g_spr'][:T+1], cb[:T+1], cs_[:T+1], cg[:T+1], cv[:T+1],
                                 T + 1, big_thr)
        checks.append(df.iloc[-1])
    inc = pd.DataFrame(checks).reset_index(drop=True)
    # 배치 (process_day, 동일 big_thr)
    batch, _ = IL.process_day(day, big_thr=big_thr)
    bsel = batch.set_index('min_of_day').loc[inc['min_of_day'].astype(int)]
    lab47 = eng.meta['lab47']
    dz = []
    for c in lab47:
        a = inc[c].to_numpy(float); b = bsel[c].to_numpy(float)
        m_ = ~(np.isnan(a) & np.isnan(b))
        dz.append(np.nanmax(np.abs(a[m_] - b[m_])) if m_.any() else 0.0)
    print(f"[replay {day}] incremental vs batch 라벨: spot {len(inc)}분, max|Δ| = {max(dz):.2e} "
          f"({'PASS' if max(dz) < 1e-9 else 'FAIL'})")
    # fup 대조: incremental 정규화→kNN vs 배치 z (labels_norm_reduced) →kNN
    nrm = pd.read_parquet(f'{OUT}/labels_norm_reduced.parquet')
    nd = nrm[nrm.day == day].set_index('min_of_day')
    fups_i, fups_b, dzs = [], [], []
    for _, row in inc.iterrows():
        mo = int(row['min_of_day'])
        if mo not in nd.index: continue
        z_i = eng.normalize(row)
        z_b = nd.loc[mo, [f'z_{c}' for c in eng.meta['reps']]].to_numpy(np.float32)
        dzs.append(float(np.abs(z_i - z_b).max()))
        fi, _ = eng.fup240(z_i); fb, _ = eng.fup240(z_b)
        fups_i.append(fi); fups_b.append(fb)
    fups_i = np.array(fups_i); fups_b = np.array(fups_b)
    m_ = ~(np.isnan(fups_i) | np.isnan(fups_b))
    print(f"[replay {day}] z max|Δ| = {max(dzs):.2e} | fup240 max|Δ| = "
          f"{np.abs(fups_i[m_]-fups_b[m_]).max():.2e} "
          f"({'PASS' if np.abs(fups_i[m_]-fups_b[m_]).max() < 1e-6 else 'FAIL'}) "
          f"| 신호일치: {(np.sign(fups_i[m_]-.5)==np.sign(fups_b[m_]-.5)).mean():.3f}")

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'build'
    if cmd == 'build':
        build(asof=sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == 'replay':
        replay(sys.argv[2])
    elif cmd == 'run':
        from i_shadow_daemon import run_daemon
        run_daemon(Engine())
