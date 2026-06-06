#!/usr/bin/env python3
"""
[I] 1단계 보강 — 라벨 정확성 깊이 검증. (유사도/거래 X — 정확성만)

part lib   (작업1): TA-Lib + pandas-ta 양쪽 대조 — 수치 일치 여부, 오차 분포
part synth (작업2+4): 합성 입력 이론 행동 + 엣지 (flat/갭/무체결/0분모)
part manual(작업3): OB/체결 라벨 수동 검산 (원시 parquet 에서 독립 경로로 손계산)
part trunc (작업5): truncation invariance — 미래 데이터 잘라도 과거 라벨 동일해야
                    (다르면 그 라벨은 day-level 미래정보 사용 = lookahead)
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import i_labeling as IL
from i_labeling import sma, ema, rsi, stoch, macd, adx, ffill_idx

OB = IL.OB; TR = IL.TR
OUT = '/Users/mark/Desktop/Mark/mark19/research/i_labeling/accuracy'
os.makedirs(OUT, exist_ok=True)
TEST_DAYS = ['2023-06-15', '2024-02-19', '2024-06-15', '2025-06-15', '2025-12-04', '2026-01-15']
BURN = 240

def build_bars(day, trunc_sec=None):
    """i_labeling.process_day 와 동일 로직의 분봉/초그리드 (대조용 복제)."""
    ob = pd.read_parquet(f'{OB}/{day}.parquet', columns=IL.obcols)
    ts = pd.to_datetime(ob['timestamp'], utc=True)
    keep = (ts.dt.date == pd.Timestamp(day).date()).values
    ob = ob[keep].reset_index(drop=True); ts = ts[keep]
    so = np.round((ts - ts.iloc[0]).dt.total_seconds().values).astype(int)
    if trunc_sec is not None:
        m = so <= trunc_sec
        ob = ob[m].reset_index(drop=True); so = so[m]
    t0 = ts.iloc[0]; n = int(so[-1]) + 1
    bp = ob['bid_0_price'].values; ap = ob['ask_0_price'].values
    mid = (bp + ap) / 2.0
    idx = ffill_idx(so, n)
    g_mid = mid[idx]
    secs_end = np.arange(1, n // 60 + 1) * 60 - 1
    secs_end = secs_end[secs_end < n]
    C = g_mid[secs_end]
    O = g_mid[np.maximum(secs_end - 59, 0)]
    H = np.array([g_mid[max(e - 59, 0):e + 1].max() for e in secs_end])
    L = np.array([g_mid[max(e - 59, 0):e + 1].min() for e in secs_end])
    return pd.Series(C), pd.Series(H), pd.Series(L), pd.Series(O), g_mid, secs_end, t0, n

# ════════ 작업1: 라이브러리 대조 ════════
def part_lib():
    import talib
    import pandas_ta as pta
    rows = []
    for day in TEST_DAYS:
        cs, hs, ls, os_, *_ = build_bars(day)
        C, Hh, Ll = cs.values, hs.values, ls.values
        ours = {}
        # 우리 구현 (i_labeling 그대로)
        for k in IL.MA_PERIODS:
            ours[f'ma(SMA{k})'] = sma(cs, k).values
        bm = sma(cs, 20); bsd = cs.rolling(20, min_periods=20).std()
        ours['boll_mid(SMA20)'] = bm.values
        ours['boll_sd'] = bsd.values
        ours['rsi_14'] = rsi(cs, 14).values
        ours['rsi_30'] = rsi(cs, 30).values
        sk, sd_ = stoch(cs, hs, ls, 14, 3)
        ours['stoch_k'] = sk.values; ours['stoch_d'] = sd_.values
        mc, msig, mh = macd(cs)
        ours['macd_raw'] = mc.values; ours['macd_hist_raw'] = mh.values
        ax, pdi, ndi, atr = adx(hs, ls, cs, 14)
        ours['adx_14'] = ax.values; ours['di_diff'] = (pdi - ndi).values
        ours['atr_raw'] = atr.values

        # TA-Lib
        tl = {}
        for k in IL.MA_PERIODS:
            tl[f'ma(SMA{k})'] = talib.SMA(C, k)
        ub, mb, lb = talib.BBANDS(C, 20, 2, 2, matype=0)
        tl['boll_mid(SMA20)'] = mb
        tl['boll_sd'] = (ub - mb) / 2.0          # TA-Lib: 모집단 std (ddof=0)
        tl['rsi_14'] = talib.RSI(C, 14)
        tl['rsi_30'] = talib.RSI(C, 30)
        fk, fd = talib.STOCHF(Hh, Ll, C, fastk_period=14, fastd_period=3, fastd_matype=0)
        tl['stoch_k'] = fk; tl['stoch_d'] = fd
        m_, s_, h_ = talib.MACD(C, 12, 26, 9)
        tl['macd_raw'] = m_; tl['macd_hist_raw'] = h_
        tl['adx_14'] = talib.ADX(Hh, Ll, C, 14)
        tl['di_diff'] = talib.PLUS_DI(Hh, Ll, C, 14) - talib.MINUS_DI(Hh, Ll, C, 14)
        tl['atr_raw'] = talib.ATR(Hh, Ll, C, 14)

        # pandas-ta
        pt = {}
        for k in IL.MA_PERIODS:
            pt[f'ma(SMA{k})'] = pta.sma(cs, length=k).values
        bb = pta.bbands(cs, length=20, std=2)
        pt['boll_mid(SMA20)'] = bb.iloc[:, 1].values
        pt['boll_sd'] = (bb.iloc[:, 2] - bb.iloc[:, 1]).values / 2.0   # pandas-ta: ddof=1 기본
        pt['rsi_14'] = pta.rsi(cs, length=14).values
        pt['rsi_30'] = pta.rsi(cs, length=30).values
        st = pta.stoch(hs, ls, cs, k=14, d=3, smooth_k=1)
        pt['stoch_k'] = st.iloc[:, 0].values; pt['stoch_d'] = st.iloc[:, 1].values
        mdf = pta.macd(cs, fast=12, slow=26, signal=9)
        pt['macd_raw'] = mdf.iloc[:, 0].values; pt['macd_hist_raw'] = mdf.iloc[:, 1].values
        adf = pta.adx(hs, ls, cs, length=14)
        pt['adx_14'] = adf.iloc[:, 0].values
        pt['di_diff'] = (adf.iloc[:, 1] - adf.iloc[:, 2]).values
        pt['atr_raw'] = pta.atr(hs, ls, cs, length=14).values

        for key in tl:
            o = ours[key][BURN:]
            scale = max(np.nanstd(o), 1e-9)
            for libname, ref in [('talib', tl[key][BURN:]), ('pta', pt[key][BURN:])]:
                m = ~(np.isnan(o) | np.isnan(ref))
                err = np.abs(o[m] - ref[m])
                rows.append(dict(day=day, label=key, lib=libname,
                                 max_abs=float(err.max()), med_abs=float(np.median(err)),
                                 p99_abs=float(np.quantile(err, .99)),
                                 max_rel_to_sd=float(err.max() / scale)))
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/lib_compare.csv', index=False)
    agg = R.groupby(['label', 'lib'])[['max_abs', 'med_abs', 'max_rel_to_sd']].max().round(8)
    print("===== 작업1: 라이브러리 대조 (전 test day 최악값, burn-in 240분 이후) =====")
    print(agg.to_string())
    # talib vs pta 자체 차이 참고
    return R

# ════════ 작업2+4: 합성 케이스 + 엣지 ════════
def part_synth():
    print("\n===== 작업2: 알려진 케이스 (이론 행동) =====")
    n = 600
    res = []
    def report(name, cond, val):
        res.append((name, bool(cond), val))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {val}")

    up = pd.Series(np.linspace(100, 200, n))             # 단조 상승
    dn = pd.Series(np.linspace(200, 100, n))             # 단조 하락
    rng_ = np.random.default_rng(3)
    flatN = pd.Series(100 + 0.01 * np.sin(np.arange(n)) + rng_.normal(0, .005, n))  # 횡보
    r_up = rsi(up, 14).iloc[-1]; report("단조상승 RSI→100", abs(r_up - 100) < 1e-6, f"{r_up:.6f}")
    r_dn = rsi(dn, 14).iloc[-1]; report("단조하락 RSI→0", abs(r_dn) < 1e-6, f"{r_dn:.6f}")
    r_fl = rsi(flatN, 14).iloc[-1]; report("횡보 RSI≈50", 35 < r_fl < 65, f"{r_fl:.2f}")
    k_up, _ = stoch(up, up, up, 14, 3)
    report("단조상승 Stoch→100", abs(k_up.iloc[-1] - 100) < 1e-6, f"{k_up.iloc[-1]:.6f}")
    k_dn, _ = stoch(dn, dn, dn, 14, 3)
    report("단조하락 Stoch→0", abs(k_dn.iloc[-1]) < 1e-6, f"{k_dn.iloc[-1]:.6f}")
    # 변동성 급증 → ATR/밴드폭 확대
    lowv = 100 + rng_.normal(0, .01, 300); highv = 100 + np.cumsum(rng_.normal(0, .5, 300))
    vs = pd.Series(np.concatenate([lowv, highv]))
    _, _, _, atr_ = adx(vs, vs, vs, 14)   # H=L=C → TR=|gap|만
    hh = vs.rolling(3).max(); ll = vs.rolling(3).min()
    _, _, _, atr2 = adx(hh, ll, vs, 14)
    bw1 = vs.rolling(20).std().iloc[290]; bw2 = vs.rolling(20).std().iloc[-1]
    report("변동성 급증 → 밴드폭 확대", bw2 > 5 * bw1, f"{bw1:.4f}→{bw2:.4f}")
    report("변동성 급증 → ATR 확대", atr2.iloc[-1] > 5 * atr2.iloc[290],
           f"{atr2.iloc[290]:.4f}→{atr2.iloc[-1]:.4f}")
    # MACD: 상승전환에서 양수
    vshape = pd.Series(np.concatenate([np.linspace(200, 100, 300), np.linspace(100, 180, 300)]))
    mc, _, _ = macd(vshape)
    report("상승국면 MACD>0", mc.iloc[-1] > 0, f"{mc.iloc[-1]:.3f}")
    report("하락국면 MACD<0", mc.iloc[299] < 0, f"{mc.iloc[299]:.3f}")
    # ADX: 추세에서 높고 횡보에서 낮음
    ax_t, pdi_t, ndi_t, _ = adx(up * 1.001, up * 0.999, up, 14)
    ax_f, *_ = adx(flatN * 1.0001, flatN * 0.9999, flatN, 14)
    report("추세 ADX 높음(>40)", ax_t.iloc[-1] > 40, f"{ax_t.iloc[-1]:.1f}")
    report("횡보 ADX 낮음(<25)", ax_f.iloc[-1] < 25, f"{ax_f.iloc[-1]:.1f}")
    report("상승추세 di_diff>0", (pdi_t - ndi_t).iloc[-1] > 0, f"{(pdi_t-ndi_t).iloc[-1]:.1f}")

    print("\n===== 작업4: 엣지 케이스 =====")
    # 완전 flat (체결없음/가격불변 구간 모사)
    fl = pd.Series(np.full(n, 100.0))
    r = rsi(fl, 14).iloc[-1]
    report("완전 flat RSI = 중립 50 이어야", abs(r - 50) < 1e-6, f"{r:.4f}  (au=ad=0 케이스)")
    k, _ = stoch(fl, fl, fl, 14, 3)
    report("완전 flat Stoch (hh==ll) 중립 50 권장", abs(k.iloc[-1] - 50) < 1e-6, f"{k.iloc[-1]:.4f}")
    bp_ = ((fl - sma(fl, 20)) / (fl.rolling(20).std() + 1e-12)).iloc[-1]
    report("완전 flat boll_pos 유한(0)", np.isfinite(bp_) and abs(bp_) < 1e-3, f"{bp_:.4f}")
    ax_, pdi_, ndi_, atr_ = adx(fl, fl, fl, 14)
    report("완전 flat ATR=0/ADX 유한", np.isfinite(ax_.iloc[-1]) and atr_.iloc[-1] < 1e-9,
           f"adx={ax_.iloc[-1]:.4f} atr={atr_.iloc[-1]:.2e}")
    # 가격 갭 (OB 끊김 모사: 동일값 유지 후 점프)
    gap = pd.Series(np.concatenate([np.full(300, 100.0), np.full(300, 110.0)]))
    r_gap = rsi(gap, 14)
    report("갭 직후 RSI=100 (상승만)", abs(r_gap.iloc[310] - 100) < 1e-3, f"{r_gap.iloc[310]:.4f}")
    report("갭 한참 후 RSI 중립 복귀? (flat 수렴 거동 확인)", True, f"{r_gap.iloc[-1]:.4f} (정의 확인용)")
    # inf/극단 스캔 (실데이터 전체)
    lab = pd.read_parquet('/Users/mark/Desktop/Mark/mark19/research/i_labeling/labels.parquet')
    labels = [c for c in lab.columns if c not in ['yr', 'day', 'sec', 'min_of_day', 'mid']]
    X = lab[labels].to_numpy(float)
    n_inf = int(np.isinf(X).sum()); n_huge = int((np.abs(X) > 1e8).sum())
    report("실데이터 inf 0개", n_inf == 0, f"inf={n_inf}")
    report("실데이터 |x|>1e8 0개", n_huge == 0, f"huge={n_huge}")
    # RSI 극단값 실데이터 빈도 (flat 함정이 실제로 발생했나)
    n_rsi0 = int((lab['rsi_14'] < 1e-9).sum()); n_st = int(((lab['stoch_k'] < 1e-12) & (lab['range_bp'] < 1e-9)).sum())
    print(f"  [info] 실데이터 rsi_14==0 행: {n_rsi0}, flat-범위 stoch 0 행: {n_st} (flat 함정 실제 발생 빈도)")
    return res

# ════════ 작업3: OB/체결 라벨 수동 검산 (독립 경로) ════════
def part_manual():
    print("\n===== 작업3: OB/체결 라벨 수동 검산 =====")
    day = '2024-06-15'
    lab = pd.read_parquet(f'/Users/mark/Desktop/Mark/mark19/research/i_labeling/vizday_{day}.parquet')
    ob = pd.read_parquet(f'{OB}/{day}.parquet')
    ts = pd.to_datetime(ob['timestamp'], utc=True)
    keep = (ts.dt.date == pd.Timestamp(day).date()).values
    ob = ob[keep].reset_index(drop=True); ts = ts[keep].reset_index(drop=True)
    t0 = ts.iloc[0]
    sec_f = (ts - t0).dt.total_seconds().values   # float seconds
    tr = pd.read_parquet(f'{TR}/{day}.parquet')
    print(f"  trades dtype={tr.timestamp.dtype}, head={tr.timestamp.iloc[0]}, side vals={tr.side.unique()}")
    tts = pd.to_datetime(tr['timestamp'], unit='s', utc=True)
    tsec_f = (tts - t0).dt.total_seconds().values

    rng_ = np.random.default_rng(5)
    checks = rng_.choice(len(lab), 6, replace=False)
    ok_all = True
    for row_i in checks:
        row = lab.iloc[row_i]; e = int(row['sec'])
        # OB snapshot: round(ts-t0) <= e 인 마지막 row (파이프라인과 동일 정의, 독립 구현)
        snap_sec = np.round(sec_f).astype(int)
        j = np.where(snap_sec <= e)[0][-1]
        b5 = sum(ob.loc[j, f'bid_{i}_size'] for i in range(5))
        a5 = sum(ob.loc[j, f'ask_{i}_size'] for i in range(5))
        obi5_m = (b5 - a5) / (b5 + a5 + 1e-9)
        mid_m = (ob.loc[j, 'bid_0_price'] + ob.loc[j, 'ask_0_price']) / 2
        spr_m = (ob.loc[j, 'ask_0_price'] - ob.loc[j, 'bid_0_price']) / mid_m * 1e4
        # flow_1m: round(tsec) in (e-60, e] 인 체결 (파이프라인 binning 과 동일 정의)
        tb = np.round(tsec_f).astype(int)
        m = (tb > e - 60) & (tb <= e)
        bv = tr['size'].values[m & (tr['side'].values == 'Buy')].sum()
        sv = tr['size'].values[m & (tr['side'].values == 'Sell')].sum()
        flow_m = (bv - sv) / (bv + sv + 1e-9)
        d_obi = abs(obi5_m - row['obi5']); d_fl = abs(flow_m - row['flow_1m'])
        d_spr = abs(spr_m - row['spread_bp']); d_mid = abs(mid_m - row['mid'])
        ok = d_obi < 1e-9 and d_fl < 1e-9 and d_spr < 1e-9 and d_mid < 1e-9
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] min={int(row['min_of_day'])} obi5 {row['obi5']:+.4f}={obi5_m:+.4f} "
              f"flow1m {row['flow_1m']:+.4f}={flow_m:+.4f} spread {row['spread_bp']:.4f}={spr_m:.4f} mid Δ{d_mid:.2e}")
    # 부호 일관성: bid 무거우면 obi>0 (정의), Buy aggressor 많으면 flow>0
    j = checks[0]
    print(f"  [정의] obi = (bid-ask)/(bid+ask) → 매수벽 우세=+ ✓ / flow = (buyV-sellV)/(tot) → 매수 aggressor 우세=+ ✓")
    # rv_300 수동 1점
    row = lab.iloc[checks[0]]; e = int(row['sec'])
    snap_sec = np.round(sec_f).astype(int)
    g = np.full(e + 1, np.nan)
    mids = ((ob['bid_0_price'] + ob['ask_0_price']) / 2).values
    idx = ffill_idx(snap_sec[snap_sec <= e], e + 1)
    g = mids[idx]
    lr = np.diff(np.log(g[max(e-300,0):e+1]))
    rv_m = np.sqrt((lr**2).mean()) * 1e4
    d_rv = abs(rv_m - row['rv_300'])
    ok = d_rv < 1e-6
    ok_all &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] rv_300 수동 {rv_m:.6f} vs 라벨 {row['rv_300']:.6f} (Δ{d_rv:.2e})")
    # dobi 수동
    e2 = int(lab.iloc[checks[1]]['sec'])
    o5 = lambda jj: ((sum(ob.loc[jj, f'bid_{i}_size'] for i in range(5)) - sum(ob.loc[jj, f'ask_{i}_size'] for i in range(5))) /
                     (sum(ob.loc[jj, f'bid_{i}_size'] for i in range(5)) + sum(ob.loc[jj, f'ask_{i}_size'] for i in range(5)) + 1e-9))
    j_now = np.where(snap_sec <= e2)[0][-1]; j_30 = np.where(snap_sec <= e2 - 30)[0][-1]
    dobi_m = o5(j_now) - o5(j_30)
    d_d = abs(dobi_m - lab.iloc[checks[1]]['dobi5_30'])
    ok = d_d < 1e-9; ok_all &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] dobi5_30 수동 {dobi_m:+.5f} vs 라벨 {lab.iloc[checks[1]]['dobi5_30']:+.5f}")
    print(f"  => 수동 검산 {'전부 일치' if ok_all else '불일치 있음!'}")
    return ok_all

# ════════ 작업5: truncation invariance (lookahead 검출) ════════
def part_trunc():
    print("\n===== 작업5: truncation invariance (미래 잘라도 과거 라벨 동일?) =====")
    day = '2024-06-15'
    THR = 5.0   # 고정 큰체결 임계 (양쪽 동일 — bigflow binning 경로까지 causality 검증)
    full, _ = IL.process_day(day, big_thr=THR)
    T_min = 700                      # 분 700 (burn 후) 까지만 데이터 존재한다고 가정
    T_sec = (T_min + 1) * 60 - 1
    trunc, _ = IL.process_day(day, trunc_sec=T_sec, big_thr=THR)
    labels = [c for c in full.columns if c not in ['yr', 'day', 'sec', 'min_of_day', 'mid']]
    fa = full[full.min_of_day <= T_min].reset_index(drop=True)
    tb = trunc[trunc.min_of_day <= T_min].reset_index(drop=True)
    assert len(fa) == len(tb), (len(fa), len(tb))
    bad = []
    for c in labels:
        a, b = fa[c].values, tb[c].values
        m = ~(np.isnan(a) & np.isnan(b))
        d = np.abs(a[m] - b[m])
        mx = float(np.nanmax(d)) if len(d) else 0.0
        if not (mx < 1e-12):
            bad.append((c, mx, int((d > 1e-12).sum())))
    if bad:
        print("  ⚠️ LOOKAHEAD 검출 — 미래 truncation 에 과거 라벨이 변함:")
        for c, mx, cnt in bad:
            print(f"    {c}: max|Δ|={mx:.3e}, 변한 행 {cnt}/{len(fa)}")
    else:
        print(f"  PASS — 전 47 라벨, 분 0~{T_min} 구간 완전 동일 (미래 정보 0)")
    return bad

if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if part in ('lib', 'all'): part_lib()
    if part in ('synth', 'all'): part_synth()
    if part in ('manual', 'all'): part_manual()
    if part in ('trunc', 'all'): part_trunc()
