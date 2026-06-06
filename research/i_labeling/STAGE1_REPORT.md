# [I] 유사도 거래 — 1단계: 라벨링 검증 보고서

**날짜**: 2026-06-06 · **목적**: 라벨(지표)이 제대로 계산되고 분포가 정상인지만. **거래/예측/edge 결론 X.**

## 데이터
- ETH 50레벨 1Hz OB + trades_perp(aggressor side+size). **t≤0 causal only**, day-boundary wrap 제거.
- 시기 고르게 STEP=6 → **203일 (2023-01-18 ~ 2026-04-27), 243,121 분(分) 행, 47 라벨**.
- 라벨링 cadence = 매 1분(분 마지막 초까지의 정보). burn-in 240분(최장 SMA 정의).
- ⚠️ 이건 **검증용 시기-전체 표본**. stage2 유사도 DB는 전체 1198일 전수 계산 필요(이번 범위 아님).

## 작업1 — 계산한 라벨 (47개, 넓게)
| 그룹 | 라벨 |
|---|---|
| 추세/이격 | ma_dev_{5,15,30,60,120,240}, ma_slope_{동일}, boll_pos |
| 모멘텀/오실 | rsi_14, rsi_30, stoch_k, stoch_d, macd, macd_hist, adx_14, di_diff |
| 변동성 | rv_{60,300,900,1800,3600}, atr_14, boll_width, range_bp, rv_ratio |
| 오더북 | obi1, obi5, obi20, obi50, obi_wtd, dobi5_30, dobi5_60, spread_bp |
| 체결 | flow_30, flow_1m, flow_5m, bigflow_5m, bigflow_norm, vol_z |
| 캔들 | body_ratio, upper_wick, lower_wick |

## 작업2 — 분포/정상성 ✅ 정상
- **범위 위반 0**: RSI/Stoch/ADX ∈[0,100], OBI/flow/body ∈[-1,1], wick ∈[0,1] 전부 통과.
- **상수 라벨 0, NaN ≈ 0** (예외: `ma_slope_240` 20% — SMA240+shift240 이중요건. 다른 라벨 0%).
- 분포 합리적: RSI 중앙 50.3(p1~p99 23~77), spread 0.043bp(≈1틱, 가격 추종), OBI 깊을수록 좁아짐(obi1 ±0.95 → obi50 ±0.37), rv bp 단위 우상향.
- **시각화 4일(viz_*.png)**: MA가 가격 매끄럽게 추종, RSI 70/30 진동, MACD 히스토그램 zero-cross, ADX 추세구간 상승, OBI/flow ±1, rv 변동구간 스파이크 → **의도대로 찍힘 확인**.
- 산출물: `labels_distribution.csv`, `hist_grid.png`, `viz_{2023..2026}.png`

## 작업3 — 중복/독립 (Spearman) ⚠️ 강한 중복 → 축약 필요
47 라벨이 4개 상관블록 + 독립단독으로 (`corr_heatmap.png`, `redundant_pairs.csv`):
- **추세/모멘텀군** (서로 |r|0.9+): ma_dev·ma_slope·boll_pos·rsi·stoch·macd·di_diff — 사실상 "가격이 최근범위 어디 + 모멘텀" 한 정보.
- **변동성군**: boll_width·atr_14·rv_60~3600·range_bp (|r| 0.9~0.98). rv는 창만 다를 뿐 거의 동일.
- **OBI군**: obi1·obi5·obi20·obi_wtd (|r|0.94+), obi50/dobi 약하게.
- **flow군**: flow_5m·bigflow_5m·bigflow_norm (|r|0.9+).
- **독립 단독**: adx_14, spread_bp, body_ratio, upper/lower_wick, vol_z, rv_ratio, obi50, dobi5_60.
- → **유효 독립 차원 ≈ 10개**. 47 raw를 그대로 거리에 넣으면 추세·변동성이 3~6배 중복가중 → **유사도 왜곡**. stage2 전 대표 선택 또는 PCA/whitening 필수.
- 산출물: `independent_groups.json`(계층군집 26그룹, |corr|>0.8 묶음)

## 작업4 — 시기 안정성 ⚠️ 변동성-스케일 라벨 정규화 필요
연도 IQR비/중앙값비 + 박스플롯(`temporal_boxplots.png`, `temporal_stats.csv`):
- **드리프트 O (정규화 필요)** — *크기가 변동성에 비례하는 라벨*:
  - rv_60~3600 (2023 중앙 0.45 → 2025 0.85, **~2배**), atr_14, boll_width, range_bp, rv_ratio
  - ma_dev/ma_slope/macd 계열 (IQR비 1.7~1.8 — 고변동기에 이격 폭 확대)
  - spread_bp (2023 0.057 → 2024 0.033, 가격↑로 틱 비중↓), bigflow_5m(비정규화 raw 거래량)
- **시기 안정 (정규화 불필요)** — *이미 z/비율로 정규화된 라벨*:
  - **rsi, stoch, adx, obi(비율 -1..1), boll_pos(z), vol_z(z), flow(비율)** → 구조적 stationary.
- 교훈: **bp 단위라도 변동성-스케일 라벨은 vol-regime 드리프트를 물려받음.** stage2에서 이 군은 rolling z-score(또는 vol-정규화) 적용해야 거리 메트릭이 "연도"에 지배되지 않음.
- ⚠️ `med_ratio`는 0중심 라벨에서 ÷0 인공물 → IQR비가 신뢰 지표(정직).

## 종합 — 1단계 통과 ✅, 2단계(유사도) 전 처리할 것
- 라벨 계산/분포/범위 **정상**, 시각화 의도대로. lookahead 없음(t≤0), wrap 제거.
- **2단계 전제조건 2개** (이번 발견):
  1. **중복 축약**: 47→~10 독립차원 (대표선택 or PCA). 안 하면 추세·변동성 과중복.
  2. **시기 정규화**: 변동성-스케일군 rolling z-score. 안 하면 유사도가 "같은 연도"로 쏠림.
- 다음(2단계): 위 2개 적용한 라벨공간에서 거리/유사도 → 과거 유사시점 검색 검증.
