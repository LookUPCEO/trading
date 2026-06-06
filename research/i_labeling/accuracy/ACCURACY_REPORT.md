# [I] 1단계 보강 — 라벨 정확성 깊이 검증 보고서

**날짜**: 2026-06-06 · **목적**: 47라벨이 진짜 정확히 계산됐는지 (망가짐 검사 ≠ 정확성). **유사도/거래 결론 X.**
**코드**: `scripts/i_acc_verify.py` (part: lib/synth/manual/trunc)

## 작업1 — 라이브러리 대조 (TA-Lib 0.6.8 + pandas-ta 0.4.71b, 6일 × burn-in 후 전 분)
| 라벨 | vs TA-Lib | vs pandas-ta | 판정 |
|---|---|---|---|
| SMA(6기간)·boll_mid | **0 (완전일치)** | 0 | ✅ |
| stoch_k/d | **0 (완전일치)** | 0 | ✅ |
| ATR | 3e-8 | 3e-8 | ✅ (EWM 시드 수렴) |
| MACD/hist (raw) | 8e-8 | 8e-8 | ✅ |
| RSI_14 | 4e-6 | 4e-6 | ✅ |
| ADX_14 | 2e-5 | 2e-5 | ✅ |
| RSI_30 | max 0.065 (bar240–300), bar500후 1e-10 | 동일 | ✅ 순수 시드 수렴 잔차 (스케일 0–100 대비 무시) |
| di_diff | **4e-6 일치** | 117 (!) | ✅ — pandas_ta 의 DMP/DMN 자체 정의가 표준(Wilder DI)과 다름. 우리+TA-Lib 2:1 표준 |
| boll_sd | 비율 **정확히 √(20/19)=1.025978 상수** | 동일 | 🐛→✅ ddof=1→0 수정 (아래) |

- EWM 계열(RSI/MACD/ATR/ADX)의 미세 차이 = TA-Lib 은 SMA 시드, 우리는 EWM-from-start — **지수 수렴으로 burn-in 240분 후 사실상 동일** (정의 차이 명시).

## 작업2 — 합성 입력 이론 행동: **12/12 PASS**
단조상승 RSI→100.000000 / 단조하락 RSI→0.000000 / 횡보 RSI 56 / Stoch→100·0 /
변동성 급증→밴드폭·ATR 확대(×~60) / MACD 상승+·하락− / 추세 ADX 100·횡보 7 / 상승 di_diff>0.

## 작업3 — OB/체결 라벨 수동 검산 (라이브러리 없음 → 원시 parquet 독립 경로 손계산)
- obi5 / flow_1m / spread_bp / mid / rv_300 / dobi5_30 — 무작위 6+2시점 **전부 일치** (Δ < 1e-9).
- 부호 일관: obi=(bid−ask)/(bid+ask) → 매수벽 우세=+ ✓, flow=(buyV−sellV)/tot → 매수 aggressor 우세=+ ✓.
- trades 타임스탬프 = epoch float sec ✓ (unit='s' 정확), side ∈ {Buy, Sell} ✓.

## 작업4 — 엣지 케이스
- 완전 flat: RSI **0 반환 (이론 50)** 🐛, Stoch hh==ll **0 반환 (이론 50)** 🐛 → 수정 (아래). boll_pos/ADX/ATR 유한 ✅.
- 실데이터 전체 스캔: **inf 0, |x|>1e8 0** ✅. RSI==0 실발생 0행 (잠재 버그였음), 진짜 flat 14분 창 14행 (수정 후 50).
- 무체결 구간: flow=0(중립) ✅. OB 갭: ffill→lr=0 (rv 미세 과소 — 문서화).
- 1단계의 NaN ≈0 (ma_slope_240 20% 제외) 유지.

## 작업5 — lookahead 라벨 단위 (truncation invariance 테스트 — 신규 검출기)
**방법**: 같은 날을 "분 700 이후 데이터 없음"으로 잘라 재계산 → 분 0~700 라벨이 전부 동일해야 (다르면 그 라벨은 미래정보 사용). centered window 류 실수를 코드리뷰가 아니라 **실증으로** 잡음.
- **검출 1건**: `bigflow_5m`/`bigflow_norm` — 큰체결 임계 = **당일 전체 q95** → 일중 lookahead (truncation 시 426/461행 변동). **6번째 함정 family 등록감.**
- 수정 후: **전 47라벨 truncation 완전 동일 (미래 정보 0)** ✅. rolling/ewm 전부 backward ✓, centered 없음 ✓.
- 라벨 시각 정의 명시: 분 라벨 = "그 분 마지막 초(e)의 끝" 시점. OB snapshot ≤ e, trades round-binning ≤ e+0.5초 → **진입이 다음 분(≥ e+1s)이면 causal** (3단계에서 진입시점 ≥ 라벨시각+0.5s 보장 필요 — 명시).

## 수정 내역 (3건 — 덮지 않고 수정)
1. 🐛→✅ **bigflow lookahead**: 당일 q95 → **이전 처리일 q95 전달** (causal). 첫 처리일은 NaN (1일 손실).
2. 🐛→✅ **boll std ddof=1 → 0** (모집단): TA-Lib/pandas-ta/차트 표준 정렬. boll_pos/width ×1.026 상수배 변화.
3. 🐛→✅ **RSI·Stoch flat → 중립 50** (구버전 0 = '바닥' 오신호). RSI 식 100·au/(au+ad) 동치 변환. Stoch flat=50 은 TA-Lib(0)과 의도적 차이 — 정의 명시.

## 수정 후 재생성/재검증 (수정이 상위 결론을 바꾸나?)
- 라벨 203일 재생성(42s) → 작업1~5 전부 재PASS (boll_sd 1e-8, truncation 47/47 동일).
- **2단계 재실행** (축약/정규화/검색): 대표 동일(22→21), 복원 R² 0.518/0.860, recency whitened 0.92/lift 1.04, 경로 corr +0.29 vs 0.00, **부호일치 0.783 vs 0.497**, N 충분 동일 → **2단계 결론 전부 유지** (변화는 노이즈 범위).

## 종합 — 1단계 보강 ✅ (정확성 확정)
- 표준지표: 라이브러리 수치 일치(완전 또는 수렴잔차). 자체지표: 수동 검산 일치 + 부호 일관.
- 이론 행동 12/12, 엣지 처리(유한성) 확인, lookahead 실증 0 (truncation 테스트 상시 재사용 가능).
- 남은 정의 차이 (버그 아님, 명시): mid-기반 OHLC(체결가 아님), EWM-from-start 시드, Stoch flat=50, range_bp 분모=open, rv 는 OB갭 ffill 시 과소, RSI_30 burn 직후 ≤0.065 잔차.
