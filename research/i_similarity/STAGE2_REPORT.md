# [I] 유사도 거래 — 2단계: 축약+정규화+유사도 검색 검증 보고서

**날짜**: 2026-06-06 · **목적**: 유사도가 "방향"을 보는지 "시기/변동성"을 보는지만 검증. **거래/예측/edge 결론 X.**

## 입력
- 1단계 labels.parquet (203일 243k분행 47라벨). 정규화 burn-in(첫 5 sampled days) 제외 후 **198일 190,202행**.

## 작업1 — 중복 축약: 47 → 21차원
- 정규화 공간에서 spearman |r|>0.7 계층군집(average linkage) → 22군집, 군집별 medoid 대표.
- **PCA 참조: EVR 90%→17차원, 95%→22차원** — 1단계 "유효 ~10차원" 추정은 **과소**였음 (|r|>0.8 블록 4개만 보고 셌던 것; 실제 독립 정보 더 많음). 교정.
- 정보손실: 탈락 25개를 retained 로 ridge 복원 — **R² min 0.533 / med 0.862**, R²<0.5 없음.
- **spread_bp 제외** (→21차원): 1틱 양자화로 per-day IQR≈0 → scale 나눗셈 폭발(z_iqr 20~45) + 드리프트 잔존(2.26). 정보량 최소(스프레드≈상수 1틱). 필요시 별도 게이트로 재방문.
- 방향 정보: 부호군 대표 전부 존재 (추세 7, OBI/dOBI 4, flow 3, 캔들 1). ±10 winsorize(고정상수).
- 대표 21: ma_dev_{5,15,240}, ma_slope_{30,120,240}, rsi_30, adx_14, atr_14, body_ratio, upper/lower_wick, obi50, obi_wtd, dobi5_30, dobi5_60, rv_ratio, flow_30, flow_1m, vol_z, bigflow_norm

## 작업2 — 정규화 (causal, lookahead 구조적 불가)
- rolling 통계 = **과거 15 sampled days(달력 ~90일)의 per-day robust 통계만, 현재 day 제외(shift 1)** → t 시점 정규화에 t 이후 정보 0.
- **부호(방향) 라벨: scale-only** (rolling IQR 나눗셈만 — location 보존) → **부호 보존 100% 확인** (구조적).
- 크기-드리프트 라벨(rv/atr/boll_width/range/rv_ratio): full robust-z (vol-regime 수준 제거가 목적).
- 결과: 연도 IQR비 raw 최대 2.56 → **정규화 후 대부분 1.0~1.2** (잔존: rv_3600 1.41 — 1h rv 의 느린 regime 은 90일 창으로 부분 제거). 박스플롯에서 변동성군 2023→2025 우상향 드리프트 소멸 확인 (`norm_before_after_boxplots.png`).

## 작업3 — 정규화 검증: "닮은 과거"의 시기 분포 (핵심)
300쿼리(2024+, 연도 stratified), pool=쿼리보다 과거 day 전부(중앙 126k행), top-100, 5조건 대조:

| 조건 | recency ratio (1=시기중립) | same-yr lift (1=중립) |
|---|---|---|
| A raw47+global-z (naive) | 0.81 [0.54,1.09] | 1.19 [0.59,1.84] |
| B 축약만(21)+global-z | 0.85 | 1.12 |
| **C 축약+causal정규화** | **0.87** | **1.12** |
| C cosine | 0.88 | 1.16 |
| **C whitened**(2023 fit) | **0.94** | **1.07** |

- **정직 발견: naive 의 시기 쏠림이 우려보다 약했다** (lift 1.19, 3~5배 아님) — 47라벨 중 다수(rsi/stoch/obi/flow)가 이미 bounded/stationary 라서. 그래도 정규화+whitening 으로 **0.81→0.94 / 1.19→1.07 일관 개선**, 매치 연도 분포가 pool 구성과 거의 일치 (`match_year_dist.png` — 전 연도 커버, 빠진 연도 없음).
- 잔존 쏠림 p10 0.57: 일부 쿼리는 여전히 최근에 몰림 — vol clustering 등 **진짜 상태 재발도 시간상 뭉치므로** 완전 중립이 목표는 아님. 단 3단계에서 "쏠림 판정"은 시기분해 audit 필수.

## 작업4 — 유사도 검색: 진짜 닮은 걸 찾나
- **과거 90분 가격경로 corr** (쿼리 vs top-50, 미래 안 봄): C 중앙 **+0.30** [0.05,0.70] vs **random 0.00**, 쿼리별 top>random 85%. → 진짜 경로 모양이 닮은 걸 찾음 (`example_match_paths.png`).
  - naive A 가 더 높음(0.40) — 추세라벨 3~6배 중복 = 사실상 "경로모양 과가중"이었던 것. 축약 공간은 OB/flow 상태(가격경로에 안 보임)를 동등 가중 → 경로 corr 약간 낮은 게 정상 (경로 corr 은 닮음의 한 단면일 뿐).
- **방향(부호) 닮음 직접 검증**: top-100 매치가 쿼리와 부호 라벨(추세·OBI·flow·캔들 15개) 일치하는 비율 — **중앙 0.782 vs random 0.499** (전 차원 0.76~0.91, `sign_agreement.csv`). **유사도가 "방향 구성이 같은 상태"를 찾는다.** (p10≈0.5 는 피처 0 근처 쿼리 — 부호 불안정, 예상됨)
- **N 충분성**: pool 중앙 126k행. 거리 rank1/10/100 = 2.29/2.76/3.28 vs pool 중앙거리 6.73 — top-100 이 pool 평균보다 훨씬 가깝고 rank100 에서 거리 급증 없음 → **N=100+ 확보 여유**. (3단계 70% 판정에 충분)
- **거리 척도**: Euclid/cosine 비슷 (경로 corr 0.30/0.35), whitening 이 시기중립 최선(0.94/1.07). → 3단계는 **whitened Euclid** 기본 + cosine 교차확인 권장. Mahalanobis ≈ whitening (동일).

## 종합 — 2단계 통과 ✅ (유사도는 "방향"을 본다)
1. 축약 47→21 (정보손실 R² med 0.86, 방향군 대표 전존)
2. 정규화 causal (lookahead 구조적 불가) + 드리프트 제거 (IQR비 2.56→~1.1)
3. 시기 분포: naive 쏠림 약했지만 정규화로 추가 중립화 (whitened 0.94/1.07)
4. 검색 동작: 경로 닮음 +0.30 vs 0, **부호 일치 0.78 vs 0.50**, N 충분

### 3단계 전 caveats (흥분 금지)
- **간극천장 대조 필수**: G/H 에서 causal 방향 gross 0.37bp(fee 미만) 확정. 유사도 = 기존 라벨의 비선형 조건부 — 3단계에서 70% 쏠림이 나와도 fee·fill·시기 audit + 간극천장 대조 먼저.
- 부호일치 0.78 은 "상태가 닮음"이지 **"미래 방향이 같음"이 전혀 아님** (이번엔 미래 안 봄).
- 203일 subsample 검증 — 3단계 판정은 전체 1198일 전수 DB 필요할 수 있음.
- rv_3600 드리프트 잔존(1.41), 일부 쿼리 시기 쏠림(p10 0.57) → 3단계 결과는 시기분해로.

### 산출물
- 코드: `scripts/i_reduce_norm.py`, `scripts/i_simsearch.py`, `scripts/i_signcheck.py`
- 데이터: `labels_norm_reduced.parquet`(21차원), `reduce_norm_meta.json`
- 검증: `norm_drift_ratio.csv`, `norm_temporal_stats.csv`, `reduction_recon_r2.csv`, `simsearch_per_query.csv`, `sign_agreement.csv`
- 시각화: `norm_before_after_boxplots.png`, `match_year_dist.png`, `recency_pathcorr_box.png`, `example_match_paths.png`
