# Regime Detector 설계 메모 (Phase 2)

**작성일:** 2026-05-22
**목적:** Microstructure regime detection → Direction trading OOD 문제 해결

---

## 1. Regime metric 후보

### 1A. 기존 OOD 차원 (BASECAMP 시도 8 covariate shift 경험 기반)
| Metric | 정의 | 왜 중요 |
|---|---|---|
| `mean_spread_bp` | spread / mid_price × 10000 (bp) | Microstructure liquidity proxy. tight ↔ wide regime 전환 |
| `mean_depth_top5` | sum(bid_0..4_size + ask_0..4_size) | Order book thickness. thin ↔ thick |
| `trades_per_sec` | 1초당 trade count | Activity. quiet ↔ active |

### 1B. 추가 후보 (event-level raw delta 에서 추출 가능)
| Metric | 정의 | 추가 이유 |
|---|---|---|
| `order_flow_imbalance` (OFI) | sum(Δbid_size - Δask_size) over window | Lo & Sadka 류 microstructure signal. 방향성 + 강도 |
| `bid_ask_imbalance` | (bid_depth - ask_depth) / (bid_depth + ask_depth) | Static OBI. 압력 |
| `large_trade_ratio` | trades > P95 size / total trades | Whale activity proxy (raw delta 없으면 size events 로 근사) |
| `depth_slope` | d(cumulative size) / d(distance from mid) | Order book shape (steep ↔ flat) |
| `mid_realized_vol_1m` | 1분 mid log-return std | Volatility regime (direction filter 와 별개로 microstructure 측) |
| `update_rate` | events / sec | Activity 의 dual proxy |

### 1C. Feature 정규화
- 매 day-level 통계 → daily aggregate (mean, p50, p95)
- Z-score 사용: train period mean/std 로 normalize (시도 17 의 adaptive z-score 패턴 따름)

---

## 2. Regime classifier 비교

| Method | 라이브러리 | 장점 | 단점 | 권장도 |
|---|---|---|---|---|
| **HMM** | `hmmlearn` (Gaussian HMM) | State transition 자연스러움, 시간적 dynamics, posterior probability | Hyperparameter (n_states) 결정 어려움, EM local optima | ⭐⭐⭐ |
| **GMM** | `sklearn.mixture` | Soft clustering, BIC 로 n_components 결정 | 시간 dependency 무시 | ⭐⭐ |
| **K-means** | `sklearn.cluster` | 빠름, 해석 쉬움 | Hard cluster, no confidence | ⭐ (baseline) |
| **Bayesian Gaussian Mixture** | `sklearn` | n_components 자동 | 느림 | ⭐⭐ |
| **Change Point Detection** | `ruptures` | 전환 시점 명시 | Online 어렵고 regime label 별도 | ⭐ |

### 권장: HMM (primary) + GMM (sanity check) + K-means (baseline)

**Rationale:**
- 시도 8 (regime conditional) 실패 = train regime distribution 과 test regime distribution 다름
- HMM 의 posterior 가 OOD detection 도 제공 (low max posterior → uncertain → trade size down)
- GMM 으로 regime 의 stationarity 검증 가능

---

## 3. Walk-forward 설계

### Data split (ETH 1198 days)
```
Train:   2023-01-18 ~ 2024-12-31  (713 days, ~60%)
Val:     2025-01-01 ~ 2025-09-30  (273 days, ~23%)
Test:    2025-10-01 ~ 2026-04-30  (212 days, ~17%)
```

### Walk-forward iterations
- Iter 1: Train [day 1..713], Val [714..923], Test [924..1198]
- Iter 2: Re-train every 30 days (rolling), capture regime drift

### Cross-symbol validation
- ETH 로 학습한 regime detector → BTC/SOL (181 days each) 에 apply
- Same regime concept 인지 검증 (microstructure regime 은 symbol-invariant 가정)

---

## 4. Regime 개수 결정

### Approach A: BIC scan (GMM)
```python
for n in range(2, 10):
    gmm = GaussianMixture(n).fit(X_train)
    bic[n] = gmm.bic(X_val)
# n* = argmin(bic) — 보통 3-5 regime
```

### Approach B: 사전 정의 (해석성 우선)
미리 정의한 4 regime:
1. **Quiet** (low spread, low activity, low vol)
2. **Active liquid** (low spread, high depth, high activity)
3. **Choppy** (medium spread, high activity, high vol)
4. **Stressed** (wide spread, thin depth, high vol)

→ Approach A 로 자동 발견 후 Approach B 와 매칭 (해석 부여)

---

## 5. Multi-regime ensemble 구조

### Dispatch
```
Input: features at time t
↓
Regime detector → regime probability p_k (k=1..K)
↓
Per-regime models: m_k 가 각 regime 의 direction prediction
↓
Ensemble output: Σ p_k × m_k(features)  OR  m_argmax(features)
```

### Per-regime model 후보
- 시도 17 (LR + Cross + Adaptive) per regime
- 또는 단일 model 에 regime one-hot 추가 (gradient boosting)

### Transition handling
- Regime switch 발생 → 1-2 분 transition zone → 거래 차단 또는 size down
- HMM posterior < 0.7 (uncertain) → 거래 차단

---

## 6. 검증 metric

1. **In-sample**: Train regime distribution → BIC, log-likelihood, regime persistence (state duration mean)
2. **Out-of-sample**: Test regime distribution vs Train → KL divergence
3. **Trading performance**: Sharpe (overall) + Sharpe per regime
4. **OOD detection**: regime posterior entropy vs daily PnL — correlation 음수 (high entropy = low PnL day) 면 성공

---

## 7. 구현 우선순위

| Step | 작업 | 시간 | Output |
|---|---|---|---|
| 1 | Feature extractor (1Hz mark19 schema → daily aggregate) | 1-2h | `regime_features.parquet` (1 row/day) |
| 2 | Regime classifier 학습 (HMM + GMM + K-means) | 2-3h | `regime_labels.parquet` (1 row/day, labels) |
| 3 | Regime 분포 분석 + 시각화 | 1h | timeline plot, transition matrix |
| 4 | Cross-symbol 검증 (BTC/SOL) | 1h | regime label 일관성 보고 |
| 5 | (Phase 4) Per-regime direction model | 별도 |  |

---

## 8. 이슈/주의

- **Stationarity**: ETH 3년 데이터 = 다양한 phase 포함. Regime label 은 stationary 해야 하지만 microstructure 자체가 변화 (Bybit fee 변경, 시장 maturity 증가 등). → Re-fit periodic.
- **Survivorship bias**: 신규 데이터 = 살아남은 Bybit Linear Perpetual. Delisted symbol 영향 없음 (ETH/BTC/SOL 대형).
- **Look-ahead 방지**: regime label 은 t-시점까지의 information 만 사용해 추정. HMM 의 경우 filtered estimate (Viterbi forward) 사용, smoothed (forward-backward) 는 backtest 에서만.

---

## 9. 다음 작업

Phase 1 Track 1 (변환 테스트) 결과 → Option A/B/C 결정 → Feature extractor 구현 시작.
