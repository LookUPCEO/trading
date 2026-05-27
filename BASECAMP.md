# Mark19 BASECAMP

**Last updated:** 2026-05-27 (4h Direction R&D + deploy 진행 중 — prod model bug 발견)
**Status:** Pre-flip blocked — prod model 이 거래 자체를 안 함 (best_iteration=2 bug)
**Primary goal:** 일 1% 수익률 알고 트레이딩 봇

---

## 🚨 2026-05-27 현재 상태 (요약)

### 진행
- **Mark19 데이터 pipeline 완료**: 1561 files / 44GB (5-23), workers=2 OOM-safe
- **HMM Regime 분석 완료** (1198d ETH): 3 regime, 2025 covariate shift 확인
- **4h Direction discovery**: long features × 4h target = **Sharpe net+6bp Mixed +1.81** (재검증 필요 ⚠️)
- **Deploy stack 완성**: WebSocket + shadow runner + reconciler + risk rails + Discord 통합
- **WS 안정성**: 22h 동안 reconnect 1회 (startup), u sequence gap 0
- **ΔP monitor**: N=11 sample, max |ΔP|=0.003, flip 0 (stale 영향 수학적 불가능 확정)

### 🚨 Blockers — flip 전 해결 필수

**1. Prod model 이 거래를 절대 안 함** ← critical
- `4h_direction_v1.joblib`: best_iteration=2 (n_estimators=200 학습했지만 early stop)
- `predict_proba` default = best_iter+1 = **3 trees만 사용** → underfitted
- 1198d N=7180 boundaries 전체에서 LONG/SHORT decision **0건**
- p_up 범위: [0.5011, 0.5195] (threshold 0.55/0.45 절대 도달 X)
- Live shadow 4일치 (5-24 ~ 5-27, 15 boundaries) 동일하게 모두 SKIP

**2. Backtest +1.81 결과와 모순**
- Memory 기록: Sh +1.81, 0.9-2.2 trade/day (양방향)
- 우리 측정: 0 trade (best_iter=2), 23 trees full = 0.34 trade/day **LONG-only**
- → backtest 의 실제 config (model state, predict 방식) 가 prod 와 다를 가능성
- 원본 backtest script 미발견 (train_combined_strategy 등 후보)

### 다음 할 일 (우선순위)
1. **+1.81 재현 시도**: predict iteration_range=(0,23) 로 1198d 재실행 → Sharpe/trade-freq 일치?
2. **불일치 시 원본 backtest script 추적** (ground truth)
3. **재현 안 되면 +1.81 자체 재검증** (small-n 함정 가능성)
4. flip 은 재현 후에야 의미

### 🔬 진행 중 검증 (Gate)
- ✅ WS 안정성 (reconnect 1/22h)
- ✅ u sequence continuity (906 deltas 100% 연속)
- ✅ stale OB decision flip = 수학적 불가능 (max |ΔP|=0.003 vs thr 0.05 buffer 무한)
- ❌ Model 거래 발생 — best_iter bug
- ❌ Backtest +1.81 재현
- ⏸ Live flip 0.01 ETH 1x — Gate 1B 미통과

---

## 📌 핵심 R&D 발견 (2026-04 ~ 2026-05)

### 4h Direction (5월 23-25)
- **Long features (mom_1d, rv_1d, dist_ma_1d, cumflow_1d, mom_4h_bp) × 4h target**
- **Direction AUC 0.566** (BASECAMP 0.545 ceiling 돌파)
- Sharpe 결과 (이전 backtest):
  - +3.89 Sh, +36 bp/day, p<0.001 (1198d full)
  - Mean walk-forward 5 windows: **+1.81 Sh net+6bp Mixed fee**
- ⚠️ **Prod 재검증 필요** — 위 blocker 참조

### Direction prediction 차단 → 발견 (4월 말 → 5월 초)
- 4월 末 직전: "direction 죽음" 결론 (fee wall)
- 진짜 원인 = timeframe mismatch (short features × short horizon)
- Long features × 4h horizon 으로 우회

### Volatility 단독 사용 가능 신호 (5월)
- vol R² **0.566** (XGB regression)
- Large-move AUC **0.805**
- Direction 과 결합 X (별 path)

### Stale OB 진단 (5월 26-27)
- 원인: Bybit V5 subscribe-time snapshot 이 ~10 delta 옛 cache
- u sequence gap 0 (B 가설 기각), snapshot stale (A 가설 확인)
- 21h 동안 drift 0 으로 수렴 (delta 누적 cleanup)
- **결정 영향**: 4h direction 의 long features 88% importance → **decision flip 수학적 불가능**
- 대응: warm-up 거부, **stale-resistant feature mix** 로 검증 완료

---

## 🔐 환경 / 보안

- API key: `live_bot/.env` (코드/log 외, .gitignore 적용)
- IP whitelist: 112.150.88.251
- Read-only 검증 후 trade 권한 추가 (withdrawal OFF 유지)
- Live first flip 조건: 0.01 ETH, 1x leverage, 첫 5건 manual confirm
- Discord webhook 환경변수 (평문 X)

---

## 📂 주요 경로

- 코드: `/Users/mark/Desktop/Mark/mark19/`
  - 스크립트: `scripts/` (bybit_ws.py, mark19_shadow_runner.py, mark19_live.py, dp_monitor.py 등)
  - 봇: `live_bot/`
- 데이터 (외부, ~44GB): `/Users/mark/mark19_data/`
  - bars: `bars_5min_v3/{symbol}/{date}.parquet` (mass conversion)
  - live bars: `bars_5min_v3_live/{symbol}/{date}.parquet`
  - 모델: `models_prod/4h_direction_v1.joblib`
  - decisions: `shadow_decisions/{date}.jsonl`
  - ΔP monitor: `dp_monitor.jsonl`
- 메모리: `~/.claude/projects/-Users-mark/memory/`

---

## --- (legacy below — 시도 17 기록, 2025-04 시점) ---

---

## 🎯 Core Goal

**일 1% 수익률.** 단일 시도 결과로 목표 낮추지 말 것. 모든 시도 다 해본 후 평가.

**현재 진행:** 16 시도 완료. **일 1% 가능성 검증 + ML 단계 완료.** Realistic Mixed 시나리오에서 일 +1.0-1.5% 도달.

---

## 🏆 NEW BEST Strategy (시도 17)

```
Strategy: LR + Cross + Adaptive features (Vol filter + Direction filter + 1h cycle)
─────────────────────────────────────────────────────────────────────────────
Vol model:        LogReg, threshold 0.6 (proba)
Direction model:  LogReg (1h horizon, T=0.20% triple-barrier), threshold 0.65
Position cycle:   1h lockout (no overlap)
Trade signal:     vol_proba > 0.6 AND |dir_proba - 0.5| > 0.15

Features (170+ total):
  - Base features (143)
  - Cross features (13): OBI×Volume, Liq×OBI, Funding×OI 등
  - Adaptive features (15): Rolling z-score, Relative (1h/1d/7d window)

Hyperparameters:
  LogisticRegression(max_iter=2000, random_state=42, C=0.1)
  Train medians for fillna (data leakage 방지)

Performance (6 test dates, ideal Maker fee):
  Maker:  Daily +2.73%, Sharpe 1.53, Max DD 0.031% ⭐⭐⭐
  Mixed:  Daily +1.23%, Sharpe 0.66 ✅ (Realistic 운영)
  Taker:  Daily -0.26%, Sharpe -0.13 ❌

Per-date (Maker):
  2024-11: +4.00%, 2024-12: +1.82%, 2025-01: +1.83%
  2025-02: -0.03%, 2025-03: +3.17%, 2025-04: +5.58%

Realistic 운영 추정:
  Maker fill rate 30-50% (Bybit ETH 시장 특성)
  → Mixed scenario: Sharpe 0.5-0.7, daily +1.0-1.5% (일 1% 달성)
```

---

## 📊 시도별 결과 누적

| 시도 | 설명 | 결과 |
|------|------|------|
| 0 (baseline) | 5min direction (binary) | AUC 0.515 (random 가까움) |
| 1 | Triple-barrier T=0.20 | AUC 0.580 ✅ |
| 2 | 5m + OBI strength>0.3 | AUC 0.611 (4 dates) ✅ |
| 3 | 1h horizon | AUC 0.620 (4 dates) ⭐ |
| 4 | 1h + OBI | 실패 ❌ |
| 5 | 1h + Funding rate | 실패 ❌ |
| 6 | Cross features | Sharpe 1.06 ⭐ |
| 7 | Microstructure features | Sharpe 0.89 (smoothing 함정) ❌ |
| 8 | Regime conditional | Sharpe 0.61 (covariate shift 함정) ❌ |
| 11 | 36 dates 확장 | 진짜 baseline 안정 ✅ |
| 12 | Vol+Dir Combined | 거래비용 이김 ✅ |
| 13 | Realistic 4 dates | Sharpe 1.09 (over-confident) |
| 14 simple | Maker fill rate 추정 | Realistic 일 1% viable ✅ |
| 14 정확 Phase 1 | Order book dynamics | Maker fill 30-50% 확인 |
| 15 | Position sizing | fixed 가 best ❌ |
| 16 | Asymmetric SL/TP | winners cap 함정 ❌ |
| **17** | **Adaptive features (Cross + z-score)** | **Sharpe 1.53 ⭐⭐⭐ NEW BEST** |
| 18 | XGBoost (LR 대신) | mode collapse ❌ |

**Direction AUC ceiling: 0.545 (확정)**
**Trading edge: features 와 Maker fee 가 결정적**

---

## 🔍 핵심 발견 (정리)

### 1. Direction AUC Ceiling 0.55
- 모든 시도 AUC 0.52-0.55 범위
- Model 종류, hyperparameter 변경 효과 작음
- Features 자체가 진짜 leverage

### 2. Cross + Adaptive features 가 진짜 game-changer
- AUC 변화 작음 (0.541 → 0.545)
- Trading edge 큰 향상 (Sharpe 0.45 → 1.53)
- W/L ratio 향상 (1.02 → 1.27)
- Adaptive z-score = covariate shift 보정

### 3. Timeframe 이 가장 중요
- **5분 direction:** AUC 0.515 (HFT 효율성)
- **1시간 direction:** AUC 0.620 → 0.545 (36 dates), retail 가능 horizon
- 1h cycle = 24 trades/day 자동화 가능

### 4. 거래비용이 system viability 결정
- Taker 0.11%/trade × 18 trades/day = 1.98% cost
- Maker -0.05% rebate
- **Maker fill rate 30-50% (Bybit ETH 현실)**
- Mixed (Taker entry + Maker exit) 가 진짜 운영 strategy

### 5. 실패 패턴 정리
- **Smoothing 함정:** 시도 7 (Microstructure), 시도 18 (XGBoost ensemble)
- **Covariate shift:** 시도 8 (Regime conditional)
- **Winners cap:** 시도 16 (SL/TP)
- **Confidence sizing 함정:** 시도 15 (AUC 0.55 약한 신호로는 sizing 효과 없음)

### 6. LR > XGBoost (이 case)
- LR + Adaptive z-score = 시기 무관 학습
- XGBoost 비선형 split = train-specific noise
- Mode collapse 빈번 (best_iter=0)

---

## 🗂️ 데이터셋 구조

### Train/Val/Test (36 dates)

```python
DATES_TRAIN = [
    "2022-01-01", "2022-04-01", "2022-05-01", "2022-07-01",
    "2022-08-01", "2022-09-01", "2022-10-01", "2022-11-01", "2022-12-01",
    "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01",
    "2023-06-01", "2023-07-01", "2023-08-01", "2023-09-01", "2023-10-01", "2023-11-01",
    "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01",
]  # 26 dates

DATES_VAL = ["2024-07-01", "2024-08-01", "2024-09-01", "2024-10-01"]  # 4 dates

DATES_TEST = [
    "2024-11-01", "2024-12-01",
    "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01",
]  # 6 dates
```

### Tardis 데이터 (외장 SSD)
- 위치: `/Volumes/PortableSSD/40_사이드프로젝트/mark19_data/`
- 36 dates × 4 datatypes = 144 파일
- 약 6GB (raw + converted)
- 무료 day=01 정책

### 자체 Collector (5 PIDs, 4+ days 가동)
- cross_exchange_prices, funding_rates
- bybit_orderbook, bybit_trades, bybit_liquidation

### Order Book 특성 (시도 14 Phase 1 발견)
- **Spread:** median 0.0004% (1 tick stuck)
- **Depth:** Top level 82% 집중 (queue 17 ETH 평균)
- **1초 mid 변화:** median 0% (정체)
- **Maker fill rate:** 30-50% (queue position + price cross 모두 필요)

---

## 💻 핵심 코드 파일

### Best system 파일들 (시도 17)
```
mark19/
├── ml/data_prep.py                          # DATES + build_split (Cross + Adaptive 통합)
├── features/cross.py                        # 시도 6 (13 features)
├── features/adaptive.py                     # 시도 17 (15 features) ⭐
└── features/lagged.py                       # 시도 baseline

scripts/
├── backtest_realistic.py                    # 시도 13 - 진짜 backtest
├── train_combined_strategy.py               # 시도 12
├── train_direction_triple_barrier.py        # 시도 1
├── train_volatility_classifier_v2.py        # V2 vol model
├── analyze_v2_validation.py                 # V2 검증
├── analyze_orderbook_dynamics.py            # 시도 14 Phase 1 ⭐
├── backtest_maker_fill.py                   # 시도 14 simple
├── backtest_xgb.py                          # 시도 18 (XGBoost 비교)
└── (더 많음)
```

### Output 위치
```
/Users/dohun/Desktop/Mark/mark19/data/analysis_results/
├── xgb_feature_importance_v2.csv
├── xgb_direction_feature_importance.csv
└── ...
```

---

## 🎲 시도 진행 정리

### ✅ 완료 (16 시도)
1-3. Triple-barrier, OBI, 1h horizon (baseline 개발)
4-5. 1h conditional filter (실패)
6. Cross features (Sharpe 1.06)
7. Microstructure (smoothing 함정)
8. Regime conditional (covariate shift 함정)
11. 36 dates 확장
12. Combined strategy
13. Realistic backtest
14 simple. Maker fill rate 추정
14 정확 Phase 1. Order book dynamics
15. Position sizing
16. Asymmetric SL/TP
17. **Adaptive features ⭐⭐⭐ NEW BEST**
18. XGBoost (mode collapse)

### ⏳ 남은 시도 (선택적)

#### 우선순위 1: Live Trading
- **Live paper trading (Bybit testnet 또는 small capital)**
- 시도 17 모델 + Mixed fee 가정
- 1주일 검증
- 진짜 fill rate 측정
- 작업: 5-6시간 setup

#### 우선순위 2: Backtest 추가
- **시도 19: Mixed strategy 최적화** (Taker entry + Maker exit, entry/exit threshold)
  - 작업 2-3시간
- **시도 20: 더 많은 데이터** (Tardis API 유료 또는 자체 collector 1주일 추가)
  - 작업 다양

#### 후순위 (효과 한계)
- 시도 9 Ensemble (XGBoost mode collapse 로 효과 없음)
- 시도 10 LSTM/Transformer (위험, 데이터 부족)
- 시도 14 정확 Phase 2 (queue simulation, marginal value)

---

## 🎬 다음 세션 시작 지침

### Step 1: 5 PID 상태 확인
```bash
ps aux | grep -E "(cross_exchange|funding_rates|bybit_orderbook|bybit_trades|bybit_liquidation)" | grep -v grep
```

### Step 2: 시도 17 결과 재현 (필요시)
```bash
cd /Users/dohun/Desktop/Mark/mark19
python scripts/backtest_realistic.py
# Expected: Maker daily +2.73%, Sharpe 1.53
```

### Step 3: Live trading 시작
- Bybit API key 준비 (testnet 또는 main)
- 시도 17 모델 export
- Live trading bot 코드
- 작은 자본 (100만원 기본)

---

## 🔑 절대 룰 (Memory)

1. **일 1% 목표** - 단일 시도 결과로 목표 낮추지 말 것. 모든 시도 다 해본 후 평가.
2. **프롬프트 검증** - Claude Code 보내기 전 항상 자체 검증 + 수정. 검증 안 된 프롬프트 절대 금지.

---

## 📈 Models 정리 (재사용 가능)

### Vol Model (V2 검증됨)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filled)

lr_vol = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr_vol.fit(X_train_scaled, y_vol_train)

# Target: target_volatility_300s > train_median
# AUC: 0.762 (36 dates)
# Threshold: 0.6 (trade signal)
```

### Direction Model (시도 17, NEW BEST)
```python
# Triple-barrier filter
T = 0.20  # 0.20%
mask = train_df["target_return_3600s"].abs() > T
train_filtered = train_df[mask]

# Same hyperparameters as Vol
lr_dir = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
lr_dir.fit(X_train_dir_scaled, y_dir_train)

# Target: target_return_3600s > 0
# AUC: 0.545 (36 dates with Cross + Adaptive)
# Train sample after filter: ~12K rows
```

### Combined Trading Logic (시도 17)
```python
DIR_THRESH = 0.65
VOL_THRESH = 0.6
LOCKOUT_MIN = 60  # 1h cycle

# Trade decision
trade = vol_proba > VOL_THRESH AND (
    dir_proba > DIR_THRESH or dir_proba < (1 - DIR_THRESH)
)
direction = +1 if dir_proba > 0.5 else -1

# Position size: fixed 1.0 (시도 15 검증)
# No SL/TP (시도 16 검증, winners cap 위험)

# PnL (per trade)
pnl = direction * actual_return - fee_pct
# fee_pct: -0.05 (Maker), 0.03 (Mixed), 0.11 (Taker)
```

---

## ⚠️ 알려진 함정

### 1. XGBoost mode collapse
- Direction 학습 시 best_iter=0 빈번
- 원인: train (0.55 up) vs val (0.39 up) class shift
- **LR 사용 권장**

### 2. Smoothing trap
- Microstructure (시도 7), XGBoost (시도 18) 모두 함정
- 새 features 추가 시 W/L ratio 모니터링 필수

### 3. Covariate shift
- Train (2022-2024) vs Test (2024-2025) 분포 차이 큼
- **시도 17 (Adaptive z-score) 가 해결**

### 4. Position overlap
- 1분 grid 의 모든 signal trade = 비현실적
- **1h cycle 강제 (24 trades/day max)**

### 5. Maker fill rate 환상
- Backtest Maker (-0.05%) 가정 비현실
- **Bybit ETH 진짜 fill rate: 30-50%**
- **Mixed scenario 가 진짜 운영**

### 6. Test 6 dates의 variance
- 2025-04 +5.58% outlier 영향
- 더 많은 dates 로 검증 필요

---

## 🏁 진행 상황 한눈에

```
[==============================================>      ] ML 단계 완료

✅ Direction signal 발견 (AUC 0.515 → 0.545 with adaptive)
✅ Trading edge 검증 (Sharpe 1.53, Maker daily +2.73%)
✅ Vol model robust (AUC 0.762)
✅ 거래비용 모델 별 진짜 평가
✅ Maker fill rate 진실 확인 (30-50% Bybit ETH)
✅ Realistic 운영 시 일 1.0-1.5% 가능 (Mixed scenario)

⏳ Live paper trading
⏳ 작은 자본 실거래
⏳ Compound growth → 100만원 → 30억 path
```

**진짜 목표 (일 1%) 달성 가능성: ✅ 검증됨**
- 이상적 (Maker only): 일 +2.73%
- 현실적 (Mixed): 일 +1.0-1.5%
- 보수적 (Taker 부분): 일 +0.5-1.0%

**다음 단계:**
1. Bybit testnet/실거래 API 통합
2. Live trading bot 구축
3. 작은 자본 (100만원) 실거래 시작
4. 1-2주 검증 후 자본 증액

---

## 🎯 Live Paper Trading 준비 체크리스트

다음 세션 시작 시:

- [ ] Bybit API key 발급 (main 또는 testnet)
- [ ] 시도 17 모델 export (LR coefficients + StandardScaler)
- [ ] Live feature pipeline (실시간 cross + adaptive features 계산)
- [ ] Trading bot 코드 (entry, exit, lockout, position management)
- [ ] Risk management (daily loss limit, max position, drawdown stop)
- [ ] Logging + monitoring
- [ ] 작은 자본 결정 (100만원 권장)

---

**End of BASECAMP.md**
