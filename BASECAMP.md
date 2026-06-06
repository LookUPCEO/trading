# Mark19 BASECAMP

**Last updated:** 2026-06-07 (I.3 70% 쏠림 ✅ — 독립표 처리 후에도 null 대비 ×10~40, 단 감쇠+저빈도)
**Status:** 🔬 [I] 유사도 기반 거래 단계검증 진행 중 (1·1+·2·2+·3 통과, 다음 4단계 폭>fee = 최종 관문)
**Primary goal:** 일 1% 수익률 알고 트레이딩 봇

---

## 🔬 2026-06-07 — I.3 70% 방향쏠림 ✅: 라벨공간에 지역 조건부 방향정보 존재

- 81,682 쿼리(2024+ 10분격자) × 독립일 매치 100 (day-dedupe — **naive top100 은 고유일 84개뿐**, 클러스터링 안 하면 표 16~32% 가짜) + causal prefix + null random 매치 대조.
- **thr70 쏠림 0.27~0.41% vs null 0.00~0.04% (×10~40)**. binomial(causal base) 전건 p<0.01. real vote 분포 fat-tail.
- OOS(룰 사전지정·고정): train 0.33~0.51% → test 0.13~0.28% — 유지하되 ~2배 감쇠. 분기 전체 양수, 2024Q1 0.88%→2026 ~0.2%.
- 건수: **thr70 일평균 0.30 에피소드** (3일 1건), thr65 1.69건/일. 4h up편중 144/45 (drift 유산).
- ⚠️ 미측정(=4단계): 쏠림 방향의 실제 hit rate, 폭, fee/fill. **간극천장 0.37bp 정면 대조가 4단계 승부.**
  판정은 최근 시기 기준 (감쇠). thr65 후보군 병행.

---

## 🔬 2026-06-06 (확장) — I.2+ 1198일 전수: DB 1.135M행, 전체에서도 "방향 구성"

- 1,436,626 분행 라벨링 (검증 코드 그대로). truncation 추가 시기 47/47 동일 (lookahead 0). 전수 inf/위반 0.
- 일관성: 겹치는 203일 45/45 라벨 1e-12 동일. bigflow 만 causal 임계 소스 차이 (corr 0.994+, 정당).
- 축약 재적합 → **동일 21차원** (203일 표본 대표성 확인). 정규화 달력 90일 창, 드리프트 1.41→1.19.
- DB 1,135,331행(1182일), pool 중앙 756k = 6배. **부호일치 0.798 vs 0.499** (개선), rank100 거리 2.79 (매치 더 가까움).
- 시기 골고루 (347/366/365/120일, 공백 없음). 직근 쏠림 아님: top100 중 ≤7일 1.0% (기저 0.9%).
- ⚠️ 3단계 전 명심: **간극천장 causal 0.37bp 대조**, fee/fill/시기 audit, 진입 ≥ 라벨+1s,
  **매치 클러스터링** (같은 날 인접 분 = 비독립 표), binomial 우연 대비. 흥분 X.

---

## 🔬 2026-06-06 (보강) — I.1+ 라벨 정확성 audit ✅: 버그 3건 수정, 결론 유지

- TA-Lib+pandas-ta 대조: SMA/Stoch 완전일치, EWM계열 ~1e-5 수렴잔차. 합성 이론 12/12. OB/체결 수동 검산 일치.
- **truncation invariance 테스트 (신규 lookahead 검출기, `i_acc_verify.py trunc`)**: 미래 잘라 재계산 → 과거 라벨 변하면 lookahead. **bigflow 검출** (당일 q95 임계 = 일중 lookahead) = **함정 family #6**.
- 수정: ①bigflow 임계→이전 처리일 q95(causal) ②boll ddof=1→0(√(20/19) 상수 확인) ③RSI/Stoch flat→50.
- 수정 후 truncation 47/47 동일(미래정보 0). 재생성 + 2단계 재실행 → **결론 전부 유지** (부호일치 0.783/0.497).
- ⚠️ 라벨 시각 = 분 마지막 초 e 의 끝 → **3단계 진입 ≥ e+1s 이어야 causal** (명시됨).

---

## 🔬 2026-06-06 — [I] 유사도거래 2단계 ✅: 축약+정규화 후 유사도는 "방향"을 본다

- **축약 47→21차원** (|r|>0.7 medoid; PCA 90%→17차원 — 1단계 "~10차원" 추정 과소 교정). 정보손실 복원 R² min 0.53/med 0.86. spread_bp 제외(1틱 양자화→scale 폭발).
- **정규화 causal**: rolling = 과거 15 sampled days, 현재 day 제외 → lookahead 구조적 불가. 부호라벨 scale-only(방향 100% 보존), 크기라벨 robust-z. 연도 IQR비 2.56→~1.1.
- **시기 검증 (핵심)**: naive 의 시기쏠림은 우려보다 약했음(lift 1.19 — 라벨 다수가 이미 bounded). 정규화+whitening 으로 recency 0.81→0.94, lift→1.07 (시기 중립 달성, 전 연도 커버).
- **검색 동작**: top 매치 과거 90분 경로 corr +0.30 vs random 0.00. **부호 일치 0.782 vs 0.499** (추세·OBI·flow 15차원 전부 0.76~0.91) → 닮은 과거 = 방향 구성이 같은 상태. pool 126k, N=100+ 여유.
- 거리 척도: **whitened Euclid** 시기중립 최선 (cosine 교차확인용).
- ⚠️ 다음 3단계(70% 미래방향 쏠림) 전 명심: **간극천장 causal 0.37bp** 대조, fee·fill·시기 audit. 부호일치 0.78 은 상태 닮음이지 미래방향 아님. 203일 subsample → 3단계 전수 DB(1198일) 고려.
- 산출물: `research/i_similarity/STAGE2_REPORT.md` + parquet/csv/png.

---

## 🔬 2026-05-30 — 진행 중: 시장 분류 연구 (열린 탐색)

이전 단정(연속체 / fractal)들이 모두 한 측정/한 방법의 한계였음이 드러남. 1단계 펼치기에서 **t-SNE silhouette 0.42 > PCA 0.27** (선형이 비선형 구조를 못 봄), **breakout% C2 63 vs C0/C1 83%** (질적 분리), **세 스케일 PC1·PC2 동일** (fractal 가능성) 등이 동시에 보임. 5갈래 (비선형 강건성/fractal/breakout/거동측정/시기안정) 깊이 탐색 진행 중.

이전 모든 단정 보류: range trading 닫힘, 시장=연속체, OB-only 한계 — 모두 한 측정/평균의 함정일 수 있음. 발견에 따라 재해석.

LSTM 사전식별(1~11bar) AUC 0.582 ≈ 정적 0.576 — 시퀀스의 추가 정보는 "깨지는 순간"에만 (사후). breakout 사전 예측 불가.

range+손절 검증 (range-v2): gross +2.15bp(4/5 walk-forward, 처음 시기일관 robust edge), 단 non-VIP maker fee 4bp 못 넘음. VIP rebate 면 +6bp/trade +0.46%/day.

박스 지형 (box-map): 박스 폭 항상 fee 넘음(100%), 넓은 박스 덜 깸(corr -0.22), 안 깬 박스 사전 식별 천장 ~0.58.

---

## 🗺️ 2026-05-28 — ETH 시장 지도 (이전 "효율적" 결론 부분 교정)

**중요: 아래 NEGATIVE RESULT 의 "5min~1d random walk → 효율적" 은 퉁침 오판이었음.** 조건부/스케일별로 다시 보니 구조가 있다. 단 "구조 있음 ≠ fee 넘는 edge" (검증 진행 중).

### Variance Ratio 스케일 스캔 (Lo-MacKinlay z, N=344,917)
| 스케일 | VR | z-stat | 판정 |
|---|---|---|---|
| 초~분 (1s~5min) | **0.40** | — | 강한 mean-rev (HFT 영역) |
| 10m~8h | 0.93~0.98 | **z=-12.8 ~ -2.2** | **유의한 mean-reversion** (퉁쳐서 놓침) |
| 12h~7d | 0.93~0.96 | z=-1.6~-0.8 | random walk |

→ "5min~1d random walk" 는 틀림. N 크니 VR 0.94 도 z=-8.8 (유의). **10분~8시간이 통계적으로 mean-reverting**.

### 조건부 VR (전체 평균이 숨긴 강한 구조)
| 스케일 | 전체 | high-vol | **low-vol** |
|---|---|---|---|
| 30m | 0.96 | 0.99 | **0.79** |
| 1h | 0.94 | 0.97 | **0.79** |
| 8h | 0.96 | 0.91 | **1.44** |

→ **low-vol + 30m~1h = VR 0.79 (강한 mean-reversion, 거래가능 스케일)**, low-vol + 8h = VR 1.44 (강한 trend). 저변동 국면에서 단기 되돌림 / 장기 추세.

### 시기축: 월 단위 진동 (연도는 착시)
- reversion/momentum, VR(2h) 모두 **월~분기 단위로 진동** (2026-04 도 강한 reversion -0.40 / VR, 2026-01 강한 momentum +0.29)
- "2023 후 효율화 소멸" = **연도 평균의 착시** (reversion월+momentum월 상쇄). 8개월 walk-forward window 가 진동을 평균 0 으로 뭉갰음.

### 과거 실패 재해석
- 4h **direction(momentum)** 이 clean OOS 음수였던 건, 그 스케일이 **mean-reverting(VR 0.94)** 인데 momentum 베팅 = **방향이 반대**였기 때문.
- low-vol 단기 mean-reversion 을 조건부로 본 적 없음 → 놓친 코스.

### 미해결 관문 (구조 ≠ 수익)
low-vol 30m~1h mean-reversion 이 tradeable 한지: (1) bid-ask bounce 아닌 순수 구조인가, (2) reversion 폭이 fee 5.9bp 넘나, (3) low-vol 실시간 causal 감지되나, (4) 진짜 walk-forward 5/5. **검증 전까지 edge 주장 금지.**

---

## 📕 2026-05-28 — mark19 OB-only 한계 (NEGATIVE — 단 위 지도로 부분 교정됨)

⚠️ 아래는 direction/momentum 각도의 negative. **"5min~1d 효율적" 부분은 위 지도가 교정** (조건부 mean-reversion 구조 존재). mean-reversion 코스는 미검증.

**결론: Bybit OB 50-level + trades + funding 으로 ETH direction tradeable alpha 없음.** 모든 horizon·조건을 clean data + 진짜 walk-forward 로 스캔 완료. 가짜 edge 로 실거래 안 한 것이 핵심 성과.

### 검증 매트릭스 (전부 clean data, 진짜 walk-forward = window별 retrain)

| 신호 | 결과 | 판정 |
|---|---|---|
| Direction (OB) 4h | OOS -0.009%/day, AUC 0.534, 3/5 | ❌ |
| Direction 6h/8h/10h/12h/1d | 전부 OOS 음수/비일관, AUC ~0.52, ≤2/5 | ❌ |
| Magnitude (vol R²0.595, large-move AUC0.917) | 99% vol clustering persistence, OB alpha만 +0.009 AUC | ❌ 자명 |
| Funding conditional (극단 p10/p90) | OOS -0.083%/day, gross +0.4bp (fee 5.9 못 넘음), t-test p=0.698 | ❌ |
| Funding feature 순증분 | walk-forward AUC +0.0036 | ❌ 미미 |
| **high_vol regime conditional** | mean +12.76bp/trade, bootstrap p=0.0005, **단 window 3/5 + 6 regime 중 1개(multiple testing)** | ⚠️ promising-UNCONFIRMED |

### fee 는 벽이 아니다 (중요)
- Round-trip: Mixed 38% maker **5.9bp**, taker **12.0bp**
- Break-even accuracy: 4h 0.5375, 1d **0.514**, 7d **0.505** (long horizon 일수록 낮음)
- → fee 는 long horizon 이면 acc 0.51 로 넘김. **진짜 벽은 그 acc 0.51 directional edge 조차 없다는 것** (예측 가능성 부재).

### 과거 "발견"의 정체 (전부 함정)
| 과거 결과 | 정체 |
|---|---|
| 4h Direction +1.81 Sharpe | **day-boundary wrap-around lookahead** (build_intraday_bars sec_of_day date 무시) |
| mark36 +1.45% | lookahead |
| vol "예측 가능" AUC 0.9 | 자명한 vol clustering persistence |
| Wide-Deep Sharpe 1.19 | small-n, p=0.85, 시기 클러스터링 |
| Tardis 시도17 1.53 | 6-date small-n + 데이터도 없음(현 환경) |
| funding conditional | gross edge 0, fee 못 넘음 |
| high_vol conditional | bootstrap 통과하나 3/5 windows = Wide-Deep 패턴 |

### high_vol 단서 (재방문용, CONFIRMED 아님)
- high_vol regime(vol 상위 1/3)에서 OOS direction acc 0.534 (overall 0.503 대비 높음), mean +12.76bp/trade, bootstrap p=0.0005
- **그러나 window 3/5 + 6 regime 중 1개만 통과(multiple testing)** → confirmed edge 아님
- 단독으로 더 파면 Wide-Deep 함정 (멈춤). **cross-exchange 등 새 데이터와 결합 시에만 재방문** 후보로 보존.

### 자산화된 것 (재사용 가능 인프라)
가짜 edge deploy 를 막은 검증 틀 — 어떤 미래 전략에도 재활용:
- 진짜 walk-forward 프레임 (window별 retrain, lookahead 차단)
- clean build pipeline (day-boundary fix 적용된 build_intraday_bars_v2/v3)
- deploy stack: WS feed, reconciler, risk rails(1x, 0.01ETH), shadow runner, Discord, ΔP monitor, predict iteration_range fix
- 함정 체크리스트: clean rebuild → 진짜 walk-forward → bootstrap → window 일관성(5/5) → multiple-testing 경계

### 남은 선택지 (강요 없음 — User 의 시간/자본 판단)
1. **cross-exchange lead-lag** (B): Binance→Bybit 선행. 유일하게 안 본 진짜 후보지만 데이터 재수집 큼 + HFT 영역(우리 latency 로 잡힐지 의심) + 같은 함정 위험.
2. **근본 pivot** (C): 예측이 아닌 구조적 수익(funding carry harvest, inventory MM on 넓은 spread alt), 또는 다른 시장(options IV — vol clustering 은 실재하니 vol product 면 활용 가능).
3. **프로젝트 재평가**: 시장이 효율적이라는 것은 정직한 발견. 알파 탐색의 ROI 자체를 재고.

---

## 🛑 2026-05-28 — 4h Direction INVALIDATED (필독)

**4h Direction 전략은 deploy 직전 lookahead 버그로 무효 확정.** 가짜 edge 로 실거래 안 한 것이 핵심 성과.

### 버그: day-boundary wrap-around (lookahead leakage)
- 위치: `build_intraday_bars_v2.py` (+v3 trades) — `sec_of_day = dt.hour*3600 + dt.minute*60 + dt.second` 가 **날짜 무시**
- raw 일별 파일은 다음날 첫 snapshot (00:00:0X) 을 포함 → 그 row 의 sec_of_day≈0~2 → bar_idx 0 → **그날 bar 0 의 마지막 sample 로 wrap** → `mid_close`가 **다음날 day-start mid 로 오염**
- 예: 2026-04-29 bar0 mid_close $2253(=4-30 시가) vs 정상 $2288 / 5-26 bar0 $2074(=5-27 시가) vs 정상 $2112
- 1198일 **모든 day 의 bar 0** 오염 → `mom_1d/mom_4h/dist_ma/rv/cumflow` 등 long features 가 day-boundary에서 **미래 가격 정보 포함 = lookahead**

### Clean vs Buggy (진짜 walk-forward, window별 retrain)
| Metric | 오염(buggy) | **CLEAN** |
|---|---|---|
| OOS Sharpe | +4.62 | **-0.10** |
| OOS %/day | +0.730% | **-0.009%** (음수) |
| AUC | 0.566 | **0.534** (=ceiling) |
| Positive windows | 5/5 | **3/5** |

→ 검증된 줄 알았던 모든 것 (**walk-forward 5/5, bootstrap p=0.002, +1.81 Sharpe, AUC 0.566**) = 전부 오염 데이터 artifact. clean 에선 edge 소멸.

### 폐기
- `4h_direction_v1.joblib` (best_iter=2 underfit) — 폐기
- `4h_direction_v2.joblib` (200 trees, 오염 feature space 학습) — 폐기
- shadow_runner/mark19_live/dp_monitor 의 model 의존 로직 — 재검증 전 사용 금지

### 함정 목록에 추가 (반복 교훈)
mark36 +1.45% lookahead / Wide-Deep p=0.85 / Tardis 시도17 p=0.55 / Stress sign-flip / **4h Direction +1.81 = day-boundary wrap-around lookahead (신규)**

### ⚠️ 같은 build script 의존 결과 = 전부 의심
- vol R²=0.566, large-move AUC=0.805 등도 같은 `build_intraday_bars` 사용 → **clean 재검증 전 신뢰 금지**
- clean rebuild 완료: `bars_5min_v2_clean`, `bars_5min_v3_clean` (ETHUSDT 1198일)

### Deploy infra 자체는 정상 (재사용 가능)
WS / shadow runner / reconciler / risk rails / Discord / ΔP monitor / predict-fix(iteration_range) — 코드는 정상. 진짜 edge 만 없을 뿐.

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
