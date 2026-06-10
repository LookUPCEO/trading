# Mark19 SEARCH TREE — non-VIP fee 넘는 robust edge 탐색

매 작업 전 이 파일을 읽고, 작업 후 노드 상태/형제를 갱신한다. 휘발 방지.

## 노드 상태 (legend)
- ✅ 살아있음 (미verified, 추후 audit 필요)
- ❌ 막힘 (정직한 negative — 단 parent 사망 ≠)
- ⬜ 안 가봄 (형제 — 우선순위 백트래킹 대상)
- 🔵 탐색 중
- 💀 verified 사망 (lookahead/artifact 확인됨)

## 백트래킹 규칙
1. 한 가지 ❌/💀 ≠ 부모 분기점 사망
2. 부모 사망 = 모든 형제 ❌/💀 (전부 확인된 경우만)
3. 가지 막히면 → 부모로 → 안 가본 형제 우선 (먼 점프 금지)
4. "엣지 없음 / R&D 종료" 결론 = 루트까지 전부 막혀야

## 단정 금지 표현 (현재 막힘 표현하되 사망 단정 X)
- ❌ "...effectively dead", "...not tradeable", "R&D 종료"
- ✅ "이 진입틀에서 ❌", "이 청산틀에서 ❌, 형제 ⬜ 남음"

---

## 트리

**[루트]** ETH 1Hz OB/trades 공간에 non-VIP fee 넘는 robust edge?

### [A] Direction 예측
- A.1 4h Direction model ❌ (clean walk-forward OOS -0.009, 더 정확히는 lookahead bug 였음 — buggy +0.73 / clean ≈ 0)
- A.2 horizon sweep (6h/8h/10h/12h/1d) ❌ (전부 OOS 음수/비일관)
- A.3 OB+funding XGBoost ❌ (funding feature AUC +0.0036, 미미)
- A.4 conditional regime XGBoost ❌ (regime adaptive 1/5 OOS)
- A.5 LSTM box sequence ❌ (사전 식별 AUC 0.582 ≈ 정적)
- 형제 ⬜:
  - **OB+trades+funding 결합 신호** (체계적 멀티스트림)
  - **비선형 모델 제대로** (CatBoost / Transformer / TabNet — 우리 XGBoost 정적·LSTM 박스에 한정)
  - **시간 결합 (요일+시간대+이벤트) conditional**
  - **진입 후 관찰형 (post-entry signal)** — B 의 30s 정면 💀 (lookahead) 단 **다른 관찰 시간** 미탐색

### [B] Magnitude
- B.1 vol R²=0.595 ✅ → 자명 (naive vol[t]=vol[t+1] R² 0.561, OB alpha +0.034)
- B.2 large-move AUC 0.917 ✅ → 자명 (naive AUC 0.908, OB alpha +0.009)
- 거래 edge ❌ (크기만 예측, 방향 X)
- 형제 ⬜:
  - **vol-aware MM sizing** (예측 가능 vol 로 size/spread 조정 — 우리는 MM 아님)
  - **straddle / vol-product** (옵션 없음, ETH perp 만)
  - **large-move 직전 회피 신호로** (예측 가능 시 cancel — 단 MM 영역)

### [C] Conditional / Regime
- C.1 funding conditional (extreme p10/p90) ❌ (OOS -0.083%/day, gross +0.4bp)
- C.2 high_vol regime conditional ❌ (3/5 windows, multiple testing — bootstrap p=0.0005 였으나 시기의존)
- C.3 시퀀스/구조/조합 314 패턴 + Bonferroni ❌ (0/314 통과)
- C.4 regime adaptive (autocorr/200MA) ❌ (causal 1/5)
- 형제 ⬜:
  - **제대로 된 며칠짜리 시장 분류** (5갈래 깊이서 silhouette 0.17~0.27 약함 확인 — 단 unsupervised k=4·5 분리 안 봄)
  - **regime-aware fee tier 변경** (특정 regime 만 거래해 빈도 줄여 다른 fee 협상)
  - **micro-regime** (분 단위 vol-cluster 켜져있는 동안만)
  - **2-feature 전수 interaction** (314 는 논리적 조합만)

### [D] Mean-reversion (단방향)
- D.1 ACF lag1~lag48 ✅ universal (lag 1 -0.022 유의)
- D.2 거래 edge ❌ (gross 0.43bp << fee 5.9bp, MM rebate 영역)
- 형제 ⬜:
  - **다른 horizon MR** (lag 6 이상 더 긴 — small)
  - **조건부 MR** (high-vol 만, low-vol 만 — VR 0.79 발견과 연결)
  - **portfolio MR** (BTC/SOL 등 multi-asset MR — 데이터)

### [E] Range / 박스
- E.1 진입틀
  - band touch (2σ) ❌ (gross +0.33bp 1Hz)
  - band 안 (1σ) ❌ (gross -1.43bp 1Hz)
  - breakout 후 pullback ✅ (1Hz gross +2.84, pos 66%) **미verified**
  - 형제 ⬜:
    - 시간 기반 진입 (특정 시각)
    - 큰 trade 직후 진입
    - OB imbalance 진입
    - momentum entry (방향 + momentum)
    - 큰 박스 사전 식별 + 진입
- E.2 청산틀
  - tight target+stop ❌
  - trailing 5bp 💀 (5min OHLC 순서 lookahead artifact — 1Hz audit 에서 -0.98)
  - trailing 30bp ✅ (5min sim +7.89, **1Hz 미verified**)
  - trailing 100bp ✅ (5min sim +9.11, **1Hz 미verified**)
  - no-stop hold ✅ (5min sim +8.90, **1Hz 미verified**)
  - 형제 ⬜:
    - 부분청산 (1/2 target + 1/2 trail)
    - 조건부 청산 (volume spike 후 청산)
    - **trailing 의 1Hz 정밀** (1Hz 에서 큰 trail 도 같은 패턴인지)

### [F] Timing 실현 (entry-conditional)
- F.1 분산 진입 ❌ (단일 평균과 동일)
- F.2 trailing 5bp 청산 💀 (5min OHLC artifact)
- F.3 진입 후 30s 신호 → 5m hold 💀 (side=sign(t=30s) 를 PnL=t=0 에 적용 lookahead)
- 형제 ⬜:
  - **부분 capture** (peak 30/50/70% — cheat oracle +29.6bp 의 진짜 의미)
  - **큰 박스 사전 식별** + 그 안 timing
  - **진입 후 60s/120s 신호** (정직 case B 로 측정)
  - **진입 후 OB 변화** 신호

### [G] Q0 oracle gross
- G.1 oracle one_way 5m: 시기 안정 12~18bp, fee 8bp >50% ✅
- G.2 단방향 (long_only/short_only): 4.6~9.6bp, fee 8bp >30-58% ✅
- G.3 vol clustering = 크기만 (방향 X) ✅
- 실현 경로 (2026-05-31 형제 전수 탐색, causal side rule t≤0 정보만):
  - 사전 vol 게이트 ❌ (크기만)
  - 진입 후 30s 💀 (lookahead)
  - **형제1 부분capture/스케일 ❌** — gap 이 죽임. causal 방향은 oracle 11.8bp 중 0.37bp 만 잡음 (3%). 그 0.37bp 의 일부를 부분 capture 해봐야 fee 고정비 대비 더 불리. peak 에서 부분청산 = 청산시점이 미래정보 = lookahead.
  - **형제2 hedge/straddle (실현가능 oracle) ❌** — delta-neutral 은 방향 PnL 0 (구조적). loser stop+winner hold 1Hz 시뮬: stop 10/20/40bp 전부 gross +0.04~0.67bp, net (fee~15) **-14bp**. "방향 안 정하고 oracle 잡기" 는 신기루 — 방향 노출 = side 결정 = 형제3 으로 환원.
  - **형제3 vol게이트 × 진입전 방향신호 ❌(fee 미만)** — causal 방향규칙 (mom5/rev5/mom15/OBI) gross: mom5 +0.008, mom15 +0.155, **OBI +0.369** (best). 전부 fee 4bp(M+M) 미만. **단 유일하게 구조 발견**: |OBI|q5 (강한 호가 불균형, t=0 causal) → 5m 방향 gross **+1.59bp**, win 0.543, |OBI|×vol 셀 최대 +2.35bp (q5×volq3). 시기: +1.22/+2.04/+1.81/+0.53 (4년 양수지만 2026 감쇠). **여전히 fee 미만 + maker fill adverse selection (range-v2 함정) 미적용** → 적용 시 더 악화.
  - **형제4 정보-진입 간극 천장 = CONFIRMED**: gap = oracle(11.77) − best causal(0.37) = **11.41bp (97%) 실현불가**. 이것이 "30s 덫" 의 일반형 — 모든 실현경로 공통 천장. oracle gross 는 풍부하나 **t≤0 정보로 잡을 수 있는 방향분은 fee 미만**.
  - OBI child A-1 (풍부한 데이터 재계산, 2026-05-31) ❌ at fee:
    - **데이터 인벤토리**: OB 50레벨 1Hz 1198일 ✅, trades(publicTrade aggressor side+size) 1198일 ✅. 이전 OBI = top5만·trades 미사용·정적 = 빈약했음 확인.
    - **깊이 가설 실패**: obi1/obi5(최상단)이 최선(gross +0.39/+0.37), 깊을수록 악화(obi20 +0.14, obi50 +0.07, q5 음수). 깊은 레벨=허수호가 노이즈. 거리가중 개선 없음.
    - **trades flow contrarian+음수**(flow5m −0.197): aggressor 매수누적→하락반전. corr(obi5,flow5m)=0.02(독립이나 약함).
    - **OB 동역학 dobi5_30 최선 단독** +0.581bp (정적보다 약간 나음).
    - **결합(등가중 obi5+dobi5_30−flow5m, 과적합회피)**: gross +0.449bp (bootstrap CI [+0.14,+0.74]), 강신호 top10% +1.33bp.
    - **시기 강한 감쇠**: 2023 +0.84→2024 +0.45→2025 +0.14→2026 +0.22 (효율화).
    - **fill adverse selection 정면(실제 trades 체결판정)**: 강신호 top10% naive(mid) +1.33 → maker 체결분 +0.83bp (37% 삭감, fill rate 0.91). **maker net −6.67bp, taker net −9.69bp**. range-v2 함정 재현.
    - 간극 천장 안 줄음: oracle 11.77 vs 실현가능(fill후) 0.83 = 여전히 ~11bp 실현불가.
  - OBI child A-1b (결합 공간 전체, 2026-05-31) ❌ at fee — A-1 "결합 다 봄"은 과장(개별+선형1개)이었음을 교정:
    - **비선형/조건부/상호작용 전부 + OOS(날짜 시간분할) + FDR(BH)**. 151 조합 시도 기록.
    - XGB 회귀 OOS gross +0.32(shallow)/+0.05(deep, 과적합 붕괴), XGB 분류 OOS −0.11.
    - 로지스틱+pairwise 상호작용(78항) OOS +0.46, 강확신10% +3.55bp (단 1모델·fill 미적용).
    - 조건부 규칙 147개 grid: train 최선 obi×vol +4.5bp대, **FDR 45/147 통과**. train상위10 OOS 평균 +0.79(대부분 붕괴, obi×vol 계열만 생존).
    - **시너지 천장**: obi5×vol "생존"은 drift-agreement selection 인공물(취약 regime 베팅)이었음 — 깨끗한 게이트해석 obi5방향×vol게이트 = **+1.09bp naive (개별 천장과 동일)**.
    - **fill adverse selection (obi5×vol OOS, 실제 trades)**: +1.09 → maker 체결분 +0.64 (fill 0.96) → net −3.4(M+M 낙관)/−6.9(M+T). 시기 2025 +1.21→2026 +0.58 감쇠.
    - 결론: 결합 공간(비선형 포함)도 fee 못 넘음. 강확신 일부 +3.5bp 도달하나 fee(4~7.5) 미만 + fill 미생존 + 취약. **이번엔 결합까지 봐서 강한 닫힘.**
  - 살아있는 child ⬜ (G 사망 아님 — in-data 경로(개별+결합) 소진):
    - **OBI/dobi 의 native 영역 = 더 짧은 horizon (1s~30s)** → root [틱~초 HFT]. 거기선 방향(OBI)은 있고 fee/latency 가 벽 (infra/tier = R&D 경계).
    - **A-2 liquidation 등 외부신호** — 단 신중: 벽이 6bp(fee−실현gross)인데 최강 in-data 결합이 +1bp·감쇠. 외부 1신호가 6bp 메울 가능성 낮음 + multiple testing 위험.
    - **다른 holding scale 긴쪽 (4h/1d)** 미측정 (단 A.2 horizon sweep 이미 ❌ 경향).

### [H] 외부 신호 (가격 반영 전 방향) — in-data 소진 후 진입
- H.1 Binance(spot) → Bybit(perp) lead-lag (2026-05-31, 파일럿 8일 1s klines) ❌ at our latency:
  - 데이터: Binance 공개 1s klines 다운로드(data.binance.vision), epoch 단위 가변(ms→2025부터 µs) 정규화. Bybit OB 와 epoch-sec 정렬.
  - lookahead 정렬: Binance 신호 close[t](정보 ~t+0.999s) → Bybit 진입 mid[t+1+δ] (δ=latency초). 신호 < 진입 보장.
  - **cross-corr: lag0(동시) +0.84 지배**. lag−1s(Bybit선행) +0.135 > lag+1s(Binance선행) +0.110 → 1s 해상도 사실상 동시, 오히려 Bybit 미세 선행. **"Binance 선행" 미지지.**
  - 방향 gross(Binance 모멘텀→Bybit, δ≥1s): **+0.04~0.05bp (≈0)**, win<0.5. 시기 **감쇠 2023 +0.115 → 2026 −0.012**. fee/fill 적용 전 이미 죽음.
  - 결론: 거래가능 timescale(latency≥1s)에 선행 없음. 선행은 sub-second(=latency/HFT 게임, 환경 밖) — 차익거래 소멸. 사용자 회의 적중.
  - 형제 ⬜: **Binance perp(futures aggTrades) vs Bybit perp** (spot 썼음 — 단 0.84 동시상관상 perp도 더 동기화일 가능성 높아 low-value), liquidation/OI, on-chain.
- H 형제 ⬜ (미탐): liquidation cascade, open-interest 급변, 펀딩 arb, on-chain flow.

### [I] 유사도 기반 거래 (사용자 아이디어, 2026-06-06~)
컨셉: 매분 라벨링(지표조합) → 과거 유사시점 검색 → 방향 70%+ 쏠리고 폭>fee면 거래.
단계검증: ①라벨링 ②유사도 ③70%쏠림 건수 ④fee넘는 건수. (각 단계 통과해야 다음)
- I.1 **라벨링 (지표계산+분포정상성) 🔵→✅ 통과** (2026-06-06):
  - 203일(2023~2026) 243k분행, **47 라벨** 계산 (추세/이격·모멘텀·변동성·OBI·flow·캔들, 전부 t≤0 causal, wrap제거).
  - 분포 정상: 범위위반 0(RSI/Stoch/ADX∈[0,100], OBI/flow∈[-1,1]), 상수0, NaN≈0(ma_slope_240만 20%). 시각화 4일 의도대로 찍힘.
  - ⚠️ **중복 강함**: 47→유효 독립차원 **~10개** (추세군/변동성군/OBI군/flow군 + 독립단독). 그대로 거리투입시 유사도 왜곡 → 2단계 전 축약(대표/PCA) 필수.
  - ⚠️ **시기 드리프트**: 변동성-스케일 라벨(rv/atr/boll_width/range/ma_dev/macd/spread)은 2023저변동→2025 ~2배. z/비율 라벨(rsi/stoch/adx/obi/boll_pos/vol_z/flow)은 안정. → 변동성군 rolling z-score 정규화 필요.
  - 산출물: `research/i_labeling/` (labels.parquet, *_distribution.csv, corr_heatmap.png, temporal_boxplots.png, viz_*.png, STAGE1_REPORT.md). 코드 `scripts/i_labeling.py`,`scripts/i_validate.py`.
  - **이번엔 거래/예측/edge 결론 X** — 라벨 정상성만 확인.
- I.1+ **정확성 보강 (라이브러리 대조+truncation) ✅ 통과, 버그 3건 수정** (2026-06-06):
  - TA-Lib+pandas-ta 대조(6일): SMA/Stoch 완전일치(0), ATR/MACD ~1e-8, RSI/ADX ~1e-5(EWM 시드 수렴). di_diff 는 TA-Lib 과 4e-6 일치 (pandas_ta DMP 자체 정의가 비표준). 합성 이론 12/12 PASS. OB/체결 수동 검산(원시 독립경로) 전부 일치·부호 일관.
  - **truncation invariance 테스트 (신규 lookahead 검출기)**: 미래 자른 재계산 → 과거 라벨 변하면 lookahead. **bigflow_5m/norm 검출** (당일 전체 q95 임계 = 일중 lookahead, 426/461행 실증) — 함정 family #6.
  - **수정 3건**: ①bigflow 임계 → 이전 처리일 q95(causal, 첫날 NaN) ②boll std ddof=1→0 (TA-Lib 대조에서 비율 정확히 √(20/19) 상수 확인 → 표준 정렬) ③RSI/Stoch flat → 중립 50 (구 0='바닥' 오신호. RSI 실발생 0행, flat 14분창 14행).
  - 수정 후: truncation **47/47 라벨 완전 동일(미래정보 0)**, 라이브러리 재대조 통과, 라벨 재생성 + **2단계 파이프라인 재실행 → 결론 전부 유지** (recency whitened 0.92/lift 1.04, 부호일치 0.783 vs 0.497, 노이즈 범위 변화).
  - ⚠️ 라벨 시각 정의: 분 라벨 = 그 분 마지막 초 e 의 끝 (trades round-binning ≤e+0.5s) → **3단계 진입은 다음 분(≥e+1s)이어야 causal**.
  - 산출물: `research/i_labeling/accuracy/ACCURACY_REPORT.md`, 코드 `scripts/i_acc_verify.py` (lib/synth/manual/trunc — 상시 재사용).
- I.2 **유사도/거리 (축약+정규화+검색검증) 🔵→✅ 통과** (2026-06-06):
  - **축약 47→21차원**: 정규화 후 spearman |r|>0.7 군집 medoid. PCA EVR 90%→17차원 — **1단계 "유효 ~10" 추정은 과소였음(교정)**. 정보손실: 탈락 25개 복원 R² min 0.533/med 0.862. spread_bp 제외(1틱 양자화→scale 폭발). 방향군 대표 전존.
  - **정규화 causal**: rolling 통계 = 과거 15 sampled days(달력~90일), **현재 day 제외 → lookahead 구조적 불가**. 부호라벨 scale-only(부호 100% 보존), 크기라벨 full robust-z. 연도 IQR비 2.56→~1.1 (rv_3600 1.41 잔존).
  - **시기 분포 (핵심)**: 정직 발견 — naive(raw47 global-z)의 시기쏠림이 우려보다 약했음(same-yr lift 1.19; 다수 라벨이 이미 bounded). 정규화+whitening 으로 recency 0.81→0.94, lift 1.19→**1.07** 일관 개선, 매치 연도분포 ≈ pool 구성 (전 연도 커버).
  - **검색 동작**: 과거 90분 경로 corr top50 **+0.30 vs random 0.00** (top>rnd 85%). **부호 일치(방향 닮음) 0.782 vs random 0.499** (부호 15차원 전부 0.76~0.91) → **유사도가 "방향 구성이 같은 상태"를 찾음**. N충분: pool 중앙 126k, rank100 거리 3.28 << pool 중앙 6.73.
  - **거리 척도**: whitened Euclid 가 시기중립 최선 → 3단계 기본 + cosine 교차확인.
  - 산출물: `research/i_similarity/` (STAGE2_REPORT.md, labels_norm_reduced.parquet 21차원, 검증 csv/png). 코드 `scripts/i_reduce_norm.py`,`i_simsearch.py`,`i_signcheck.py`.
  - ⚠️ caveats: 부호일치 0.78 = "상태 닮음"이지 미래방향 아님(이번엔 미래 안 봄). 203일 subsample — 3단계는 전체 1198일 전수 DB 고려. 일부 쿼리 잔존 시기쏠림(p10 0.57) → 3단계 판정은 시기분해 필수.
- I.2+ **전체기간 확장 (1198일 라벨링 + DB 재구축) ✅** (2026-06-06):
  - 검증된 코드 그대로 STEP=1: **1,436,626 분행** (2023-01-18~2026-04-27, 달력 공백 없음). truncation 추가 시기(2023 저변동/2026) **47/47 동일 — lookahead 0 재확인**. 전수 inf/범위위반 0.
  - **일관성**: 겹치는 203일 45/45 라벨 1e-12 동일. bigflow 만 예상차이(causal 임계 6일전→1일전, corr 0.994+, 양쪽 다 과거만 = 정당). **축약 전체 재적합 → 동일한 21차원** (203일 표본이 대표성 있었음). 정규화 D_WIN=90(달력), 드리프트 최대 1.19 로 개선.
  - **DB**: 유효 1,135,331행(1182일), pool 중앙 756k = 203일판의 **6.0배** (3단계 표본). 검색 품질 개선: **부호일치 0.798 vs 0.499**, rank100 거리 3.21→2.79 (pool 깊어져 매치 더 가까움). 전체에서도 "방향 구성"을 봄.
  - **시기**: 연도 골고루(347/366/365/120일), 과대표 없음. 직근 쏠림 아님 — top100 중 ≤7일 1.0%(기저 0.9%), ≤30일 5%(기저 3.8%, lift 1.24). recency whitened 0.87/lift 1.15 (어제 포함 효과, 약함).
  - 산출물: `research/i_similarity/FULL_EXPANSION_REPORT.md`, labels.parquet(전수)/labels_step6.parquet(보존), 21차원 DB 1.135M행. 코드 `i_full_consistency.py`,`i_proximity.py`.
- I.3 **70% 방향쏠림 ✅ (v2 — horizon 기반 독립으로 교정)** (2026-06-07):
  - v1(day당 1개)은 과처리 (사용자 지적 적중 — 같은 날 5h 떨어진 5m 미래는 안 겹침). **올바른 독립 = 미래 창 비겹침 (같은 day |Δt|≥horizon)**. greedy 거리순, null 도 같은 제약. 유효 N: 5m~1h med 94~99 (고유일 86~90 + 같은날 복원), 4h 77 (물리한계).
  - 재측정: thr70 0.16~0.25% vs null 0.01~0.04% (×15~25) — **비독립 부풀림 없음** (v1보다 오히려 보수). OOS train 0.19~0.32%→test 0.10~0.13%. v1 결과는 lean70_per_query.parquet, v2 는 *_v2_* 로 보존.
- I.4 **hit rate + 폭 + net ✅ 부분통과 — mark19 최초 fee 초과 후보 (단 test 통계 미확정)** (2026-06-07):
  - **hit rate (쿼리 실제 미래, 구조적 OOS)**: thr70 — 30m hit 0.637/gross +16.1bp, 1h 0.640/+45.0, 4h 0.684/+101.5 (base ~0.50). net(T+T): +5.1/+34.0/+90.5bp. ⚠️ **I.5 에서 정정**: 당시 'CI 0 제외' 는 gross CI 혼용 — **net 직접 CI 로는 30m ✗ (95%도 0 포함), 1h ✓(95%)/✗(Bonf), 4h ✓✓**. thr65 전부 fee 미달, thr70 5m/10m 한계.
  - **audit 전부 통과**: cheat injection(배선 정상), outlier 아님(1h med +28, top3 제외 +38), drift 수집 아님(분기벤치 ≈0), day 군집 약함(139건/112일). 약점: **up-lean 주도** (1h down hit 0.54 +10bp — short 약함).
  - **간극천장 0.37bp 와 모순 아님 — 정의역이 다름**: 천장 = 전수·고빈도 평균 gross. 이건 희소(0.2%) 조건부 선택, per-trade +16~+101bp. k-NN 지역 조건부가 선형규칙 밖 국소 구조를 봄 (실증, "천장 깨짐" 단정은 안 함).
  - **정직 한계**: ①test(2025Q3~) 단독: 점추정 양수(+19.9/+48.1/+62.8) but **CI 전부 0 포함** (n=27/36/17 소표본) ②1h 2026 -6.2bp 음수(n=17), 4h 만 3년 전부 양수 ③일수익 thr70 합 ~15bp/day (test ~8) — **목표 50bp/day 미달** ④다중검정 형식 보정 미적용(10셀 중 3 생존, 단 CI 폭은 보수보정 생존권) ⑤분기 변동 큼 (8/8 일관 아님 — 4h 최선).
  - 산출물: STAGE4_REPORT.md, lean70_v2_per_query.parquet, lean70_v2_hit_net.csv, lean70_v2_net.png (누적 net 1h/4h 전기간 우상향).
- I.5 **walk-forward + 단조 강건성 ✅ — 4h thr70 "약하지만 진짜" 에 최근접 (확정 아님)** (2026-06-07):
  - **walk-forward 구조**: pool=prefix+룰 사전지정+whitening 2023 → 전 81,682 쿼리 구조적 OOS. 순수성: 대표선택 2023 vs 전체 상관 max|Δr| 0.194/med 0.012 (오염 무시), 사후선택은 Bonferroni 99.5%(10셀) 보정.
  - **폴드(반기 5)**: 4h **5/5 양수** (+86/+126/+72/+47/+55, 2026 포함), 1h 3/5 (2026H1 -17), 30m 2/5.
  - **누적 net CI (교정: net 직접+day-mean)**: **4h n=79 net +74.7, Bonf 99.5% [+8.3,+146.7] 유일 단독 생존**. 1h +27.7 95%✓/Bonf✗(-1.7 마지널). 45m +21.0 95% 마지널. **30m ✗ — stage4 'net CI 0 제외' 주장은 gross/net 혼동 오류, 정정**. 결합(408건/225일) +19.5 99.5% [+2.0,+38.2]✓. **2025Q3~ 단독 ✗** [-5.4,+56.0] (점추정 +23.6 동부호 — 표본한계 우세 판단, 단정 금지).
  - **thr 단조 (0.60~0.78)**: 매끄러움 — gross 가 fee 를 thr≈0.67 에서 연속 통과 = 구조 (톱니 없음).
  - **horizon 촘촘 (15m~6h)**: 매끄럽지 않음 (정직) — 거친 증가 + **2h/3h 골 (edge 0.07~0.09, 열린 질문)** + 4h 돌출. 4h confounder 점검: 시간대 제한 시 30m/1h 오히려 약해짐 → **4h 돌출은 인공물 아님 (보수적 달성)**. 사전등록 가설: fee/|move| 구조 확증, ~4h 한계 후 감쇠는 6h n=27 미검증.
  - **why**: ①fee 11bp = 5m |move| 의 65% → 4h 의 10% (짧은 h 구조적 불가) ②hit-edge 0.06→0.37 동반상승 = **net↑ 는 폭×예측 둘 다** ③lean 순간 = 고변동 순간 (|move| 시장 중앙 ~2배). 부수관찰: 늦은 시간대(US) lean 강함 (사후 — 결론 아님).
  - 산출물: STAGE5_REPORT.md, lean70_v2_per_query_hfine.parquet, why_horizon_decomp.csv, why_horizon.png. 코드 i_wf_folds/i_wf_net_ci/i_why_horizon.py.
- I.6a **"닮음" 정의 검증 (라벨 조합 31개) ❌ → 21차원 유지 확정** (2026-06-07):
  - 사전등록 (결과 전 commit): 의미 5군 조합 31개 정의, 선택=train(2024Q1~2025Q2, g65 지표)만, 판정=OOS(2025Q3~)만, Bonferroni 분모 31.
  - train 상위: V(3차원) +8.64 / V+O+C +6.67 / O +5.91 (베이스라인 +6.35 3위).
  - **OOS 판정: 진출 3개 전멸** — V net -36.0, O -56.7(음수), VOC +12.9 < 베이스라인 +23.6. CI 양수 0개. 일수익 -3.6~+1.0 vs 베이스라인 +8.1bp/day. **"train 좋음=과적합 기본값" 실증 — 사전등록 분리가 trailing/B-30s 재현 차단한 사례.**
  - why: 쏠림의 질 = 상태 전체(5군)의 동시 일치. 부분공간 = 가짜 이웃 + 이벤트 희소화 (n 408→51~63). stage2 부호일치(15차원 전부 0.76+) 와 정합.
  - 산출물: STAGE6_REPORT.md, simdef/ (phase1/2 parquet+summary). 코드 i_simdef.py/i_simdef_judge.py.
- I.6-1 **능동청산 ❌ + shadow 전향검증 가동 ✅** (2026-06-07):
  - **A 능동청산 (사전등록 5규칙, 4h thr70 79건)**: 전부 고정 hold 패배 — FIXED +74.7 > R1반전강 +52.8 > R3 +45.0 > R5익절 +34.1 > R2 +32.0 > **R4손절 +12.0 최악**. 시기 전/후반 모두 FIXED 최고. trailing 함정 회피 (결정 ≤ 정보시점, 청산가 동등취급). why: lean=고변동 순간 → 중간 트리거 다 걸림, 신호 가치 = "4h 끝까지" (winners run). **이 규칙공간 ❌ — 고정 hold 유지** (다른 청산공간 ⬜ 남으나 방향성 명확).
  - **B shadow 인프라**: 검증 i_labeling 함수 그대로 + 동결 artifact (DB 1.135M×21). **replay 일치검증(2026-03-15): 라벨/z/fup240 전부 max|Δ|=0.00e+00, 신호일치 1.000** — 백테스트=실시간 bit-identical. 데몬 가동 (분 jsonl + outcomes.jsonl, 1Hz 수집 재개 겸용, big_thr 전일 q95 자동갱신).
  - caveats: 정규화 통계 ~5주 stale (수집공백 5/1~6/6 — 백필 옵션), 유효신호 다음 UTC 자정부터, 기대 0.09건/일 → 결론까지 수개월, 머신 상시가동 필요.
  - 산출물: STAGE6_1_REPORT.md, active_exit_results.csv, shadow/ (artifact+logs). 코드 i_active_exit.py, i_shadow.py, i_shadow_daemon.py.
- I.6-2 **능동청산 재구현 (재예측 갱신) ❌ — 고정 4h hold 유지 확정** (2026-06-07):
  - 6-1 A 정정 (가격 익절/손절 = 의도 오구현) → 진짜 의도: 매분 동일엔진 재예측, 방향유지=hold(fee 0)/전환=flip(만 fee)/소멸=청산. 사전등록 V1/V2/V3.
  - **V1 연장+전환 +73.8 ≈ FIXED +74.7 (동률)** — flip 은 단 2회/79건 (반대 thr70 이 4h 내 사실상 안 옴 → '갈아탐 가치' 실재 X). **V2 중립소멸 -0.2 (hold med 8분 — 즉시청산화, 사전등록 예측 적중)**, V3 방향상실 +22.6. 일수익 +8.45 vs +8.40 — **빈도 효과 0** (병목은 보유시간 아니라 진입 빈도: lean ~3일 1건).
  - 원리: 신호 정보는 진입 순간에 응축 — 직후 중립 재예측은 lean 의 정상 소멸이지 새 정보 아님. lookahead 구조 차단 (결정 at m = m 末 정보, 가격 mid[m] 동일관행).
  - **청산 개선 가지 합계 8규칙 (6-1 가격 5 + 6-2 재예측 3) 전패 → within-day 틀에서 ❌.** 일수익 개선은 진입 빈도가 병목.
  - 산출물: STAGE6_2_REPORT.md, repredict_summary.csv, repredict_V*.parquet. 코드 i_repredict_exit.py.
- I.6-3 **thr 곡선 (0.60~0.72) ❌ 개선 없음 — thr 0.70 = train/test 동시 균형점, 빈도 한계 확정** (2026-06-07):
  - 사전등록: 곡선+train argmax→OOS 판정 (단일값 cherry-pick 방지, 기존 parquet 재임계).
  - 곡선 매끄러움 (train/test 단조), fee 하한 0.67~0.68 (5단계 예측 적중). **thr<0.67 = 빈도 늘수록 손해** (0.60: -300bp/day).
  - **균형점: 결합 thr0.70 train +18.99 → test +8.08 둘 다 1위** (plateau 0.69~0.72 안정 = 구조). 1h 단독 train 0.69 는 test 4위 — 단일 horizon 선택의 과적합 실증 (결합이 강건). 30m thr0.72 부활은 n=12 노이즈.
  - **목표 대비: 결합 thr0.70 = 15.1bp/day (30%) 가 최선.** thr0.68 (빈도 2.5배) +9.1 악화, 0.66 음수. **빈도는 파라미터가 아니라 신호의 본질 (고합의 ~0.2%) — thr 축 닫힘.**
  - 일수익 개선 3축 전부 닫힘: 닮음 정의 (6a) / 청산 (6-1,6-2) / thr (6-3). [I] 단독 일수익 = 목표의 30%.
  - 산출물: STAGE6_3_REPORT.md, thr_curve.csv/png.
- I.6-4 **thr 분포 정밀 (평균 함정 교정) ❌ — 빈도 한계 분포 수준으로 확정** (2026-06-07):
  - 사전등록 48셀 (저밴드 4 × 분할 11+1), 시도 48 (분모 일치). 서로소 fup 밴드 분포.
  - **분포 발견**: 저밴드 (0.60~0.70) 승률 43~46% 거의 평탄, median -5~-8 — "0.66 평균 음수" 안에 이익거래 45.4% 실존 (사용자 지적 사실). 단 **0.70 에서 질적 점프** (승률 58.3%, med +17.8) — 평균 곡선의 매끄러움 아래 실체는 0.70 임계의 분포 도약 = "70% 합의" 직관 정합.
  - **부분집합 OOS 전멸 0/5**: train 양수 (최강 [0.66,0.68)×down +7.0) → test 전부 음수 (-13.6 등). 95% CI 생존 0.
  - 판정: 이익 "거래" 는 있으나 식별 가능한 "부분집합" 없음 — 빈도↑=질↓ trade-off 못 깸. 6-3 결론 유지 + 근거 업그레이드 (평균→분포+OOS).
  - 산출물: STAGE6_4_REPORT.md, thr_dist_cells.csv.
- I.6-5 **thr 0.685~0.71 소수점 정밀 ❌ 개선 없음 — 점프 ~0.70 실재(ramp), 0.70 확정** (2026-06-10):
  - 사전등록 (점프 위치+표본한계 명시). fine bins (각 n=21~86) 는 노이즈(Wilson CI 겹침) — 단 **rolling 승률(윈도 300)은 명확**: 0.60~0.70 평탄 노이즈(~45%, med 음수) → **0.70 onset 상승 → 0.72 plateau ~60%**. 6-4 "0.70 점프" 는 coarse 인공물 아니라 실재, 단 step 아닌 ramp.
  - **OOS 안정**: test 0.685~0.70 음수(승률 41%, med -11) / 0.70+ 양수(57%, +18). **더 일찍 진입 = OOS 손해.** 일수익도 strength≥0.695 +13.2 < 0.70 +15.1 (전체), test +7.2 < +8.1 → **0.70 이 3중(전체·OOS·일수익) 최선.**
  - 판정: 점프 실재·~0.70·OOS 안정 — 빈도 개선 0 ("이해 > 개선", 예측대로). 표본한계 정직 (소수점 위치 ±2tick 불확실, "0.70 미만 못 씀" 만 강건).
  - 산출물: STAGE6_5_REPORT.md, thr_fine.png. shadow 전향 2건 누적 (06-09 net -9.5/+147.85, n=2 무의미).
- **일수익 개선 4.5축 전부 닫힘** (정의 6a / 청산 6-1·2 / thr평균 6-3 / thr분포 6-4 / thr소수점 6-5). 현행 21차원·whitened·thr0.70·고정 4h hold·결합 = [I] ETH 최종 운영점.
- I.7 **multi-horizon 동시운용 + 시기×horizon ❌ (빈도 못 풂) — 단 SOL.2 정정 + 합의 발견** (2026-06-10):
  - **상관(핵심)**: thr0.70 신호 동시발생 89~113x(독립 대비)·동시 시 **같은방향 100%** → multi-horizon = 독립분산 아님, 중복 베팅. 자본 1/3 분산 일수익 **+5.0 < 단일 4h +8.4** (상관이 분산 죽임). 이벤트 5x(408 vs 79)나 시간상 뭉침 → 빈도 문제 못 풂.
  - **SOL.2 감쇠 정정**: full DB 로 최근 2025Q3+ **4h +51.8**(hit 0.765, 양수) — SOL.2 의 -44.6 은 182일 DB 인공물. **감쇠 미확정(open), DB 깊이가 최근 신호 핵심, shadow 가 판정.** regime 이동 아님(4h 희소·1h 일관·30m 노이즈).
  - **합의 발견**: 3+ horizon 동방향 hit **0.81** net +156(n=16) / 2+ hit 0.73 — 합의=질 강화(진짜). 단 빈도 감소·고strength 와 겹침·n 극소. 질 부스트이지 빈도 해결 아님.
  - audit: top3제외 +29.8(outlier X), 6시도 전부 보고. 빈도 한계 = 6축 모두 확인 (구조적 ~15bp).
  - 산출물: STAGE7_REPORT.md. 코드 i_multihorizon.py.
- I.8 **경로 모양(과거길이 L × 미래 H) 유사도 ❌ — 순간 21차원 완패** (2026-06-10):
  - 모양 요약 5특징(순수익/기울기/곡률/반전율/종말모멘텀) × L 12개{3~240분}, 정규화(÷window vol)·whiten 2023, t≤0. 미래 H 7개{5~240분}. 83 (L,H) 짝.
  - **빈도는 오히려 풍부** (L=240,H=5 n=338 ≈ 순간 결합 408) — 단 **전부 net 음수·hit 0.49~0.60** (예측력 없음). train net 양수 6/47, n≥30 양수 1개(L180/H60 +5)→test -9.5 사망. Bonferroni 47 생존 0.
  - why: 명시적 경로는 순간 라벨에 이미 redundant (ma_slope_5~240=경로 기울기, macd=모멘텀, rsi=위치) + OB/flow 맥락 상실 → strictly 나쁨. 본인 "오는 길" 정보는 이미 순간 추세군이 포착.
  - 판정: 빈도/질 개선 8축(정의/청산/thr평균/분포/소수점/multi-horizon/경로) 전부 ❌. **순간 21차원·thr0.70·고정4h·결합 = [I] 최종형 재확정.** 감쇠는 shadow 가 판정.
  - 산출물: STAGE8_REPORT.md, pathshape_LH.csv, pathshape_heatmap.png. 코드 i_pathshape.py.
- I.6 ⬜ (남은 형제들):
  - **shadow 결과 누적 대기** (가동 중 — 2025Q3+ 미확정/진위 선결, 수개월)
  - **수집 공백 백필** (5/1~6/6 — pool·정규화 최신화)
  - **[I] 밖 (root-level 형제)**: 다른 자산(SOL/BTC) 이식성 — 같은 파이프라인 재적용 (사용자 전략 결정) / 타 신호 결합

### [SOL] 유사도 거래 이식 (자산 일반성 검증) — [I] ETH 의 root-level 형제
핵심 질문: ETH 15bp/day = "ETH 한계" 인가 "방법 한계" 인가. SOL 이 더 강하면 방법 OK(ETH가 효율적이었음), SOL 도 같으면 코인 공통 한계.
- SOL.1 **데이터+인벤토리 ✅ 토대 OK** (2026-06-10):
  - SOLUSDT 이미 보유: OB 50레벨 1Hz **2025-11-01~2026-05-01, 182일 연속(공백0)**, trades 동일. ETH 와 겹치는 6개월 (ETH 181/182 커버). **형식 ETH 완전 동일 → 어댑터 불필요** (i_labeling SYMBOL 훅 추가).
  - 기초특성 (겹치는 12일): **통념 반전 — SOL 마이크로 변동성 ≈ ETH (rv 1.07x)**, 스프레드만 23x (틱/가격; 4h 엔 경미). 유동성 우려 기우 (depth 9.9x, 거래량 $1.3B/day). → "SOL 이 더 거칠어 edge 클 것" 근거 약함. 판가름은 'kNN 방향 edge 가 SOL 에 존재하나'.
  - ⚠️ caveat: SOL 182일 = 시기 적음 (폴드 1~2, 단독 OOS 약함). 겹침이 ETH 감쇠 후기(2026 약세) → **동일 182일에 ETH 재측정해 공정 대조 필수**. ETH+SOL 약한 multiple testing.
  - 산출물: research/sol/STAGE1_REPORT.md, characteristics.csv. 코드 sol_characteristics.py.
- SOL.2 **라벨링→유사도→70%→hit/net + 동일 182일 ETH 대조 ❌ (이 윈도우 양쪽 다 edge 없음)** (2026-06-10):
  - 파이프라인 전부 SOL 에 그대로 작동 (라벨 47, 축약 SOL 23차원/ETH182 21차원, 정규화 자체 재적합, whitening 각 45일).
  - **유사도 방향 구조 = 양쪽 존재**: 부호일치 SOL 0.712 / ETH182 0.740 vs random ~0.50 → 방법은 SOL 에 일반화 (닮은 상태=방향 구성 닮음).
  - **거래 edge = 동일 182일에 SOL·ETH 둘 다 thr0.70 net 음수** (SOL day -17.8 [-120,+67] n=17, ETH182 -7.9 [-35.8,+17.4] n=64). **full-period ETH 스타 4h 가 이 최근 윈도우선 ETH 도 -44.6** (감쇠 확정 — I.5 2026 약세 연장). ETH182 30m 만 +33.9(CI 0제외, n=21) but 6셀중1·full 과 불일치 = regime/노이즈.
  - 판정: "15bp = ETH 한계냐 방법 한계냐" → **시간/regime 한계** (둘 다 최근 감쇠). SOL 이 "방법 일반·ETH 효율적" 입증 못함. **통계 매우 약함** (182일, n 17~64, CI 거대, whiten 45일). multiple testing.
  - **함의**: I.5 "4h thr70 약하지만 진짜" 에 경고 — 최근 6개월 ETH 4h 음수 → 감쇠 진행 가능. **shadow 전향이 최종 판정 (선결).** SOL 확장은 새 edge 안 줌.
  - ⚠️ **I.7 에서 정정**: 이 "ETH 4h -44.6 = 감쇠" 는 **182일 작은 DB + 45일 whitening 인공물**이었음. full DB(1.135M, whiten 2023)로 같은 2025Q3+ 보면 4h **+51.8**(양수, n=17). → **감쇠 단정 철회, open.** DB 깊이가 최근 신호의 핵심. (단 부호일치 SOL 0.712 = 방법 SOL 일반화는 유효.)
  - 산출물: research/sol/STAGE2_REPORT.md, sim_sol/ sim_eth182/ lean_*.parquet. 코드 sol_analyze.py.
- SOL.3 ⬜ (이 윈도우 ❌ — 단 짧음): SOL 과거 백필(시기 확보) 시 재검 가능. 우선순위 낮음 (shadow 전향 선결).

### ⬜ 안 가본 큰 가지 (root-level 형제)
- **틱~초 HFT 영역** (MM tier 영역, latency 인프라 필요) — OBI/dobi 의 native 영역. H.1 도 여기로 수렴(선행=sub-second).
- **일~주 거시** (펀더멘털, on-chain — OB 너머 데이터)
- **VIP rebate tier 도달** (자본 규모, R&D 밖)
- **shadow as a service** (남이 만든 신호 follow — 의사결정 outsource)

---

## 살아남은 진짜 발견 (거래 edge 아닌 시장 구조)
이건 트리의 결론이 아니라, 탐색 과정에서 얻은 description:

- **scale-invariant wave** (크기만 다름, 거동 universal — silhouette 0.17~0.27)
- **mean-reversion universal but fee 1/14** (lag1 ACF -0.022, gross 0.43bp)
- **5m windows are monotone path** (range/net 1.21, AUC direction 0.64 — 묘사이지 edge 아님)
- **gross 존재** (Q0 oracle one_way 8.8~18.4bp 시기 안정 fee 8bp 초과 50%+)
- **진짜 벽**: gross 있음, **실현/방향예측이 벽** (entry timing + maker fill + selection bias)
- **MM 이 micro layer (5min 7bp 진동) 점유**, 우리는 큰 사건 (50bp+) 영역만
- **간극 천장 (2026-05-31)**: 5m oracle 11.8bp 중 causal(t≤0) 방향이 잡는 건 0.37bp (3%). 97% 가 정보-진입 간극 → "30s 덫" 의 정량형. 모든 실현경로 공통.
- **강한 OB imbalance → 5m 방향 (causal, 2026-05-31)**: |OBI|q5 gross +1.59bp, win 0.543, 4년 양수(2026 감쇠 +0.53). mark19 최초의 진짜 causal 방향신호 — 단 fee 미만 + fill 함정 미적용. native 영역은 더 짧은 horizon(HFT).

## 폐기된 틀린 결론 (부활 금지)
다음 문장은 함정의 산물이었음. 다시 쓰지 말 것:

- ❌ "5min~1d random walk" — 평균 함정 (PCA 0.27 만, t-SNE 0.42 / 객관 시기 안 봄)
- ❌ "gross 부족이 본질 벽" — Q0 reframe (gross 풍부, 실현/예측이 벽)
- ❌ "trailing 죽음 = 4번째 종료" — sample 3 단정 (41% 살아남음 / no-stop hold +8.9)
- ❌ "OB-only 효율적 시장" — 평균 함정 (조건부 VR 0.79, 시기별 진동 미반영)
- ❌ "ETH 1Hz native + selection-bias-without + 시기 안정 = 진짜 edge" (B 신호) — lookahead

## 6번 죽은 함정 family (반복 금지)
| # | 신호 | 함정 |
|---|---|---|
| 1 | 4h Direction +1.81 | day-boundary wrap-around lookahead |
| 2 | range-v2 +3.22 | 38% random Bernoulli fill 낙관 |
| 3 | trailing 5bp +13.54 | 5min OHLC 순서 가정 (intra-bar) |
| 4 | "trailing 죽음" 단정 | sample 3개 → 단정 (41% 살아남는 분포 무시) |
| 5 | B 30s +1.08bp | side=sign(t=30s) PnL=t=0 적용 (lookahead) |
| 6 | bigflow 라벨 | 큰체결 임계=당일 전체 q95 (일중 lookahead — truncation 테스트로 검출·수정) |

---

**마지막 업데이트**: 2026-06-10 (I.8 경로모양 ❌ — 과거길이 L×미래 H 83짝: 쏠림은 풍부(L=240 n=338)하나 전부 net 음수·hit~0.5(예측력 0), OOS 생존 0. why: 순간 라벨(ma_slope/macd/rsi)에 경로 이미 redundant. 순간 21차원 완승. 개선 8축 전부 닫힘 = [I] 최종형 재확정. shadow 가 감쇠 판정. 이전: I.7 multi-horizon, SOL.2)
