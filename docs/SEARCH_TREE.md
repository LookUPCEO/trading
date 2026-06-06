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
- I.2 **유사도/거리 (축약+정규화+검색검증) 🔵→✅ 통과** (2026-06-06):
  - **축약 47→21차원**: 정규화 후 spearman |r|>0.7 군집 medoid. PCA EVR 90%→17차원 — **1단계 "유효 ~10" 추정은 과소였음(교정)**. 정보손실: 탈락 25개 복원 R² min 0.533/med 0.862. spread_bp 제외(1틱 양자화→scale 폭발). 방향군 대표 전존.
  - **정규화 causal**: rolling 통계 = 과거 15 sampled days(달력~90일), **현재 day 제외 → lookahead 구조적 불가**. 부호라벨 scale-only(부호 100% 보존), 크기라벨 full robust-z. 연도 IQR비 2.56→~1.1 (rv_3600 1.41 잔존).
  - **시기 분포 (핵심)**: 정직 발견 — naive(raw47 global-z)의 시기쏠림이 우려보다 약했음(same-yr lift 1.19; 다수 라벨이 이미 bounded). 정규화+whitening 으로 recency 0.81→0.94, lift 1.19→**1.07** 일관 개선, 매치 연도분포 ≈ pool 구성 (전 연도 커버).
  - **검색 동작**: 과거 90분 경로 corr top50 **+0.30 vs random 0.00** (top>rnd 85%). **부호 일치(방향 닮음) 0.782 vs random 0.499** (부호 15차원 전부 0.76~0.91) → **유사도가 "방향 구성이 같은 상태"를 찾음**. N충분: pool 중앙 126k, rank100 거리 3.28 << pool 중앙 6.73.
  - **거리 척도**: whitened Euclid 가 시기중립 최선 → 3단계 기본 + cosine 교차확인.
  - 산출물: `research/i_similarity/` (STAGE2_REPORT.md, labels_norm_reduced.parquet 21차원, 검증 csv/png). 코드 `scripts/i_reduce_norm.py`,`i_simsearch.py`,`i_signcheck.py`.
  - ⚠️ caveats: 부호일치 0.78 = "상태 닮음"이지 미래방향 아님(이번엔 미래 안 봄). 203일 subsample — 3단계는 전체 1198일 전수 DB 고려. 일부 쿼리 잔존 시기쏠림(p10 0.57) → 3단계 판정은 시기분해 필수.
- I.3 70% 방향쏠림 건수 ⬜ (다음: whitened 21차원 공간 top-N 의 미래방향 쏠림 — fee·fill·시기 audit + 간극천장(causal 0.37bp) 대조 필수)
- I.4 fee 넘는 폭 건수 ⬜
- ⚠️ 주의(트리 맥락): G/H 에서 **간극천장** 확정(causal 방향 0.37bp, fee 미만). 유사도가 새 정보를 만드는 게 아니라 *기존 라벨의 비선형 조건부*이므로, 2단계+에서 "유사시점 방향쏠림"이 나와도 **fee·fill·시기 audit + 간극천장 대조** 필수 (promising 흥분 금지).

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

## 5번 죽은 함정 family (반복 금지)
| # | 신호 | 함정 |
|---|---|---|
| 1 | 4h Direction +1.81 | day-boundary wrap-around lookahead |
| 2 | range-v2 +3.22 | 38% random Bernoulli fill 낙관 |
| 3 | trailing 5bp +13.54 | 5min OHLC 순서 가정 (intra-bar) |
| 4 | "trailing 죽음" 단정 | sample 3개 → 단정 (41% 살아남는 분포 무시) |
| 5 | B 30s +1.08bp | side=sign(t=30s) PnL=t=0 적용 (lookahead) |

---

**마지막 업데이트**: 2026-06-06 ([I] 유사도거래 2단계 ✅통과 — 47→21차원 축약(정보손실 R² med 0.86), causal 정규화(드리프트 2.56→1.1, lookahead 구조불가), 유사도가 "방향 구성"을 봄(부호일치 0.78 vs 0.50, 경로 corr +0.30 vs 0). whitened Euclid 권장. 다음 I.3 70% 쏠림 — 간극천장 0.37bp 대조 필수. 거래/edge 결론 X. 이전: 1단계 라벨링 ✅)
