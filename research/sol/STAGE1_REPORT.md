# [SOL] 1단계 — 데이터 다운로드 + 인벤토리 + ETH 정합 보고서

**날짜**: 2026-06-10 · **목적**: SOL 데이터 토대 + ETH 정합 + 기초특성만. **라벨/거래 X.**
**핵심 질문 (이 가지의 이유)**: ETH 15bp/day = "ETH 한계" 인가 "방법 한계" 인가 → SOL 이 가름.

## 작업1 — 다운로드 범위 (이미 보유)
- **SOLUSDT 이미 존재**: OB 50레벨 1Hz **2025-11-01 ~ 2026-05-01, 182일 연속 (공백 0)**, 4.5GB. trades 182/182 커버. BTCUSDT 도 동일 기간 존재.
- ETH(2023-01-18~2026-06-09, 1198일) 와 **겹치는 6개월** (ETH 동일기간 181/182 커버 — 1일만 ETH 누락).
- ⚠️ **SOL 과거(2023~2025-10) 미보유**: ETH 장기 이력은 Tardis 소스 — SOL 확장은 추가 비용/시간. 현 182일 = ETH STEP=6 검증표본(203일)과 비슷 규모.

## 작업2 — 인벤토리 (실제 확인, 가정 X): **ETH 와 완전 동일 → 어댑터 불필요**
- OB: 203컬럼 (timestamp/update_id/sequence + bid/ask 0~49 price·size), ~86400행/일 — ETH 와 컬럼·구조 동일.
- trades: 11컬럼 (timestamp/symbol/side/size/price/tickDirection/trdMatchID/grossValue/homeNotional/foreignNotional/RPI) — **ETH 와 동일** (i_labeling 이 쓰는 timestamp/side/size 포함). side ∈ {Buy,Sell}, timestamp = epoch float.
- → **i_labeling.py 그대로 적용 가능** (SYMBOL=SOLUSDT 환경변수 훅 추가 완료).

## 작업3 — SOL vs ETH 기초 특성 (겹치는 12일 중앙값) — 유리·불리 단서
| 지표 | ETH | SOL | SOL/ETH |
|---|---|---|---|
| 가격 | $2,777 | $121 | — |
| **rv (per-sec, bp)** | 1.24 | 1.33 | **1.07x** |
| **spread (bp)** | 0.036 | 0.829 | **23x** |
| tick | $0.01 | $0.01 | 동일 |
| top-5 depth | $151k | $1,489k | 9.9x |
| 거래량 (notional/day) | $4.4B | $1.3B | 0.29x |

- **통념 반전**: SOL 이 ETH 보다 훨씬 변동적일 것이라 예상했으나 **1Hz 마이크로 변동성은 거의 동일 (1.07x)**. "SOL 이 더 크게 흔들려 폭 유리" 가설 **미지지** (적어도 초 단위).
- **스프레드 23x 넓음**: 같은 $0.01 틱이 $121 SOL 에선 0.83bp, $2777 ETH 에선 0.036bp. → **rv/spread (신호여지/마찰) = ETH 34.5 vs SOL 1.6** — SOL 은 마이크로 마찰이 변동성을 거의 다 잠식. 단 **4h 보유 전략엔 영향 작음** (gross 수십 bp vs spread 0.83bp; fee 11bp 가 지배).
- **유동성 우려는 기우**: SOL top-5 depth 오히려 ETH 의 9.9x, 거래량 $1.3B/day — fill 충분. "작은 시장" 우려 미지지.
- **fee 환경 동일**: non-VIP taker 5.5bp/leg 는 % 기준 코인 무관.

## 작업4 — 파이프라인 재사용 계획
| 구분 | 항목 |
|---|---|
| **그대로 재사용** | i_labeling (라벨 47개), i_acc_verify (truncation), i_reduce_norm (군집), i_simsearch, i_lean70_v2, i_thr_curve — SYMBOL 만 SOLUSDT |
| **SOL 자체 재적합** | 정규화 rolling med/IQR (스케일 다름), whitening fit (SOL 2023 없음 → 가장 이른 2025-11 fit), 대표 21차원 (재군집 — 다를 수 있음), thr (0.70 은 ETH 특수 — 재탐색) |
| **caveat** | SOL 182일 = 시기 적음 → walk-forward 폴드 1~2개뿐, SOL 단독 OOS 약함. ETH+SOL = 약한 multiple testing. 겹치는 기간이 ETH 의 **감쇠 후기(2026 약세 구간)** → ETH 도 이 윈도우로 재측정해 공정 비교 필요 |

## 종합 — 데이터 토대 OK, 라벨링 갈 준비 완료
- SOL OB/trades 형식 ETH 완전 동일, 182일 연속, 어댑터 불필요.
- **SOL 사전 전망 (정직)**: 통념과 달리 마이크로 변동성 ≈ ETH → "SOL 이 더 거칠어 edge 클 것" 근거 약함. 스프레드 마찰만 큼 (4h 엔 경미). → **"15bp 가 ETH 한계인가 방법 한계인가" 의 답은 'SOL 이 더 변동적' 이 아니라 'kNN 방향 edge 가 SOL 에도 존재하는가' 로 판가름** (자산 특수성 vs 방법 일반성). 다음: ETH 검증 코드로 SOL 라벨링 → 정확성 → 유사도 → 70% → hit/net, 단 **겹치는 동일 182일에 ETH 도 재측정** 해 공정 대조.
