# Mark19 BASECAMP

**Last updated:** 2026-06-28 (I.32 31 재검증 — 31 본질 맞음, "≈0"→"방향 54% 맞지만 fee 와 정확히 상쇄=본전, 방향우위 실재". 데몬fup=백테fup corr 0.961(모순해소, shadow 우호적=경계노이즈+소표본 행운). 라이브 백테처럼 거래 불가(+90.5=행운/레버리지). causal 진입 다 해봄=earliest 못 이김. 레버리지=위험만. 유일 판정=실측) // 이전: I.31확장 fup 강도별 ❌ — earliest 진입 fup max 0.762 갇힘(강fup 도달 불가), 진입 thr↑는 표본붕괴 OOS검증불가. fup↑→방향+크기↑는 in-sample 실재이나 OOS 표본0. 레버리지·선별 둘 다 marginal 극복 실패. 거래가능·OOS edge ≈0, 유일 판정=실측) // 이전: I.31 earliest-crossing 재검토 — 0.084%/+90.5 헤드라인 상당부분 측정 인공물(stride-10 91분위 행운+비거래가능 mid-run). 라이브 실제 OOS=+0.26bp/day CI[−34,+74]=0과 구분 불가. [I] 진짜 edge 미확립. 레버리지 흔들림. 유일 판정=실측. 죽음 단정은 X) // 이전: I.30 롱전용 vs 롱+숏 — 숏=음수EV 데드웨이트(6단계 정합), 롱 전용 권고(거래당↑·구현 단순). 정직: 라이브 진입방식선 롱도 marginal — 헤드라인 +90.5는 stride/표본 산물, 최종판정 라이브/shadow. 라이브 숏게이트 확인 대기)
**Status:** 🔥 [I] 실거래 ARMED 수정본 (PID 54043, 4h 정각·thr0.70·단일·1x, qty quantize 적용). shadow(PID 14414) 정상 누적(outcome 6건 hit 0.83). 감쇠 = shadow 주도구 + 이제 실거래도 체결 가능
**Primary goal:** 일 1% 수익률 알고 트레이딩 봇

⚠️ **운영 주의**: 실거래 데몬 **PID 54043**(수정본, ARMED, $180/레버1/$60손실한도/CAP$60per trade). 다음 thr0.70 신호 시 실주문(~$48 명목) 나감 — 첫 체결 모니터. shadow PID 14414 별도. sleep 비활성. auto-restart 없음(재부팅 시 수동 재기동). kill: `touch research/i_similarity/shadow/live/KILL`.

---

## 🔬 2026-06-28 — I.32: 31 재검증 (shadow vs 백테 모순) — 31 맞음, "본전" 으로 교정

- 본인 의심 2개 검증: 31 틀렸나 + 라이브 백테처럼 가능한가.
- **모순 해소(핵심)**: 데몬fup vs 백테fup 4995분(6/07~6/15) **corr 0.961·평균차 0.000 = 같은 신호**(31 안 틀림). shadow 우호적 6~7건 = 0.70 경계 ±0.01 노이즈+소표본 행운(데몬이 m984 데몬fup0.701/백테0.640 우연히 잡아 +147 승자, 더 나은 신호 아님).
- **라이브 진입=earliest**(MAX_CONCURRENT=1, 31 가정 맞음). **작업3 causal 진입 다 해봄**(높은기준 OOS붕괴/지속 te0/상승중 +1.44 CI 0포함) — earliest 유의하게 못 이김.
- **작업4 평균함정 교정(핵심)**: hit(방향) 0.538=진짜 약한 우위. gross +6.9 vs fee 11. **방향우위0.038×|move|154≈12bp≈fee → 본전**(신호 진짜지만 fee 와 상쇄). +90.5=stride 91분위 행운, 무dedup +53=67% 겹침 레버리지(거래불가).
- **판정**: 31 본질 맞음, "≈0 무작위"→"방향 54% 맞지만 fee 와 정확히 상쇄=본전, 방향우위 실재". 라이브 백테처럼 불가. 레버리지=본전 증폭=위험만. 돈화=fee↓(역선택) or 큰움직임 선별(도달불가). **유일 판정=실측**(데몬=백테 확인됨). 죽음/삶 단정 X.

---

## 🔬 2026-06-28 — I.31확장: fup 강도별 거래당 ("적게 크게") ❌

- 본인 질문: 강한 fup(0.85+)만 골라 거래당↑ → earliest marginal(−4.1) 극복?(레버리지 아닌 선별).
- **핵심 구조**: earliest 진입 fup = max 0.762(p50 0.711) — kNN 합의 점진상승, 0.85 도달 전 이미 0.70 진입·4h 블록. **고fup@진입 0.85+ = 0건**(라이브가 강fup 진입 기회 없음).
- 진입 thr↑: 표본붕괴(0.80 n7/te0) OOS 검증불가, 검증되는 0.75 te −197.9(n4 노이즈). 6-3/6-5·9 빈도붕괴 재현.
- 작업4: fup↑→hit(0.571→0.598)+|move|(175→259)↑는 in-sample 실재(방향+크기)이나 OOS 표본0 검증불가.
- **판정**: "적게 크게" ❌ — 강fup 영역 도달불가/검증불가. 레버리지(16)·선별(이번) 둘 다 marginal 극복 실패. 죽음 단정 X(fup-레벨 구조 in-sample 실재)지만 거래가능·OOS 부분 ≈0. 유일 판정=실측.

---

## 🔬 2026-06-28 — I.31: earliest-crossing 재검토 (0.084% 진위) ⚠️ 헤드라인 인공물

- 30단계 발견(라이브 earliest 진입 롱 marginal) 정면 검토. dense(매분 2024+).
- **작업1 stride**: 매분 무dedup +53.0(n645). **stride-10 +98.7=매분 분포 91분위 행운(n62)** — phase 마법 아님, 소표본 낙관(부트스트랩).
- **작업2 phase**: fup 레벨↑=단조 강함([.70,.72)+28/[.75,.80)+134). earliest(라이브)=최약 순간 −4.1. +53의 강함=mid-run 고fup(one-way 거래불가). confirm in-sample +128(n18)이나 OOS 표본0.
- **작업3 thr↑**: in-sample 개선이나 OOS 붕괴(0.75 te−198)=6-3/6-5 과적합 재현.
- **판정**: 0.084%/+90.5=stride 행운+비거래가능 분 포함 인공물. **라이브 실제(earliest·OOS)=+0.26bp/day CI[−34,+74]/trade=0과 구분 불가**(약양수·고분산, 연도 2024+16.7/2025−34.9/2026−22.9). shadow 6건=2026 우호적 소표본. **레버리지 0.5%(16) 흔들림**(~0 base 증폭=위험만). 롱전환=음수EV 제거이지 수익원 아님. **죽음 단정 X**(점추정 약양수, fup-레벨 구조 in-sample 실재). **유일 판정=라이브/shadow 실측**(백테 CI 한계). [I] 진짜 edge 확립 미달.

---

## 🔬 2026-06-28 — I.30: 롱 전용 vs 롱+숏 (숏 제거 효과)

- 숏 6단계(25~28-1) 데드 확정 → 숏 빼면? dense(매분=라이브 운영점) causal 독립 4h thr0.70.
- **정직성(방법론 정합)**: 숏 A stride-10 +60.9(n17 플룩)/B dense −12.4/C dense-causal −14.3. 롱도 A +98.7(n62)→C **−4.1**(n225). earliest-crossing(라이브 진입)=최약. **+90.5/0.084%일 헤드라인=stride-10 소표본+유리 진입phase 산물**.
- 거래당: 롱전용 −4.1 > 양방향 −10.4(숏 빼면↑). 숏 비중 46%, 숏 거래당 −14.3(음수EV). 일수익 full 롱전용 −1.07>양방향 −4.93(+3.86), OOS te 롱 +0.26<양방향 +1.94(숏 test +6.0=n4플룩, CI 다 0포함=무의미).
- **판정**: 숏=데드웨이트 → **롱 전용 권고**(거래당·구현 단순·6단계 정합). 일수익 우위 noise라 단정 X. **롱도 honest measure 에선 marginal → 최종 판정 라이브/shadow.** 라이브 숏게이트 반영은 사용자 확인 후.

---

## 🔧 2026-06-28 — I.29: 실거래 버그 수정 + 리허설 강화 + 재arm ✅

- 점검 발견 버그 2개 수정: **#2(치명)** qty round(.,3)=0.035 step 위반 → `quantize_qty()` step floor(한도초과 방지)+minQty/minNotional(실 명세 qtyStep 0.01/minQty 0.01/minNotional 5). **#1(잠복)** `get_order_by_link` 실 BybitClient 추가(realtime→history)+`get_instrument_spec`. daemon 기동 spec fetch, place_entry quantize.
- **리허설 강화(I.13 거짓통과 재발 방지)**: Mock 이 실 step/min 검증(0.035 거부)+인터페이스 정합(실 클라 메서드 완비, get_wallet_equity 누락도 잡아 추가)+과거 실패 3건 재현. **21/21 PASS**(12→21), preflight 15/15.
- 검증: 실 API spec fetch 정상, 과거 3건(6/19숏·6/25롱·6/28숏) 모두 qty 0.03 명목 $47~51 valid. 재arm 전 거래소 포지션 **0**·equity **$188.32 불변**(체결 0 failed safe 확인).
- **재arm**: 구 49338 클린종료(orphan 0, shadow 14414 무영향) → 수정본 **PID 54043 ARMED**(DRY=False ARM=True, spec 로깅, WS 연결, KILL 없음, ledger 0). 다음 thr0.70 신호 시 실주문 모니터.

---

## 🔧 2026-06-28 — 운영 점검: shadow 건강, 실거래 주문 실행 깨짐 발견

- 두 데몬 생존(shadow 20d21h / live 11d18h, CPU·MEM 정상, KILL 없음).
- **실거래: 신호 3건(6/19 숏0.27, 6/25 롱0.73, 6/28 숏0.30) 발화했으나 전부 주문 실패** → 실주문 0은 '신호 대기' 아니라 '실행 불가'. 돈 손실 0(체결 0, 잔고 $188.32 불변 = failed safe). **6/25 롱 fup0.733(핵심 edge) 포함 3건 누락.**
  - 버그#2(치명): qty round(.,3)=0.035 등 → Bybit step 0.01 위반 Qty invalid. 버그#1(잠복): get_order_by_link 실클라 부재(멱등 무력). 근본=리허설 Mock 괴리(12/12 거짓확신).
- **shadow 정상**: outcome 6건(고유), 방향 hit 5/6=0.83, 평균 net +33.7bp, 공백 0 — 감쇠 주도구 건강(n 무의미). 백테 0.68 과 어긋남 없음.
- sleep 비활성·auto-restart 없음. **조치(qty 수정+get_order_by_link 추가+재arm)는 사용자 확인 후.** 산출물: OPS_CHECK_2026-06-28.md.

---

## 🔬 2026-06-22 — I.28-1: 하락 특화 라벨 + kNN(닮음, 우리 진짜 방법) ❌ (방법 무관, 시장 본질 확정)

- 본인 지적 타당: 27/28 은 GBM(예측), 우리 방법은 kNN(닮음). 4h 상승=kNN. → 28 하락 10라벨을 kNN 으로 다시(공간 3: 21거울/하락10단독/21+10결합).
- **핵심 정렬도**: 이웃 down합의 ↔ 쿼리 실제 down **corr −0.001~−0.002 ≈0** — 닮은 하락 전조 상황의 미래가 하락에 안 쏠림(상승 kNN 부호일치 0.78·hit 0.68 과 극명 대조, stairs vs elevator).
- 3 공간 전부 전 horizon·임계 net 음수, Bonferroni 생존 0. 하락 특화 라벨이 거울 대비 개선 0. 소표본 고hit(30m≤.30 n24 0.708)은 test 회귀=25 플룩 재현.
- **GBM(28 AUC 0.51)·kNN(28-1 정렬 0.00) 두 방법 일치** → 하락 예측불가는 모델 탓 아님. 큰 하락 발생은 사전 미시상태와 무관(돌발).
- 하락 분기 6단계 모두 ❌(25→26→27→27-1→28→28-1). 유일 미탐: 청산/OI(H 분기 ⬜). **4h 양방향 합산 재확정.** 라이브 49338 유지(검증만).

---

## 🔬 2026-06-22 — I.28: 하락 특화 라벨 신설 ❌ (라벨 부족 vs 시장 본질 가름 → 시장 본질)

- 27 한계(있는 47=거울) → 하락 전조 10라벨 신설(1초 OB/체결 파생, causal): 지지붕괴/비대칭호가/하방변동성/매도공격성. i_labeling 미변경(별도 i_downlab.py, 라이브 안전).
- **작업1**: 6개가 기존 47과 진짜 다른 정보(corr<0.2: bid_dep_chg·book_thin_asym·sell_spike) — 표현 부족 일부 사실.
- **작업2 핵심**: 그러나 down AUC 안 올림 — 47old 0.509 → 47+10 0.510(Δ+0.001=노이즈), 새 라벨 단독 0.507(무작위). 거래화 short hit 0.490·net−11·실패.
- **판정: 라벨 부족 아니라 시장 본질.** 새 down 정보 존재해도 예측력 0 → 큰 하락은 OB/체결 사전상태에 안 새김(효율+급락 돌발, 27/27-1 일관).
- **유일 한계**: 청산 데이터 미수집(H 분기 ⬜) — 청산 캐스케이드는 미검. "OB+체결 미시구조로는 하락 예측불가"가 정확.
- 하락 분기 5단계 모두 ❌(25→26→27→27-1→28). 빈도/일수익 0. **4h 양방향 합산 재확정.** 라이브 49338 유지(검증만).

---

## 🔬 2026-06-22 — I.27-1: 하락 단기 분단위(5~50분) ❌ (급락 가설 반증, 하락 예측불가 최종)

- 감사(본인 지적 적중): 27 shortest=30m, 26=15m → **5/10/25/40/50m 진짜 미탐**. dense stride=1(825k)로 5~50분 8개 촘촘 채움.
- 작업2: 전 셀 net 음수. 강한 down 단기(10m hit 0.595/20m 0.599=약한 진짜 방향성)이나 gross +1~3 ≪ fee.
- **작업3 핵심(급락 가설 반증)**: **down |move|=up 의 ~절반**(5m 22 vs 47, 50m 57 vs 113bp). kNN down-lean 은 급락 아닌 작은 하락(grinding) 고름 — 진짜 급락(elevator)은 사전 미식별. 짧은 하락=정확도↓(0.55)+폭↓(절반) → fee/|move| 벽 더 강함(예외 아님).
- 작업4: Bonf-24 생존 0. 25단계 15m/20m 0.66 = dense 0.576/0.596 = stride10 플룩 확정.
- **하락 예측불가 최종 확정(분단위, 25→26→27→27-1 전부 ❌).** 빈도/일수익 0. **4h 양방향 합산 재확정.** 라이브 49338 유지(검증만).

---

## 🔬 2026-06-17 — I.27: 하락 전용 시스템 (처음부터 독립) ❌ (하락 예측불가 확정)

- 26 한계(down=4h상승 거울) 본인 지적 타당 → 3축 자유: horizon cross-day 30m~2d(긴쪽), 표현 47라벨+GBM 최대자유, 임계 0.60~0.70. 파이프라인 sanity UP-4h hit 0.661/net+98.7 재현(이웃선택 교정).
- **작업1+3**: 전 horizon gross<fee·net음수. 단기 30m 약한 방향성(hit 0.57 train&test=엘리베이터)이나 gross+3.8<<fee. **긴 horizon 악화**(2d hit 0.449 코인미만)="내려갈때도 계단" ❌. Bonferroni-22 생존 0.
- **작업2(표현)**: OOS AUC(down) 0.51~0.52(무작위), 확신상위10% short net음수. **하락 선행지표(OBI/flow 매도압력) 부재** — 표현 병목 아님, 하락 자체가 예측불가.
- **왜 비대칭**: 큰상승=모멘텀지속(사전 미시상태 식별=stairs, hit0.68)/큰하락=급락(무특징=elevator, hit≈0.50)+강세장 drift. 구조적.
- **하락 분기 완전 종결(25 거울→26 표본보강→27 전용).** 빈도/일수익 0. **4h 양방향 합산 재확정.** 라이브 49338 유지(검증만).

---

## 🔬 2026-06-17 — I.26: down 전면 재검증 (표본 보강) ❌ (보강 후 down 약함 확정)

- 25단계 "down-4h-thr0.70 test n=4 판정불능"은 본인 지적대로 못 본 것 → **dense stride=1 검색(현 artifact 825k 쿼리)**로 독립 n **17→190(11배)** + 임계완화 0.30/0.35/0.40.
- **down-4h-thr0.70**: dense n=190 → **full hit 0.484(코인플립 미만), net −14.3**. 연도 2024 0.391/2025 0.595/2026 0.500 = regime 의존. **25단계 0.765 = n=17 플룩 확정.**
- 임계×horizon 24셀 전부 gross<fee·net 음수·hit 0.48~0.60. 강한down 단기 hit 0.58~0.60(엘리베이터 일부진실)이나 gross<<fee. **Bonferroni-24 거래가능 셀 0.**
- ⚠️ **방법론 인공물**: 강도순 독립greedy=선택 lookahead(거래불가) → 위양성. 시간순(causal)=−12.2 = full −15.5 일치, 무조건 공매도(−11)와 차이≈0 → down-lean 방향정보 0. (4/5/25 무영향).
- **왜**: 강세장+drift≈0 → down-lean 이 하락 코인 이상 못 짚음. 구조적. **4h 양방향 합산 재확정.** 라이브 49338 유지(검증만). down 분기 종결.

---

## 🔬 2026-06-17 — I.25: 방향별 최적 horizon (롱/숏 비대칭) ❌ (비대칭 진짜·거래화 불가)

- 방법: 기존 per_query 2파일 병합(추가검색X), up-lean fup>=.70/down-lean fup<=.30, net=dir*frq-fee, train→test OOS, Bonferroni 방향2×horizon12=24.
- **엘리베이터 가설 부분진실**: down 단기(15~20m) hit 0.667/0.652 > up 0.583/0.588 = "하락 빠름" 실재. **but 거래불가** — down 단기 전부 gross<fee(작은 움직임, fee/|move| 벽) + OOS 불안정(15m tr0.72→te0.50).
- down 비단조: 단기 hit>0.5 → 2~3h **역신호(0.31/0.33, gross -61/-16)** → 4h 0.765 but **test n=4 판정불능**. up 4h 단조 최적(+98.7).
- **둘 다 train 최적=4h** → 방향분리=현행 4h합산 동일(롱4h/숏4h, test +51.8bp/day, one-way 충돌 0). **빈도·일수익 개선 0.** Bonferroni-24 OOS 생존 0.
- 4/5(fee벽)/20(down 약함)/9~10 일관 — 방향 나눠도 edge 는 강합의·큰움직임(4h)에만. **4h 양방향 합산 재확정.** 라이브 49338 유지(검증만).

---

## 🔬 2026-06-17 — I.24: hold 분단위/조건부 ❌ (과적합, 4h 정각 재확정)

- A 분단위(3h~5h, 13 hold): train net 5h까지 단조↑(+62→+107)=2024-25 강세 drift 과적합. OOS: train최적 5h → test +50.6 < **4h test +51.8**. Bonferroni 13, OOS 4h 못 넘음.
- B 조건부: 신호별 hold 차등 없음(강/약/고변동 전부 긴 hold=drift). 조건부 룰 test +46.1 < 고정 4h +51.8.
- train '긴 hold=더 많은 net'은 강세 drift 유산, OOS(약세) 미전이. 6-1/6-2 'winners run=4h 고정 최선'·19 비단조 재확인.
- **4h 정각 재확정. hold 축 소진. 운영점 4h·thr0.70·단일·1x 불변.** 라이브 49338 유지(검증만, 즉시교체 X).

---

## 🔬 2026-06-16 — I.23: stale artifact 갱신 (단 June regime ≈ old)

- 구 norm 1/31~4/30(6주). raw 6/7~6/15 수집(공백 5/1~6/6).
- **핵심: June(6/7~15) vol regime ≈ 구 norm (rv 0.95~0.98x, bounded 동일)** → stale norm 신호 거의 안 왜곡(우려보다 작았음).
- 갱신: 재라벨(1207일) → reduce_norm(reps 동일 21차원, 신호 불변) → artifact rebuild. **norm 2/9~6/15, DB to 6/15, big_thr 2.51→2.88**.
- 라이브 안전 교체(실주문 0): 구 48983 클린종료(orphan 0) → 새 artifact 재arm PID 49338. 재검증 fup 0.519, 에러/KILL/ledger 0.
- 5/1~6/6 공백 잔존. 감쇠 실측의 깨끗한 토대 마련.

---

## 🔬 2026-06-16 — I.22: ★ 실거래 ARMED (라이브 주문 데몬 가동)

- arm 전 점검: live_bot/.env trade-only 키, **mainnet(TESTNET=false), equity $188.32, 포지션 0**.
- **i_live_daemon.py = 검증 부품 결합**: 신호 i_shadow.Engine(bit-identical) / 실행 i_live(안전 15/15) / 주문 OrderManager(리허설 12/12). shadow 수집(PID 14414)과 별도·persist 안 함.
- DRY 배관 검증(라이브 WS): 실시간 fup240(votes 91), signal 0, 에러 0, ledger 0 ($0 위험).
- **arm: DRY_RUN=false LIVE_ARM=yes (PID 48983)** — 매분 fup, thr0.70 발화시 taker 진입→4h reduce_only 청산, 매분 reconcile. 안전장치(손실한도 $60/kill/deadline) 활성.
- **실주문 0건** (빈도 ~0.09/일 → 첫 주문 며칠뒤). 첫 체결/슬리피지 실측은 발화 시.
- ⚠️ **caveat**: 정규화 윈도우 ~4/30(6주 stale)→첫 실주문 전 갱신 권장. 자율 실거래(머신 상시가동 필요). 감쇠 실측 = 모든 R&D의 최종 판정.

---

## 🔬 2026-06-16 — I.21-1: DTW 자기 운동장 ❌ (공평 검증 후에도)

- 본인 지적(4h 방향에만 묶음 불공평) 타당 → 다른 horizon/예측대상/강일치 공평 검증.
- horizon: 거래가능 5m~4h 전부 DTW hit ≤ 유클리드 (1d만 높으나 stage20 중첩 인공물).
- 변동성: DTW corr 0.216 ≈ 유클리드 0.223 (동률). 강한 모양일치: 방향 hit 0.510(전체 0.504) — smoke 0.579는 n=38 노이즈.
- 단 강일치 |4h| 128>107bp = **변동성↑(B branch 크기, 방향 아님)**. DTW 가 기존 vol 라벨 대비 더 주는 것 없음.
- **모양 정보 4회 확인** (8 redundant/18 희석/21 4h비정렬/21-1 자기운동장 무용) — 시간적 모양은 방향에 순간 21차원 못 이김.
- **4h 유클리드 순간상태 최종, DTW 종결, 신호 표현 탐색 완전 종료.** 키우는 길=레버리지(16, 감쇠 의존). 감쇠 판정(shadow) 유일 미해결.

---

## 🔬 2026-06-16 — I.21: DTW 파동 모양 ❌ (새 정보이나 방향 비정렬)

- Sakoe-Chiba band DTW(과거 60분 정규화 경로), 21차원 top-300 후보→DTW 재순위(자체 numpy 배치, 651s).
- redundancy: **겹침 0.37 = 새 정보** (8단계 redundant 와 다름).
- **방향 비정렬(핵심)**: 동일 조건 DTW hit 0.571 ≤ 유클리드 0.585, test -1.29 vs +2.19. 워핑 모양은 다른 이웃 찾으나 4h 방향 더 못 맞춤.
- **모양 정보 3회 확인**: 8(경로 redundant)/18(봉 희석)/21(DTW 비정렬) — 모양·경로·워핑 전부 순간 21차원 못 이김. **방향 edge=순간 상태 구성, 시간적 모양 아님.**
- 4h 유클리드 순간상태 확정. **신호 표현 탐색 종결.** 빈도+거래당+표현 모두 소진. 키우는 길=레버리지(16, 감쇠 의존). 감쇠 판정(shadow)이 유일 미해결.

---

## 🔬 2026-06-16 — I.20: 1d 기각 (자본중첩 인공물), 4h 최종 확정

- **결정타(자본중첩)**: 1d hold·one-way 단일포지션 비중첩 강제 → n 261→79, hit 0.625→**0.519(코인)**, per-trade +162→**-8**, 일수익 +49.7→**-0.76(음수)**.
- 261 신호 93일 몰림(2.8/일) → **같은 1d 움직임 중복 계상** + up-편향(237/24)이 부풀린 인공물. CI 중첩 전에도 Bonferroni [-33,+197] 0포함.
- **19 정정**: 19 "audit 생존"은 kNN 투표 독립성만, 거래 실행 중첩(같은 움직임 반복진입) 못 봄 = 진짜 killer (19 caveat #3 적중).
- 1d 너머 2d/3d/5d 노이즈. **4h 만 비중첩(0.09/일)에서 hit 0.66·CI 0제외 생존 → 4h 단일 = [I] 최종 확정.**
- 빈도(9축)+거래당(horizon) 전부 소진. 키우는 길 = 16단계 레버리지(4h, 감쇠 의존)뿐. 감쇠 판정(shadow)이 유일 미해결.

---

## 🔬 2026-06-15 — I.19: 긴 horizon (4h 너머) — "4h 단조 peak" 거짓, 1d 새 lead

- 본인 지적: 4h는 5분~6h 중 최선, 8h~며칠 미탐. cross-day 연속가격(1198일, gap 1개 2023-12-13 마스킹), 각 horizon 자체 lean.
- **비단조**: 4h(hit.68,+90) → 6h/8h(hit 0.44/0.40 음수 반전) → 12h(.61,+32) → **1d(.64,+162)** → 2d/3d 노이즈.
- **1d = [I] 가장 강한 새 lead**: hit 0.64, per-trade +162, 일수익 full +49.7/test +13.2(≫4h +8.4/+2.9), **2024/25/26 전부 양수**, 독립성(창 비겹침)+gap audit 생존.
- **미확정 (4h 확정 유지, 흥분 금지)**: day-CI [-3,+168] 0포함(per-trade 변동 거대 95% 미달), 6h/8h 반전 미설명(비매끄러움=취약 가능), 자본중첩으로 full +49.7 과대(1d hold 단일포지션), 꼬리위험.
- **다음**: 1d shadow 전향 추가 로깅(재시작 시) → 무감쇠+CI 확정. 6h/8h 반전 원리 분해. 4h 운영점 유지.

---

## 🔬 2026-06-15 — I.18: MTF 봉 라벨 + AND/OR ❌ (봉=새 정보이나 결합 무용)

- 본인 지적 적중: 봉은 redundant 아님 (게이트 통과 0/14 >0.9, 30m/60m corr 0.27~0.57, 8단계 경로와 다름). "봉=horizon 중복" 단정은 비약이었음.
- **그러나 결합 두 방식 모두 ❌**: ①봉을 유사도 kNN 추가 = hit 0.684→0.576·test +2.90→-1.40 (14단계 funding 처럼 방향 희석) ②봉 AND/OR 모멘텀 = 봉 방향 hit 0.44~0.45 역예측(MR), AND k≥8 도 0.451.
- **새 교훈**: 비중복 정보(funding/봉)라도 kNN 추가하면 이웃선택 흐려 hit↓. 21차원은 방향 task 에 튜닝된 공간 — 더 많은 정보 ≠ 더 나은 방향 kNN.
- **단일 4h thr0.70 (0.084%/day, 1x) 확정. 빈도 개선 전면 소진.** 키우는 길 = 16단계 레버리지(감쇠 의존)뿐.

---

## 🔬 2026-06-15 — I.17: 방향 구조 패턴 ❌ (약신호 사전식별 3번째 실패)

- thr 낮춤(0.55~0.65) + 방향 구조 (5m,10m,30m,1h,4h lean 부호 조합). 구조×thr 183개 (Bonf 분모 183).
- **OOS 95% 생존 1개뿐** (우연 기대 ~4.6 미만, Bonferroni 미생존) = 노이즈. train 양수 36개 대부분 test 음수.
- **눌림목(단기↓장기↑) 직관 ❌**: n=497 hit 0.445(<0.5) train -5.2 test -36.9 = 오히려 역신호. "all-up"=합의 k5 재탕.
- **약신호 사전식별 3회 실패** (6-4 부분집합/10-2 보완동의/17 구조 — 다른 축 전부 OOS 전멸). why: edge는 강합의에만, 약신호域은 recoverable signal 부재.
- **단일 4h 0.084% 1x 천장 재확정. 빈도 개선 경로 전부 소진.** 키우는 길 = 16단계 레버리지(감쇠 의존)뿐.

---

## 🔬 2026-06-15 — I.16: 레버리지 — 0.5% 조건부 도달, 감쇠하면 독

- 프레임: 빈도 포기, 한 방 최대화. 파산확률 판정. 4h thr0.70 **단독**(14단계 합의 k≥1 분리).
- **4h 단독 = OOS 양수(+52bp)·f* 안정**(train 18.7x/OOS 16.4x) — 14단계 합의(OOS -28.8%·불안정)와 결정적 차이. 14단계 Kelly 실패는 신호(합의) 탓이지 Kelly 탓 아니었음.
- 몬테카를로(full): 3x +0.36%/파산0.5%, **5x +0.74%/파산4.0%** → 처음으로 0.5% 도달. OOS(감쇠): 5x +0.30%/파산6.7%.
- **감쇠 민감도(핵심)**: 절반 감쇠 5x 파산 18.5%, edge 소멸 순손실+파산 57%. 레버리지=증폭이지 생성 아님.
- 합의선별 일수익 경로 아님(k5 95일1건). i.i.d. 부트스트랩 군집/꼬리 과소(실 파산>표).
- **판정: 레버리지=지속하면 답(0.5%) 감쇠하면 독 — 지속 미확정.** 확인 전 저배율(2~3x +0.17~0.36%/파산~1%) 신중한 천장, 5x+ 강확인 후.
- **0.084% 천장은 1x 기준이었음 교정.** 레버리지로 0.5% 가능성 열림 — 단 shadow/소액 라이브로 감쇠 확인이 선결.

---

## 🔬 2026-06-14 — I.14: 외부+Kelly+regime 셋 다 ❌ (0.5% 구조적 불가)

- 데이터 현실: liquidation/OI 과거 미수집 → 외부 = funding 만.
- **① 외부(funding)**: 신호 파괴 — 21+funding kNN hit 0.684→**0.496**(코인플립), test +2.90→-4.24. funding 느린 준상수(8h, 37% 0.0001)라 거리를 funding regime 으로 끌어 방향 유사도 희석. G/H 재확인.
- **② Kelly**: 합의 k≥1 거래 OOS 음수(고정 -28.8%) → Kelly 손실 증폭(full -74.5% maxDD 76% 파산근접). f* n=9 train 24~30x 불안정. 9단계 파산위험 정량 확인.
- **③ regime**: 고변동 in-sample hit↑(rv-hi 0.757)=묘사(lean=고변동) but 게이트시 test n=2(4h 희소→표본 소멸). tradeable 아님.
- **판정: 0.084%→0.5% 새정보/사이징/regime 어느 경로도 불가.** 단일4h ~0.084%(목표 1/12)=구조적 천장. edge 진짜(마찰통과)이나 빈도/크기로 못 키움.
- **[I] 개선 탐색 전면 소진.** 남은 건 감쇠 판정(shadow 9일째/실거래)뿐.

---

## 🔬 2026-06-14 — I.13: 실주문 구멍 ①② 메움 (실주문 0, Mock 리허설 12/12)

- ⚠️ 실자금/실거래소 주문 0건 (전부 Mock/DRY).
- 재시도/부분체결/타임아웃/거부 + **멱등성**(order_link_id 상태조회 후 재전송, 중복 방지). 실패를 성공 오인 안 함(미기록).
- 신호→주문 루프: i_shadow.Engine → 안전장치 → OrderManager taker → ledger 실체결 → 4h reduce_only 청산.
- **실설계 발견(중요)**: one-way 퍼프 넷팅 → 30m/1h/4h 동시·다방향 보유 불가 → **라이브=단일 4h(MAX_CONCURRENT=1)**. stage 7~10 운영점과 정합.
- **실거래 일수익 = 단일 4h ~8.4bp/day(0.084%)** — 결합 0.15%는 one-way 실행 불가 (정직한 다운그레이드).
- preflight 15/15 + 리허설 12/12. 남은 건 사용자(trade-only 키 발급 + 실 testnet 1회)→별도확인 후 소액($180/레버1/손실$60).

---

## 🔬 2026-06-14 — I.12: 소액 라이브 준비 (안전장치 15/15 PASS, 실주문 0)

- ⚠️ 점검만, 실주문 0건 (전부 DRY_RUN).
- 안전장치 preflight 15/15: 1회/총노출/동시 한도, dedupe, kill switch, 4h 청산보장, 손실한도 $60, 출금호출 없음, LIVE_ARM 이중게이트.
- 인프라: live_bot BybitClient 재사용 (taker place_market/reduce_only 청산/get_position/cancel_all). 신호=i_shadow.Engine(bit-identical). shadow 데몬 hot-patch 안 함(별도 프로세스).
- 계획: $180 자본, $60/trade ×3, 레버 1, 손실한도 $60, 빈도 0.1~0.3/day → 검증 수개월.
- **실주문 전 구멍 3**: 주문 retry/부분체결 강화, 라이브 WS 신호→주문 루프 통합, trade-only API 키(출금X 사용자)+testnet. 메우고 별도확인 후 실주문.
- 감쇠=손실각오($180 한도). 마찰 통과했으나 감쇠면 손실 — 그게 라이브 검증 목적.

---

## 🔬 2026-06-14 — I.11: 실거래 마찰 ✅ 통과 — 0.15%/day 환상 아님

- ETH 4h thr0.70 **per-trade +90bp >> 마찰 합계 ~0.3bp** (슬립 0.05 + funding 0.17 + 지연 ±2 노이즈).
- taker net +92.5/trade, 결합 일수익 **15.1→15.24** (test 8.1→11.24), CI[+28,+123] 유지 — 거의 불변.
- **구조적 강건**: large-move 방향신호(gross +100bp)라 마찰<<gross. 과거 죽은 micro 신호(OBI +1.3/range-v2 +3.2)는 gross≈마찰이라 maker fill 에 죽었으나 이건 다름.
- taker 현실적(fill 보장), maker 불필요. caveat: 시장충격 미모델(소액 무시), 감쇠 별개(shadow), SOL 스프레드 크나 edge 無.
- **판정: [I] 백테스트 환상 아님. 실거래 가능성 = 감쇠 여부에만 달림 — shadow 전향이 유일 관문.**

---

## 🔬 2026-06-14 — I.10-2: 약신호域 보완 ❌ 완전 종결 (구조적 불가)

- 본인 지적 타당(10/10-1 약신호 통째 버림, 6-4 이익 45% 실존). 10짝×밴드3=30셀 정면.
- **약신호 오답 실제 독립** (corr [0.50,0.55) 0.02~0.07) — 단 거긴 각 horizon hit<0.5 → 약밴드 동의해도 hit 0.46(코인 미만), 30셀 전부 net 음수, train 최선 → test -57.1 (6-4 전멸 재현).
- **구조적 종결**: edge(hit>0.5)는 강신호域에만(거긴 중복), 오답독립은 약신호域에만(거긴 edge無) → 동시성립 영역 부재 = fee 넘는 보완 구조적 불가.
- 6-4(부분집합)+10-2(보완동의): 약신호 이익 45% 는 실재하나 사전식별 조건 없음 2회 독립 확인.
- 보완 분기 전체 종결(10→10-1→10-2). horizon=같은정보 시간변형. **in-data 개선 완전 소진 = [I] 최종형.**

---

## 🔬 2026-06-14 — I.10-1: 보완 불가 확정 (1h&4h 0.75 = n=4 노이즈)

- 본인 지적 타당(강신호 오답상관 n 작아 단정 이름). 10짝 전수 재검.
- **1h&4h 0.75 = n=4(3/4) 무의미** — thr0.65 로 표본 늘리니(동시 133) **0.38(<0.5=중복)**. 표본 충분 짝 전부 redundant (A틀릴때 B맞 0.05~0.27).
- 짝 union: full +10~11 이나 **test 전부 음수**(질 희석 < 단일4h test 2.90). 교집합=합의 k2 재탕(n=14).
- **fee 넘는 오답독립 짝 없음** — horizon=같은정보 시간변형 (강쏠림 같이, 틀릴때도 같이). 1h&4h=다중검정 인공물.
- 10단계 보완불가 표본보강 후 확정. **in-data 보완/조합 완전 소진 = [I] 최종형.** shadow 전향이 유일 미해결.

---

## 🔬 2026-06-14 — I.10: 선택적 보완 ❌ — 강신호域 중복으로 보완 불가

- 조건부 강점 일부(4h=early 0.68, 30m=late/range, 1h=hivol) + 전체 오답상관 낮음(0.06~0.11).
- **그러나 거래 강신호域(thr0.70)은 중복**: 30m 틀릴때 4h 맞을확률 0.30(같이 틀림). 오답 독립은 fee 못넘는 약신호域뿐 → 활용 불가.
- R-union 빈도↑(368 vs 79)나 hit↓(0.56<0.67)·일수익 +7.72<8.40·**test -10.5** (질 희석). R-route=현행 재포장.
- 9(맞을때 겹침=교집합)·10(틀릴때도 겹침=중복) → **horizon=같은정보 시간변형 확인** (7단계 89~113x 귀결).
- **일수익 10축(정의/청산/thr×3/multi-h/경로/합의/보완) 전부 닫힘 → 순간21차원·thr0.70·고정4h·결합 = [I] 최종형.**
- **유일 미해결 = 감쇠 여부 (shadow 전향, 7일째 outcome 2).** in-data 신호/표현 탐색 소진.

---

## 🔬 2026-06-14 — I.9: 합의 두 길 ❌ 일수익 — 단 합의=질 신호 확정

- horizon 5개 병합. **합의깊이 k 단조: hit 0.58(k1)→0.78(k5), net -0.3→+284bp** (k4,5 CI 0제외, n 9~10) — 질 신호 확정.
- **A(드물게 크게) ❌**: 3+ per-trade +140 거대하나 희소(0.05/day) → size1 +6.4 < 현행 15.1, 목표=8x레버=파산위험.
- **B(합의+낮은thr) ❌**: 합의된 0.65 도 fee 못넘음(3+ +2.5 CI 0). 양수셀=고strength 동어반복.
- test 현행(8.1) 넘는 셀 0. **일수익 9축(정의/청산/thr×3/multi-h/경로/합의) 전부 닫힘 = [I] 최종형.**
- takeaway: k=신뢰도지표 → shadow k 로깅(다음 재시작) 하면 confidence 전향 검증 가능.
- 감쇠 여부 open — shadow 가 유일 미해결 판정 (7일째, outcome 2건, 06-10~14 신호 0).

---

## 🔬 2026-06-10 — I.8: 경로 모양 ❌ — 순간 21차원이 경로를 이김

- 과거길이 L 12개{3~240분} × 미래 H 7개 = 83짝. 모양 요약 5특징, 정규화, whiten 2023, t≤0.
- **빈도는 오히려 풍부** (L=240,H=5 n=338 ≈ 순간 408) — 단 **전부 net 음수·hit 0.49~0.60** (예측 0). OOS 생존 0 (Bonferroni 47).
- why: 명시적 경로는 순간 라벨에 redundant (ma_slope_5~240=기울기, macd=모멘텀, rsi=위치) + OB/flow 맥락 상실. "오는 길" 정보 이미 순간이 포착.
- **개선 8축 (정의/청산/thr평균/분포/소수점/multi-horizon/경로) 전부 ❌** → 순간 21차원·thr0.70·고정4h·결합 = [I] 최종형 재확정.
- 감쇠 여부 open — **shadow 전향이 유일한 미해결 판정** (선결). 새 표현/파라미터 탐색 소진.

---

## 🔬 2026-06-10 — I.7: multi-horizon ❌ + SOL.2 감쇠 정정 + 합의 질강화

- **상관**: thr0.70 신호 동시발생 89~113x·동방향 100% = 중복(독립분산 아님). 자본 1/3분산 일수익 +5.0 < 단일4h +8.4. 빈도 못 풂.
- **SOL.2 정정 (중요)**: full DB(1.135M, whiten 2023)로 최근 2025Q3+ **4h +51.8 양수**(hit 0.765) — SOL.2 의 -44.6 은 **182일 작은DB 인공물**. **감쇠 단정 철회, open** — DB 깊이가 최근 신호 핵심. (단 부호일치 SOL 0.712 = 방법 SOL 일반화는 유효.)
- **합의 발견**: 3+ horizon 동방향 hit 0.81 net+156(n=16), 2+ hit 0.73 — 질 강화(진짜) 단 빈도 감소.
- 빈도 한계 6축 (정의/청산/thr평균/thr분포/소수점/multi-horizon) 모두 닫힘 → 단일 4h 운영점 유지.
- **감쇠 여부 = open, shadow 전향이 최종 판정 (선결).** 합의-필터 shadow 부가기록 후보.

---

## 🔬 2026-06-10 — SOL.2: 동일 182일 SOL vs ETH — 둘 다 edge 없음 (감쇠 확정)

- 파이프라인 전부 SOL 작동. **부호일치 SOL 0.712 / ETH182 0.740 vs ~0.50** → 유사도 메커니즘 SOL 일반화 (닮은 상태=방향구성).
- **거래 edge: 같은 182일(2025-11~2026-05) SOL·ETH 둘 다 thr0.70 net 음수** (SOL -17.8 n=17, ETH182 -7.9 n=64, CI 0포함).
- **full-period ETH 스타 4h 가 이 최근 윈도우선 ETH 도 -44.6** = 감쇠 확정 (I.5 2026 약세 연장). 30m 만 +33.9(n=21, regime/노이즈).
- 판정: "15bp = ETH한계냐 방법한계냐" → **시간/regime 한계** (양 자산 최근 감쇠). SOL 이 "방법 일반·ETH 효율" 입증 못함.
- ⚠️ 통계 매우 약함 (182일, n 17~64, CI 거대). **함의: I.5 '4h 약하지만 진짜' 에 경고 — shadow 전향이 최종 판정.**

---

## 🔬 2026-06-10 — I.6-5: thr 소수점 — 점프 ~0.70 실재, 0.70 확정 (이해 > 개선)

- rolling 승률(윈도300): 0.60~0.70 평탄 노이즈(~45%, med 음수) → **0.70 onset 상승 → 0.72 plateau ~60%**. 6-4 점프 실재 (coarse 인공물 아님), step 아닌 ramp.
- OOS: test 0.685~0.70 음수(41%, med-11) / 0.70+ 양수(57%, +18) — **더 일찍 진입 = OOS 손해**. 일수익도 0.70 이 전체·test 최선.
- 빈도 개선 0 (사용자 예측 "이해 우세" 적중). 표본한계 정직 (소수점 ±2tick 불확실, "0.70 미만 못 씀"만 강건).
- **일수익 4.5축 (정의/청산/thr평균/thr분포/thr소수점) 전부 닫힘** → 현행 = ETH 최종 운영점.
- shadow 전향 2건 (06-09 net -9.5/+147.85, n=2 무의미). 다음: shadow 누적 / 백필 / [I]밖 SOL 이식 (사용자 전략).

---

## 🔬 2026-06-07 — I.6-4: thr 분포 (평균 함정 교정) — 0.70 은 분포의 질적 점프

- 저밴드 (0.60~0.70) 승률 43~46% 평탄, median -5~-8 — 이익거래 45% 실존 ('평균 음수≠전부 손해' 적중).
- **0.70 에서 도약: 승률 58.3%, med +17.8** — 평균 곡선 아래 실체 = 임계 점프. "70% 합의" 직관 정합.
- 사전등록 48셀: train 양수 5 → **OOS 0/5 생존** (최강 후보 +7.0 → -13.6). 빈도↑=질↓ 못 깸.
- → 빈도 한계 분포 수준 확정. 일수익 4축 (정의 6a/청산 6-1·2/thr평균 6-3/thr분포 6-4) 전부 닫힘.

---

## 🔬 2026-06-07 — I.6-3: thr 곡선 ❌ 개선 없음 — 빈도는 신호의 본질

- 사전등록 곡선+OOS: 결합 thr0.70 train +18.99 → test +8.08bp/day **둘 다 1위** (plateau 0.69~0.72).
- fee 하한 0.67~0.68 (5단계 예측 적중). **thr<0.67 = 빈도 늘수록 손해** (0.60: -300bp/day).
- 1h 단독 train 0.69 → test 4위 (단일 horizon 선택 과적합 실증 — 결합이 강건).
- **일수익 3축 (닮음정의/청산/thr) 전부 닫힘.** 최선 = 결합 thr0.70 = 15.1bp/day = 목표의 30%.
- 다음: shadow 누적 (2025Q3+ 미확정 해소 선결), 백필, 구조분해(사전등록 후), [I]밖 (타 신호 결합).

---

## 🔬 2026-06-07 — I.6-2: 재예측 갱신 ❌ (진짜 의도 구현) — 들고 가라 재확정

- 매분 동일엔진 재예측: 방향유지=hold(fee 0)/전환=flip(만 fee 5.5×2)/소멸=청산. 사전등록 V1/V2/V3.
- **V1 연장+전환 +73.8 ≈ FIXED +74.7 동률** — flip 단 2/79회: 반대 thr70 이 4h 내 사실상 안 옴.
- **V2 신호소멸청산 -0.2 (hold med 8분)** — lean=고립점 → 즉시청산화 (사전등록 예측 적중). V3 +22.6.
- 일수익 +8.45 vs +8.40 (빈도효과 0). **원리: 신호 정보는 진입 순간에 응축** — 직후 중립 재예측은 정상 소멸.
- 청산 개선 합계 8규칙 전패 (6-1 가격5 + 6-2 재예측3) → **고정 4h hold 확정. 일수익 병목 = 진입 빈도.**

---

## 🔬 2026-06-07 — I.6-1: 능동청산 ❌ (들고 가라) + shadow 가동 ✅

- A: 사전등록 5규칙(반전/소멸/익절/손절) 전부 고정 4h hold 패배 — FIXED +74.7 최고, 손절 +12 최악.
  why: lean=고변동 순간 → 중간 트리거 다 걸림. **신호 가치 = 4h 끝까지 (winners run).**
- B: shadow 데몬 가동 — replay 일치검증 **0.00e+00 (bit-identical)**, 매분 fup240 기록 + thr70 신호/4h outcome.
  1Hz 수집 재개 겸용 (4/30 중단 복구). caveats: 정규화 5주 stale (백필 옵션), 유효신호 6/8 UTC~, 0.09건/일.
- 모니터: `research/i_similarity/shadow/logs/daemon.log`, `outcomes.jsonl`. 머신 상시가동 필요.

---

## 🔬 2026-06-07 — I.6a 닮음 정의 탐색 ❌: 과적합 방어 작동, 21차원 유지

- 사전등록 31개 정의 (의미 5군 조합), 선택 train/판정 OOS 엄격 분리.
- train 상위 V/V+O+C/O → **OOS 전멸** (V -36, O -57, VOC +13 < 베이스 +23.6). "train 좋음 = 과적합 기본값" 실증.
- 부분공간은 빈도도 죽임 (이벤트 408→51~63, 일수익 +8.1 → -3.6~+1.0bp/day).
- why: 70% 쏠림의 질 = 추세+변동성+호가+체결+캔들 **동시 일치** — 좁히면 가짜 이웃.
- → 21차원 = 최선 "닮음" (단순한 것이 생존). 일수익 개선은 능동청산/시간대 구조로. **다음: shadow 전향검증.**

---

## 🔬 2026-06-07 — I.5 walk-forward+단조: 4h 가 진짜에 최근접, 30m 은 정정 하향

- walk-forward: 전 쿼리 구조적 OOS (pool prefix+룰 사전지정). 폴드: **4h 5/5 양수** (2026 +55), 1h 3/5, 30m 2/5.
- **누적 net CI 교정** (net 직접·day-mean): **4h +74.7bp Bonf99.5% [+8.3,+146.7] 유일 단독생존**. 결합 408건 +19.5 [+2.0,+38.2] 생존. **⚠️ 정정: stage4 '30m net CI 0 제외'는 gross/net 혼동 — 30m 은 95%도 ✗**. 2025Q3+ 단독 미확정 (점추정 +23.6 동부호 — 표본한계 우세, 단정 금지).
- **thr 매끄러운 단조** (fee 교차 ~0.67) = 구조. **horizon 비단조** — 2h/3h 골(edge 0.07, 열린질문), 4h 돌출은 confounder 역방향 확인 (불리한 시간대에서 달성 = 진짜).
- **why**: fee 11bp = 5m |move| 65% → 4h 10% (짧은 h 구조적 불가) + hit-edge 0.06→0.37 동반상승 (폭×예측 둘 다) + lean = 고변동 순간.
- 부수관찰 (사후, 결론 아님): 늦은 시간대(US 세션) lean 강함 (30m/1h hit 0.71).
- → I.6: **shadow 전향검증 최우선**, 능동청산, down-lean/2h3h골/시간대 구조 분해, 빈도확대(사전등록 후만).

---

## 🔥 2026-06-07 — I.3v2+I.4: 사상 첫 fee 초과 후보 (정직 한계 포함)

- 독립 기준 교정 (사용자 지적): day당1개 → **미래 창 비겹침** (같은 day |Δt|≥horizon). 쏠림 재측정 — 부풀림 없음 (thr70 0.16~0.25% vs null ×15~25).
- **hit rate (구조적 OOS)**: thr70 30m/1h/4h hit 0.64/0.64/0.68, gross +16/+45/+102bp, **net(T+T 11bp) +5.1/+34.0/+90.5bp, day-cluster CI 전부 0 제외** (n=190/139/79).
- audit 통과: cheat injection 정상, outlier 아님 (med 기준 유지), drift 수집 아님 (분기벤치≈0), 군집 약함. 누적 net 전기간 우상향.
- 간극천장 0.37bp 와 모순 아님 — 천장은 전수·고빈도 틀, 이건 희소(0.2%) 조건부 per-trade +16~100bp.
- **정직 한계 (edge 확정 아님)**: test 단독 CI 0 포함 (n 소표본), short(down-lean) 약함 (1h down hit 0.54), 1h 2026 음수, **일수익 ~0.15%/day < 목표 0.5%**, 다중검정 형식보정 미적용.
- 죽은 것: thr65 전부 (fee 미달), thr70 5m. → I.5: walk-forward 강건성 / 능동청산 / down-lean 분해 / shadow 라이브.

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
