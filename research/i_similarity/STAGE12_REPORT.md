# [I] 12단계 — 소액 라이브 준비: 인프라 + 안전장치 점검 보고서

**날짜**: 2026-06-14 · **코드**: i_live.py(실행기, DRY_RUN 기본), i_live_preflight.py · ⚠️ **실주문 0건 (점검만).**
**판정 요약: 안전장치 15/15 PASS (전부 DRY, 실주문 0). 주문 인프라(BybitClient)·신호경로(i_shadow.Engine bit-identical)·안전장치(kill/한도/dedupe/4h청산보장/정합) 준비됨. 단 실주문 전 메울 구멍 3개: ①주문 실패 retry/부분체결 핸들링 강화 ②라이브 신호→주문 WS 루프 통합 ③trade-only API 키 발급(출금X, 사용자). 구멍 메우고 별도 확인 후 실주문.**

## 작업1 — 주문 실행 인프라 (기존 live_bot/execution.py 재사용)
- BybitClient (V5 signed API) 존재: `place_market`(taker IOC — 11단계 taker 확정), `place_market(reduce_only=True)`(청산), `get_position`, `cancel_all`, `get_wallet_equity`. API 키/시크릿 env, testnet 지원.
- 진입 = taker market (fill 보장, 11단계 결론). 청산 = reduce_only market (포지션만 닫음, 신규 X).
- ⚠️ **구멍**: 현재 에러 핸들링 = `raise_for_status` (기본). 실주문 전 **재시도/부분체결/타임아웃 핸들링 강화 필요**.

## 작업2 — 포지션 추적 + 정합성
- 시스템 장부(Ledger): 포지션별 진입가/수량/방향/notional/**청산기한(deadline)** 기록, positions.json + ledger.jsonl.
- `reconcile()`: 거래소 `get_position` size vs 장부 qty 대조, 불일치(>0.01) 시 **KILL 파일 생성 + 중단** (청산 누락/중복 방지).

## 작업3 — 안전장치 (preflight 15/15 PASS, 실주문 0)
| 장치 | 검증 |
|---|---|
| 1회 주문 한도 (CAP_PER_TRADE $60) | 초과 거부 ✅ |
| 동시 포지션 상한 (MAX_CONCURRENT 3) | 4번째 거부 ✅ |
| 총 노출 상한 (CAP_TOTAL $180) | 초과 거부 ✅ |
| 중복 신호 dedupe (같은 day_horizon_min) | 거부 ✅ |
| **kill switch** (KILL 파일) | 진입 거부 + 전체청산 ✅ |
| **4h 청산 보장** (deadline 만료 강제청산) | 작동 ✅ |
| 손실 한도 ($60 = 총의 1/3) | 도달 시 진입 거부 ✅ |
| 출금 호출 없음 | 코드에 withdraw 부재 ✅ |
| LIVE_ARM 이중 게이트 | DRY_RUN=false + LIVE_ARM=yes 둘 다 필요 ✅ |

## 작업4 — 신호 = 실행 일치
- 신호 = `i_shadow.Engine` (동결 artifact, 6-1 에서 **replay bit-identical 0.00e+00 검증**). 실행기는 이 엔진 재사용 → 신호 변화 0.
- **가동 중 shadow 데몬(7일째)은 hot-patch 안 함** — 실행기는 별도 프로세스로 같은 엔진/artifact 사용 (수집 연속성 보존). 실주문 arm 시 라이브 WS 신호루프를 실행기에 통합 (shadow_daemon 패턴).
- DRY_RUN 기본 = paper, LIVE_ARM=yes+DRY_RUN=false = live. paper→live 전환 시 신호 동일 (같은 엔진).

## 작업5 — 소액 운용 계획
- **자본 $180**, 1회 명목 **$60** (× 동시 3 = 총 노출 $180), **레버리지 1** (감쇠 시 손실 확대 방지).
- 신호 = 4h thr0.70 결합 (30m/1h/4h 각 1포지션 가능), taker 진입/청산.
- **손실 한도 $60** (총의 1/3) 도달 시 자동 중단. kill switch 수동.
- 빈도: thr0.70 ~0.1~0.3건/day → **검증 수개월** (감쇠/마찰 실측 누적).
- 기록: ledger.jsonl (실거래) vs 백테스트 net 대조 → 감쇠 여부 판정.

## 판정
- **안전장치 준비됨 (15/15), 인프라 재사용 가능.** 실주문 0건으로 전 장치 작동 확인.
- **실주문 전 필수 보완 (구멍 정직히)**:
  1. 주문 실패/부분체결/타임아웃 **재시도 로직 강화** (현재 raise_for_status 기본).
  2. 라이브 **WS 신호→주문 루프 통합** (i_live.py 는 실행기+안전장치까지; 신호수신 루프는 arm 시 shadow_daemon 패턴으로 결합).
  3. **trade-only API 키 발급** (출금 권한 절대 X — 사용자 액션) + testnet 1회 리허설.
- **감쇠 = 손실 각오** ($180 한도 내). 마찰 통과(11단계)했으나 신호 감쇠면 손실 — 그게 이 라이브의 검증 목적.
- 다음: 위 3개 보완 → testnet 리허설 → 별도 확인 후 소액 실주문 시작. (이번 stage 는 점검까지.)
