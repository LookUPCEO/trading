# Mark19 연구 규율 (절대 준수)

이 파일은 매 세션 자동 로드된다. 데이터를 만지는 매 작업 전 이 규율을 의식적으로 적용한다.

## 매 작업 첫 행동

1. `docs/SEARCH_TREE.md` 읽기 — 현재 위치, 형제 노드, 폐기된 결론 확인
2. 작업 후 트리 갱신 (노드 상태, 형제 추가) + commit
3. 발견 즉시 `BASECAMP.md` / memory 에 기록 (휘발 방지)

## 평균의 함정 금지 (최우선)

- 어떤 결론도 **"전체 평균 한 숫자" 로 내지 말 것**. 평균은 추세시기+박스시기, winner+loser, 다른 종류를 섞어 상쇄시켜 진짜 신호를 숨긴다.
- 모든 측정은: **분포(p10~p99) + 시기분해(객관적 P1~P5.x) + 조건부**.
- "universal/일관" 결론 전에 반드시 **객관적 시기분할 안에서 재확인**. 연도 경계(2024/2025/...)는 임의 함정. **rv_60d change-point 등 객관 분할** 사용.
- 한 측정(range/net 등)으로 "거동 같다" 단정 금지. **여러 측정 비교** (range/net, breakout%, ACF, jump, skew, ...).
- **eta² / sep_ratio** 등 effect-size 도 같이 (p-value 만 보지 말 것; n 크면 작은 차이도 유의).

## 표본 함정 금지

- **몇 개 sample (3개 등)로 결론 금지**. 전수 또는 통계적 충분 표본.
- 출력에 sample 3-5개만 보였다고 그것이 분포의 대표라고 가정 X. **항상 전수 분포 + percentile**.
- "대표 N개" 로 좁히지 말 것.

## 부분→전체 비약 금지

- 한 과정/한 측정 결과를 전체 결론으로 일반화 금지.
- **AUC ≠ tradeable** (AUC 0.58인데 PnL 음수 사례 있음, AUC 0.642 진짜인데 PnL lookahead 사례 있음). 판정은 **실제 PnL 분포 + fee 차감 후**로.

## Lookahead 금지 (5번 죽은 함정)

- 진입 시점에 존재하지 않는 정보로 진입/방향/필터 결정 금지.
- **PnL 시작 시점 = 진입 시점. side 결정 정보 시점 ≤ 진입 시점 필수**.
- 5min OHLC 의 high/low 순서 가정 금지 (1Hz 로 확인).
- day boundary wrap-around 금지 (raw 파일 안 다음날 첫 snapshot 제거 필수).
- **매 신호 cheat injection sanity** — future 정보 주입 시 비정상 양수 나오면 코드 정상; 그러나 cheat 결과 자체도 **버리지 말고 정보로 보존** (oracle gross 의 정보 가치).
- 매 PnL 계산을 다음 형태로 분해 검증:
  ```
  pnl = sign(side_signal_time_t_s) × (price[exit_time] - price[entry_time])
  단 t_s ≤ entry_time 인가? 아니면 lookahead.
  ```

## Fill / Fee 현실 금지사항

- maker fill "채워진다" 가정 금지. **실제 fill rate + selection bias 적용** (38% 가정이 실제 favorable 3.3% 였음).
- 청산 비대칭 반영 — **stop hit = taker 강제** (RT fee 더 큼).
- fee 는 결과 (실행/예측의), 고정 상수 아님. 단 **non-VIP 보수 기준** (maker +2bp, taker +5.5bp per leg; rebate 가정 금지).
- "Mixed 5.9bp" 류 단일 숫자 금지. 시나리오 (maker+maker / maker+taker / taker+taker) 표로.

## 가지치기 / 백트래킹 규율

- `SEARCH_TREE.md` 매 작업 전 읽고 후 갱신.
- **한 가지 막힘 ≠ 엣지 없음**. 부모 분기점 형제 다 막혀야 그 분기점 사망.
- 막히면 **형제로 백트래킹 (먼 점프 금지)**.
- "엣지 없음" / "R&D 종료" 은 **루트까지 전부 막혀야**.
- 단정 표현 금지: "effectively dead", "not tradeable", "R&D endpoint" 류 금지. **"이 가지에서 ❌, 형제 ⬜"** 형태로.

## 닫힌 목록 금지

- 파라미터 / 스케일 / 진입틀을 미리 몇 개로 박지 말 것.
- 공간으로 보고, **"왜 그 값인지" 원리로 좁히거나 체계적으로 훑기**.
- "대표 N개" 단어 자체를 의심.

## 흥분 금지

- promising 신호일수록 더 의심. **모든 promising 이 audit 에서 죽었다**:
  - 4h Direction +1.81 → wrap-around lookahead
  - range-v2 +3.22 → 38% fill 낙관
  - trailing +13.54 → 5min OHLC 순서
  - B 30s +1.08bp → side(t=30s)·PnL(t=0) lookahead
- 양수 나오면 자랑 전에 **lookahead / fill / 시기 / 표본 audit 먼저**.
- "8/8 시기 양수" 자랑 → 그 직후 lookahead 노출된 사례 있음. 시기 안정 ≠ 진짜.

## 죽으면 분해

- 한 가지 ❌ 됐을 때 "그냥 종료" 금지.
- **숨은 결론이 있을 수 있음**: cheat injection 의 +37bp 가 보물이었던 사례.
- 분포 (positive/negative/middle), 시기, 조건별로 분해 — 가린 발견 쫓기.

## 목표

- **일 1% 목표**. 단일 약한 결과로 목표 낮추기 금지.
- **진짜 edge 존재부터 확정** (키우기는 다음).
- "MM tier rebate 영역" 으로 도망가지 말 것 (자본 규모 R&D 밖).

## 시각화 / 원리

- 숫자 한 개로 압축 말고 **분포 / 공간 시각화** (사람이 볼 수 있게 저장).
- 결과마다 **"왜 이 숫자인가" 원리 가설**.
- 통계만 보고 원리 모르면 과적합 / 함정 구분 불가.

## 매 작업 commit

- 작업 전: 가설/방법 commit.
- 작업 후: 결과 / 발견 / 폐기 결론 / 트리 갱신 commit.
- commit 메시지에 "why" 포함.

## 커밋 메시지 형식 (참고)

```
<scope>: <one-line headline>

<context: hypothesis / method>
<key numbers (distribution, not single)>
<finding (what survives)>
<caveats (what wasn't tested / what could kill this)>
<next branch in SEARCH_TREE.md>
```

---

**마지막 업데이트**: 2026-05-31
**관련 파일**:
- `BASECAMP.md` — 프로젝트 상태, 발견 누적
- `docs/SEARCH_TREE.md` — 탐색 트리, 백트래킹 가이드
- `~/.claude/projects/-Users-mark/memory/` — 영구 메모리
