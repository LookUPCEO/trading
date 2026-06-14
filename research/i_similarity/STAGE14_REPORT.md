# [I] 14단계 — 외부데이터 + Kelly + regime (순차) 보고서

**날짜**: 2026-06-14 · **코드**: i_ext_label.py, i_kelly.py, i_regime.py
**판정 요약: 셋 다 ❌ — 단일 4h(0.084%/day) 못 넘음. ①외부(funding)는 신호 파괴(hit 0.68→0.50), ②Kelly는 OOS 손실 증폭(파산위험), ③regime은 in-sample hit↑ 있으나 게이트하면 표본 소멸(test n=2). 0.084%→0.5% 는 새 정보/사이징/regime 어느 경로로도 도달 불가. 단일 4h 가 구조적 천장. 데이터 현실: liquidation/OI 과거 미수집 → 외부는 funding 만.**

## ① 외부 데이터 (funding) — ❌ 신호 파괴
- 데이터 현실: **funding 만 전체기간 보유** (liquidation/OI 과거 아카이브 없음 — 미수집). funding 2라벨(rate, 30일 z, causal 8h 설정값) → 21차원에 추가(robust z) → kNN 재계산.
| | n | hit | per-trade | 일수익 full | test |
|---|---|---|---|---|---|
| base21 (현행) | 79 | **0.684** | +90.5 | +8.40 | +2.90 |
| ext23 (+funding) | 115 | **0.496** | +3.2 | +0.43 | **-4.24** |
- **funding 추가가 hit 를 0.684→0.496(코인플립)로 파괴.** funding 은 느린 준상수(8h 블록, 37% 0.0001 고정)라 kNN 거리를 "같은 funding regime"으로 끌어 → 방향 유사도(edge 원천)를 희석. 다른 이웃이 뽑혀(n 79→115) 방향 예측력 상실. G/H(funding conditional fee미만, C.1)·사용자 회의 재확인. **외부 ≠ 새 유용 정보.**

## ② Kelly 베팅 — ❌ OOS 손실 증폭 (파산위험)
- 합의깊이 k → train hit/edge → 연속 Kelly f* (mean/var). k별 f*: k1 2.3x → k5 29.8x (n=9 train, 불안정).
| bankroll (OOS 2025Q3~, n=87) | 최종 | 일수익 | maxDD |
|---|---|---|---|
| 고정 1x | 0.712 | **-0.095%** | 30% |
| 1.0-Kelly (cap 5x) | 0.255 | -0.245% | **76%** (파산근접) |
| 0.25-Kelly (cap 2x) | 0.678 | -0.106% | 34% |
- **합의 k≥1 거래가 OOS 에서 음수** (고정 -28.8%) → Kelly 가 손실을 **증폭** (full -74.5%, maxDD 76%). 빈도 안 늘리고 크기만 키우니, 음수 신호엔 독. f* 가 n=9 train 의 mean/var 로 24~30x = 과적합·불안정. **9단계 파산위험 정량 확인.** (※ 단일4h test 는 +2.90 양수지만, k≥1 합의셋·4h hold 는 30m/45m 약신호 섞여 OOS 음수 — Kelly 대상 자체가 부적합.)

## ③ regime 게이트 — ❌ in-sample 패턴 있으나 게이트 시 표본 소멸
- regime 별 4h thr0.70 hit (full): **고변동 강함** — rv급증-hi 0.757(+169bp), atr-hi 0.727(+155), adx-hi 0.684(+123). "lean=고변동 순간"(stage5)과 정합 = 진짜 묘사.
- **게이트 OOS**: train 최고 regime(ATR-hi) → **test n=2** (hit 0.50, -103.7). 4h thr0.70 이 이미 희소(79건/851일)라 regime 한정 시 test 표본 소멸 → 일수익 순효과 측정 불가/음수.
- 고변동 regime 이 hit 높은 건 **묘사이지 tradeable 게이트 아님** (빈도 줄여 일수익 못 올림). mark18-R 분류 약함과 별개로, 희소성이 게이트를 막음.

## 종합 (작업5) — 셋 결합 미추구
- 각각 ❌ (외부 희석 / Kelly 증폭 / regime 소멸) → 결합은 실패의 복합 + multiple testing 가중일 뿐. 추구 안 함.

## 판정
1. **셋 다 단일 4h(0.084%/day) 못 넘음.** 새 정보(외부)·운용(Kelly)·필터(regime) 어느 것도 개선 없음.
2. **0.084%→0.5% 는 구조적으로 도달 불가** (in-data 10축 + 외부 + Kelly + regime 전부 소진). 단일 4h ~0.084%/day(one-way taker) 가 [I] 의 천장.
3. 정직한 묘사 1건: 고변동 regime 이 hit↑ (in-sample) — 단 희소성에 막혀 게이트 불가. (lean=고변동 재확인.)
4. **[I] 결론 확정**: 순간 21차원·thr0.70·고정 4h·단일 = 유일 운영점. 일수익 ~0.084% (목표 1% 의 1/12). edge 는 진짜(마찰 통과)이나 **빈도/크기로 키울 수 없음** = 구조적 소액.
- 다음: 개선 탐색 소진. **남은 건 감쇠 판정(shadow/실거래)** 뿐. 그 외 = root-level 다른 가지 또는 [I] 현 규모 수용.
