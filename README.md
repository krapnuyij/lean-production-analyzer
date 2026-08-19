# Production LEAN Improvement Analyzer

제조업 생산혁신 / LEAN 직무 지원을 위한 1~2일 규모의 미니 포트폴리오 프로젝트이다.

## 프로젝트 목적

가상의 신발 제조공장 생산 데이터를 기반으로 다음 분석 흐름을 보여주는 것을 목표로 한다.

```
생산 데이터 → KPI 모니터링 → 이상 Line 탐지 → 공정 Drill-down
→ Bottleneck 식별 → Loss 원인 분석 → 개선 전/후 효과 측정 → AI 개선 리포트
```

기술적 복잡성보다 명확한 비즈니스 로직, 데이터의 논리적 일관성, 분석 결과의 설명 가능성을 우선한다.

## 현재 구현 범위

이번 단계에서는 아래를 구현했다.

1. 프로젝트 개발환경 세팅 (conda `lean-production`, Python 3.11)
2. 로컬 Git repository 초기화 및 GitHub private repository(`origin`) 연결
   (아직 commit/push는 수행하지 않은 상태이다)
3. Synthetic production dataset 생성 및 검증

Streamlit Dashboard, AI Report, ML 모델 등은 아직 구현하지 않았다.

## 데이터에 대한 안내

`data/production_data.csv`는 **실제 기업 생산 데이터가 아니다.** 분석 시나리오(LINE-B Stitching 공정의
Bottleneck과 LEAN 개선 효과)를 보여주기 위해 직접 설계한 **synthetic(합성) manufacturing dataset**이다.

- 대상: 가상의 신발 제조공장
- 생산라인: LINE-A, LINE-B, LINE-C
- 공정: Cutting, Stitching, Assembly, Finishing
- 기간: 총 30일 (Day 1~15 Before / Day 16 Improvement(transition) / Day 17~30 After)
- row 단위: 날짜 × 생산라인 × 공정 = 1 row → 총 30 × 3 × 4 = 360 rows
- random seed를 고정하여 동일 코드 실행 시 동일한 데이터가 재현된다.
- Day 16(Improvement)은 개선활동이 실제로 적용된 전환일이며, 이후 2일(Day 17~18)은 성능이
  After 수준으로 점진적으로 안정화되는 ramp-up 구간이다. **Before/After 비교 분석에서는 Day 16을
  제외**한다.

핵심 시나리오: **LINE-B의 Stitching 공정이 개선 전 Bottleneck**이며, Day 16의 LEAN 개선활동
(Changeover 표준화, 자재/공구 사전 준비, 공정 내 Line Balancing) 이후 생산성이 개선된다.

### Line throughput 모델 (단순화된 steady-state 구조)

이 데이터셋은 상세한 공장 discrete-event/WIP simulation이 아니라, **병목 공정이 Line 전체의
일일 생산량을 제한하는 단순화된 steady-state 모델**이다.

날짜 × Line 단위로:

1. `planned_qty`를 Line 단위 생산계획으로 독립 생성한다. (그날의 실제 capacity로부터 역산하지 않는다)
2. Cutting/Stitching/Assembly/Finishing 각 공정의 `cycle_time_sec`, `downtime_minutes`을 생성하고,
   `capacity_qty = operating_minutes * 60 / cycle_time_sec` (숨겨진 efficiency 계수 없음)로
   공정별 생산 capacity를 계산한다.
3. 해당 날짜 Line의 실제 생산량은 `actual_qty = min(planned_qty, 4개 공정 capacity의 최솟값)`으로
   결정되고, 같은 날짜/Line의 4개 공정 row가 이 값을 동일하게 공유한다. 즉 가장 느린 공정(병목)이
   그날 Line 전체의 산출량을 제한한다.
4. 공정별 `defect_rate`를 적용해 각 공정 row의 `good_qty`/`defect_qty`를 계산한다.

재작업(rework) 흐름이나 공정 간 WIP 버퍼는 모델링하지 않는다.

## 환경 설정

conda 환경(Python 3.11)을 사용한다.

```bash
conda create -n lean-production python=3.11 -y
conda activate lean-production
python -m pip install -r requirements.txt
```

## 데이터 생성 및 검증

```bash
python src/generate_data.py     # data/production_data.csv 생성
python scripts/validate_data.py # 데이터 일관성 검증 + KPI sanity check 출력
```

`scripts/validate_data.py`는 raw data로부터 다음 KPI를 계산한다. (CSV에는 저장하지 않는다.)

- `production_attainment = actual_qty / planned_qty`
- `defect_rate = defect_qty / actual_qty`
- `scheduled_good_uph = good_qty / (480 / 60)`
  전체 shift(8시간, downtime 포함) 기준 실제 output 속도. 병목/생산성 분석에서 기본으로 사용하는 UPH이다.
- `runtime_good_uph = good_qty / (operating_minutes / 60)`
  실제 설비 가동시간(downtime 제외) 동안의 output 속도. 보조 진단용 KPI이다.
- `downtime_rate = downtime_minutes / 480`

Line/Process 단위로 집계할 때는 일별 ratio의 단순 평균이 아니라 `sum(actual_qty)/sum(planned_qty)`
같은 합계 기반 weighted aggregation을 사용한다.

### Downtime reason 구조

`downtime_minutes`는 하나의 reason에 전부 귀속되지 않고, 다음 5개 컬럼으로 분해되어 있으며 그 합이
`downtime_minutes`와 정확히 일치한다.

`changeover_minutes`, `machine_stop_minutes`, `material_delay_minutes`, `quality_issue_minutes`,
`worker_absence_minutes`

`primary_downtime_reason`은 그중 minute이 가장 큰 reason이고, `primary_defect_reason`은 해당 공정 row의
대표 defect 원인이다(품질 Pareto 분석을 위한 event-level 구조는 이번 범위에서 만들지 않았다).

## 프로젝트 구조

```text
lean-production-analyzer/
├── .gitignore
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/
│   └── production_data.csv
├── src/
│   └── generate_data.py
└── scripts/
    └── validate_data.py
```
