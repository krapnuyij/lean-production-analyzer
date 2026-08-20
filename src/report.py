"""AI Improvement Report generation for the Production LEAN Improvement Analyzer.

Numbers are never computed by the LLM. `build_report_context()` reuses the existing
src/metrics.py / src/analysis.py functions to compute every fact and figure first;
only that small, structured context dict is sent to the model. The LLM's only job
is to phrase a short, structured Korean summary from numbers it did not derive
itself -- it never sees the raw (date, line, process) dataset.

Provider: Google Gemini, via the official `google-genai` SDK -- the single LLM
provider used in this project (no multi-provider abstraction). The API key is read
from the GEMINI_API_KEY environment variable (optionally via a local .env file,
never hard-coded, never committed -- see .env.example).
"""

import json
import os

import pandas as pd
from dotenv import load_dotenv

from src import analysis, metrics

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_REQUEST_TIMEOUT_MS = 30_000
GEMINI_MAX_ATTEMPTS = 1

SYSTEM_PROMPT = """당신은 제조 공장의 LEAN 생산 개선 담당자를 돕는 리포트 작성 보조자이다.

다음 원칙을 반드시 지킨다.
- 제공된 분석 데이터(JSON)만 근거로 사용한다. 제공되지 않은 공장 상황, 설비 상태, 인력 문제를
  사실인 것처럼 만들어내지 않는다.
- 숫자를 새로 계산하거나 임의로 변경하지 않는다. 제공된 숫자만 인용한다.
- 원인과 결과의 인과관계를 단정하지 않는다. "~와 함께 개선되는 패턴이 나타났다"처럼 완곡하게 표현한다.
- Before/After 변화나 개선활동과 KPI 변화 사이의 인과관계를 암시하는 "~을 통해 개선되었다",
  "~로 인해 향상되었다" 같은 표현은 사용하지 않는다. 대신 "개선 후", "Before/After 비교에서",
  "~와 함께 변화하는 패턴이 나타났다"처럼 관찰된 동시 변화만 표현한다.
- Loss 변화는 "Loss가 개선되었다"보다, 실제 측정값을 그대로 가리키는 "Downtime이 감소/증가했다"
  같은 표현을 우선한다.
- 이 데이터는 synthetic(합성) 생산 데이터에서 계산된 것임을 인지하고, 실제 기업 사실인 것처럼 서술하지 않는다.
- 개선 제안은 확정적 지시가 아니라 "가능한 후속 조치"로 표현한다.
- 데이터에 없는 구체적인 설비 고장 원인, 인력 배치 등은 추측하지 않는다. 원인을 알 수 없는 경우
  "추가 확인 필요"라고 표현한다.
- 장황한 컨설팅 보고서가 아니라, 제조 현장 담당자가 빠르게 읽을 수 있는 간결한 한국어로 작성한다.
- Line, Process, Bottleneck, Cycle Time, Downtime, Changeover, Loss, UPH, KPI, Capacity, Pareto 등
  현재 대시보드에서 쓰는 용어는 번역하지 않고 그대로 사용한다.

반드시 다음 5개 섹션을 이 순서와 제목 그대로 사용해 Markdown으로 작성한다. 각 섹션은 2~4문장 또는
짧은 bullet list로 간결하게 작성한다.

### 현황 요약
### 주요 Loss
### 개선 효과
### 후속 개선 과제
### 다음 개선 우선순위

"다음 개선 우선순위" 섹션은 번호 매긴 목록으로 2~3개만 제안하고, 데이터 범위 안에서 확인 가능한
내용에 근거한다."""


class MissingAPIKeyError(RuntimeError):
    """Raised when GEMINI_API_KEY is not configured."""


def get_api_key() -> str | None:
    """GEMINI_API_KEY from the environment (or a local .env file), never hard-coded."""
    return os.environ.get("GEMINI_API_KEY") or None


def build_report_context(df: pd.DataFrame, line: str, process: str) -> dict:
    """Small, structured, JSON-serializable analysis context for the AI report.

    Every value is computed by reusing the existing metrics.py / analysis.py
    functions -- the same ones the other 3 screens use -- so nothing here is
    invented or recalculated with new logic. Percentages are pre-multiplied to
    plain numbers (e.g. 78.5, not 0.785) so the LLM never has to interpret a
    fraction vs. a percentage.
    """
    before_line_df = df[(df["line"] == line) & (df["period"] == "Before")]
    bottleneck_df = analysis.identify_daily_bottleneck(before_line_df)
    bn_summary = analysis.summarize_bottlenecks(bottleneck_df)
    primary_process = bn_summary.index[0]

    before_process_df = df[(df["line"] == line) & (df["process"] == process) & (df["period"] == "Before")]
    before_process_kpi = metrics.calculate_process_kpis(before_process_df).loc[process]
    pareto = analysis.calculate_downtime_pareto(before_process_df)

    kpi_table = metrics.calculate_improvement_kpis(df, line, process)
    before = kpi_table.loc["Before"]
    after = kpi_table.loc["After"]

    def pp(before_val, after_val):
        return round(metrics.calculate_improvement_delta(before_val, after_val, kind="pp") * 100, 1)

    def pct(before_val, after_val):
        return round(metrics.calculate_improvement_delta(before_val, after_val) * 100, 1)

    reason_comparison = analysis.calculate_downtime_reason_comparison(df, line, process)
    reason_diff = reason_comparison["After_avg"] - reason_comparison["Before_avg"]
    improved_reasons = reason_diff[reason_diff < 0].sort_values().index.tolist()
    worsened_reasons = reason_diff[reason_diff > 0].sort_values(ascending=False).index.tolist()

    return {
        "target": {"line": line, "process": process},
        "bottleneck": {
            "primary_process": primary_process,
            "bottleneck_days": int(bn_summary.iloc[0]["days"]),
            "bottleneck_total_days": int(len(bottleneck_df)),
            "bottleneck_share_pct": round(float(bn_summary.iloc[0]["share"]) * 100, 1),
        },
        "before_process_kpi": {
            "scheduled_good_uph": round(float(before_process_kpi["scheduled_good_uph"]), 1),
            "runtime_good_uph": round(float(before_process_kpi["runtime_good_uph"]), 1),
            "cycle_time_sec": round(float(before_process_kpi["avg_cycle_time_sec"]), 1),
            "downtime_minutes": round(float(before_process_kpi["avg_downtime_minutes"]), 1),
            "defect_rate_pct": round(float(before_process_kpi["defect_rate"]) * 100, 1),
        },
        "before_top_losses": [
            {
                "reason": reason,
                "share_pct": round(float(row["share"]) * 100, 1),
                "downtime_minutes": round(float(row["downtime_minutes"]), 1),
            }
            for reason, row in pareto.head(3).iterrows()
        ],
        "improvement": {
            "production_attainment_before_pct": round(float(before["production_attainment"]) * 100, 1),
            "production_attainment_after_pct": round(float(after["production_attainment"]) * 100, 1),
            "production_attainment_delta_pp": pp(before["production_attainment"], after["production_attainment"]),
            "scheduled_uph_before": round(float(before["scheduled_good_uph"]), 1),
            "scheduled_uph_after": round(float(after["scheduled_good_uph"]), 1),
            "scheduled_uph_change_pct": pct(before["scheduled_good_uph"], after["scheduled_good_uph"]),
            "cycle_time_before_sec": round(float(before["avg_cycle_time_sec"]), 1),
            "cycle_time_after_sec": round(float(after["avg_cycle_time_sec"]), 1),
            "cycle_time_change_pct": pct(before["avg_cycle_time_sec"], after["avg_cycle_time_sec"]),
            "downtime_before_min": round(float(before["avg_downtime_minutes"]), 1),
            "downtime_after_min": round(float(after["avg_downtime_minutes"]), 1),
            "downtime_change_pct": pct(before["avg_downtime_minutes"], after["avg_downtime_minutes"]),
            "defect_rate_before_pct": round(float(before["defect_rate"]) * 100, 1),
            "defect_rate_after_pct": round(float(after["defect_rate"]) * 100, 1),
            "defect_rate_delta_pp": pp(before["defect_rate"], after["defect_rate"]),
        },
        "loss_change": {
            "improved_reasons": improved_reasons,
            "worsened_reasons": worsened_reasons,
        },
    }


def build_report_prompt(context: dict) -> str:
    """User-turn content: the structured context as JSON plus a short instruction.
    No raw CSV rows are ever included here -- only this pre-computed summary."""
    context_json = json.dumps(context, ensure_ascii=False, indent=2)
    return (
        "다음은 Python으로 미리 계산한 생산 분석 결과이다. 이 수치만 근거로 리포트를 작성하라.\n\n"
        f"```json\n{context_json}\n```"
    )


def generate_ai_report(context: dict) -> str:
    """Call Gemini once with the structured context and return the Markdown report
    text. Raises MissingAPIKeyError if GEMINI_API_KEY isn't set -- callers should
    check `get_api_key()` before offering the generate button, but this is a
    defensive backstop."""
    api_key = get_api_key()
    if not api_key:
        raise MissingAPIKeyError("GEMINI_API_KEY가 설정되어 있지 않다.")

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=GEMINI_REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=GEMINI_MAX_ATTEMPTS),
        ),
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_report_prompt(context),
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.3),
    )
    return response.text
