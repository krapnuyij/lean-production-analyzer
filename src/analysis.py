"""Bottleneck and downtime-loss analysis for the Production LEAN Improvement Analyzer.

Operates on the raw (date, line, process) grain -- process capacity, cycle time, and
downtime are genuine per-process measurements, so no deduplication is needed here
(unlike the Line-level KPIs in src/metrics.py).
"""

import pandas as pd

DOWNTIME_REASON_COLUMNS = [
    "changeover_minutes", "machine_stop_minutes", "material_delay_minutes",
    "quality_issue_minutes", "worker_absence_minutes",
]
DOWNTIME_REASON_LABELS = {
    "changeover_minutes": "Changeover",
    "machine_stop_minutes": "Machine Stop",
    "material_delay_minutes": "Material Delay",
    "quality_issue_minutes": "Quality Issue",
    "worker_absence_minutes": "Worker Absence",
}


def add_capacity(df: pd.DataFrame) -> pd.DataFrame:
    """process_capacity = operating_minutes * 60 / cycle_time_sec (units/day)."""
    df = df.copy()
    df["capacity_qty"] = df["operating_minutes"] * 60.0 / df["cycle_time_sec"]
    return df


def identify_daily_bottleneck(df: pd.DataFrame) -> pd.DataFrame:
    """Per (date, line): the process with the lowest production capacity that day.

    This is a data-driven determination, not a hard-coded label -- whichever process
    has the smallest `capacity_qty` for that date/line is the bottleneck.
    """
    cap = add_capacity(df)
    idx = cap.groupby(["date", "line"])["capacity_qty"].idxmin()
    bottleneck = cap.loc[idx, ["date", "line", "period", "process", "capacity_qty"]]
    return bottleneck.rename(columns={"process": "bottleneck_process"}).reset_index(drop=True)


def summarize_bottlenecks(bottleneck_df: pd.DataFrame) -> pd.DataFrame:
    """Count of days / share each process was the bottleneck within the
    (already filtered, e.g. one line + one period) `bottleneck_df`."""
    total_days = len(bottleneck_df)
    counts = bottleneck_df["bottleneck_process"].value_counts()
    summary = counts.rename("days").to_frame()
    summary["share"] = summary["days"] / total_days
    return summary.sort_values("days", ascending=False)


def calculate_downtime_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Total downtime minutes per reason for the given (already filtered, e.g. one
    line + period + process) `df`, sorted descending, with share and cumulative
    share -- the basis for the Pareto chart and the top-N loss insight."""
    totals = df[DOWNTIME_REASON_COLUMNS].sum().sort_values(ascending=False)
    totals.index = [DOWNTIME_REASON_LABELS[c] for c in totals.index]
    pareto = totals.rename("downtime_minutes").to_frame()
    grand_total = pareto["downtime_minutes"].sum()
    pareto["share"] = pareto["downtime_minutes"] / grand_total
    pareto["cumulative_share"] = pareto["share"].cumsum()
    return pareto


def build_analysis_summary(line: str, period: str, process_kpis: pd.DataFrame, pareto: pd.DataFrame) -> str:
    """Rule-based (non-LLM) Korean text summary of the Bottleneck Analysis screen,
    entirely from computed numbers: the bottleneck process, how it differs from the
    other processes' average, and the top downtime loss reasons.

    `bottleneck_process` and the joined reason names are runtime-determined by the
    data, so no alternating Korean particle (은/는/이/가/을/를/와/과) is ever attached
    directly to them -- each is followed only by the invariant "의", or by a fixed
    Korean noun ("원인", "퍼센트") that carries the particle instead. This keeps the
    sentence grammatical no matter which Process or reason combination comes back.
    """
    bottleneck_process = process_kpis["scheduled_good_uph"].idxmin()
    bn = process_kpis.loc[bottleneck_process]
    others = process_kpis.drop(index=bottleneck_process)

    top_n = min(2, len(pareto))
    top = pareto.iloc[:top_n]
    top_share = top["share"].sum()
    top_names = ", ".join(top.index.tolist())

    lines = [
        f"{line}의 {period} 기간을 분석한 결과, {bottleneck_process}의 Process Capacity가 가장 낮게 나타나 "
        f"주요 Bottleneck으로 확인되었다.",
        "",
        f"다른 Process 평균과 비교한 {bottleneck_process}의 주요 지표:",
        f"- Scheduled UPH: {bn['scheduled_good_uph']:.1f} (다른 Process 평균 {others['scheduled_good_uph'].mean():.1f})",
        f"- Cycle Time: {bn['avg_cycle_time_sec']:.1f}초 (다른 Process 평균 {others['avg_cycle_time_sec'].mean():.1f}초)",
        f"- Downtime: {bn['avg_downtime_minutes']:.1f}분 (다른 Process 평균 {others['avg_downtime_minutes'].mean():.1f}분)",
        "",
        f"Downtime Loss 중 상위 {top_n}개 원인({top_names})이 전체의 {top_share:.1%}를 차지한다.",
    ]
    return "\n".join(lines)
