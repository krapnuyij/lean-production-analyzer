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


def calculate_downtime_reason_comparison(df: pd.DataFrame, line: str, process: str) -> pd.DataFrame:
    """Downtime minutes per reason, Before vs After, for one (line, process)
    (index: reason label). Day 16 ("Improvement") is excluded, matching Before/After
    everywhere else.

    Before spans 15 days and After spans 14 days, so a raw total would unfairly
    favor whichever period has more days. The primary, day-count-independent
    comparison is therefore the daily average ("{period}_avg", minutes/day); the
    raw total and day count ("{period}_total", "{period}_days") are kept as
    clearly-separate columns for callers that want them as supplementary detail
    (e.g. chart hover text). Unlike `calculate_downtime_pareto` (a single-period
    ranking), this shows how the Loss structure itself changed.
    """
    subset = df[(df["line"] == line) & (df["process"] == process) & (df["period"].isin(["Before", "After"]))]
    totals = subset.groupby("period")[DOWNTIME_REASON_COLUMNS].sum().reindex(["Before", "After"])
    day_counts = subset.groupby("period")["date"].nunique().reindex(["Before", "After"])
    averages = totals.div(day_counts, axis=0)

    totals.columns = [DOWNTIME_REASON_LABELS[c] for c in totals.columns]
    averages.columns = [DOWNTIME_REASON_LABELS[c] for c in averages.columns]

    result = averages.T.add_suffix("_avg").join(totals.T.add_suffix("_total"))
    for period, days in day_counts.items():
        result[f"{period}_days"] = int(days)
    return result


def build_improvement_summary(
    line: str, process: str, kpi_table: pd.DataFrame, reason_comparison: pd.DataFrame
) -> str:
    """Rule-based (non-LLM) Korean text summary of the Improvement Impact screen.

    Every number comes from `kpi_table` (see metrics.calculate_improvement_kpis) and
    `reason_comparison` (see calculate_downtime_reason_comparison) -- nothing is
    hard-coded. Direction words (상승/하락, 개선/악화, 감소/증가) are chosen from the
    actual computed sign so the sentence stays accurate even if a KPI moved the
    "wrong" way. Causation is deliberately not asserted for the downtime-reason
    finding -- only the co-occurring pattern is described.
    """
    before = kpi_table.loc["Before"]
    after = kpi_table.loc["After"]

    attainment_pp = (after["production_attainment"] - before["production_attainment"]) * 100
    uph_pct = (after["scheduled_good_uph"] - before["scheduled_good_uph"]) / before["scheduled_good_uph"] * 100
    cycle_pct = (after["avg_cycle_time_sec"] - before["avg_cycle_time_sec"]) / before["avg_cycle_time_sec"] * 100
    downtime_pct = (
        (after["avg_downtime_minutes"] - before["avg_downtime_minutes"]) / before["avg_downtime_minutes"] * 100
    )
    defect_pp = (after["defect_rate"] - before["defect_rate"]) * 100

    attainment_word = "상승했다" if attainment_pp > 0 else "하락했다"
    uph_word = "개선되었다" if uph_pct > 0 else "악화되었다"
    cycle_word = "감소했다" if cycle_pct < 0 else "증가했다"
    downtime_word = "감소했다" if downtime_pct < 0 else "증가했다"
    defect_word = "감소했다" if defect_pp < 0 else "증가했다"

    if cycle_word == downtime_word:
        uph_conn = "향상되었고" if uph_pct > 0 else "저하되었고"
        cycle_downtime_line = (
            f"Scheduled Good UPH는 {abs(uph_pct):.1f}% {uph_conn}, Cycle Time과 Downtime은 각각 "
            f"{abs(cycle_pct):.1f}%, {abs(downtime_pct):.1f}% {cycle_word}."
        )
    else:
        cycle_downtime_line = (
            f"Scheduled Good UPH는 {abs(uph_pct):.1f}% {uph_word}. Cycle Time은 {abs(cycle_pct):.1f}% "
            f"{cycle_word}. Downtime은 {abs(downtime_pct):.1f}% {downtime_word}."
        )

    lines = [
        f"{line} / {process} 개선 후 생산계획 달성률은 {before['production_attainment']:.1%}에서 "
        f"{after['production_attainment']:.1%}로 {attainment_word} ({attainment_pp:+.1f}%p).",
        cycle_downtime_line,
        f"불량률은 {before['defect_rate']:.1%}에서 {after['defect_rate']:.1%}로 "
        f"{abs(defect_pp):.1f}%p {defect_word}.",
    ]

    top_before_reason = reason_comparison["Before_avg"].idxmax()
    reason_before_val = reason_comparison.loc[top_before_reason, "Before_avg"]
    reason_after_val = reason_comparison.loc[top_before_reason, "After_avg"]
    if reason_before_val > 0 and reason_after_val < reason_before_val:
        reason_change_pct = (reason_after_val - reason_before_val) / reason_before_val * 100
        lines.append(
            f"Before 기간 최대 Loss였던 {top_before_reason} Downtime이 {abs(reason_change_pct):.1f}% "
            f"감소했으며, 이와 함께 생산성이 개선되는 패턴이 나타났다."
        )
    else:
        lines.append(
            f"Before 기간 최대 Loss였던 {top_before_reason} Downtime은 After 기간에도 뚜렷하게 "
            f"줄어들지 않은 것으로 나타났다."
        )

    reason_avg_increase = reason_comparison["After_avg"] - reason_comparison["Before_avg"]
    increased_reasons = reason_avg_increase[reason_avg_increase > 0].sort_values(ascending=False).index.tolist()
    if increased_reasons:
        lines.append(
            f"반면 {', '.join(increased_reasons)} Downtime은 After 기간에 증가해 후속 개선 과제로 확인되었다."
        )

    return "\n".join(lines)


def build_analysis_summary(
    line: str, period: str, bottleneck_process: str, process_kpis: pd.DataFrame, pareto: pd.DataFrame
) -> str:
    """Rule-based (non-LLM) Korean text summary of the Bottleneck Analysis screen,
    entirely from computed numbers: the bottleneck process, how it differs from the
    other processes' average, and the top downtime loss reasons.

    `bottleneck_process` must be the Capacity-based Primary Bottleneck already
    determined via `identify_daily_bottleneck()` / `summarize_bottlenecks()` --
    this function does not re-derive a bottleneck from UPH or any other metric, so
    the warning banner, the Pareto chart, and this summary always agree on the same
    process.

    The joined reason names are runtime-determined by the data, so no alternating
    Korean particle (은/는/이/가/을/를/와/과) is ever attached directly to them --
    each is followed only by the invariant "의", or by a fixed Korean noun ("원인",
    "퍼센트") that carries the particle instead. This keeps the sentence
    grammatical no matter which Process or reason combination comes back.
    """
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
