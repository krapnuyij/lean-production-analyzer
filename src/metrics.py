"""KPI calculation for the Production LEAN Improvement Analyzer dashboard.

The source data (data/production_data.csv) is at (date, line, process) grain, but
`planned_qty` / `actual_qty` are Line-level values that repeat identically across a
Line's 4 process rows for the same date (see src/generate_data.py: the whole Line's
daily output is capped by its slowest process). Summing those columns directly over
process rows would therefore quadruple-count production quantity.

Line-level KPIs (Production Overview) always go through `create_line_daily_data`,
which deduplicates to one row per (date, line) first. Process-level KPIs (Bottleneck
Analysis) use the raw (date, line, process) rows as-is, since cycle time, downtime,
and defect rate genuinely differ row by row.
"""

import pandas as pd

SHIFT_MINUTES = 480.0  # standard 8h shift, matches src/generate_data.py
FINAL_PROCESS = "Finishing"  # a Line's shippable good/defect output is measured here


def create_line_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse (date, line, process) rows to one row per (date, line).

    `planned_qty` / `actual_qty` are taken once per (date, line) instead of summed
    across the 4 process rows. The Line's final good/defect output is taken from the
    Finishing process, since that is the last step before a unit is shippable.
    """
    line_level = (
        df.drop_duplicates(subset=["date", "line"])[["date", "line", "period", "planned_qty", "actual_qty"]]
        .reset_index(drop=True)
    )

    finishing = df[df["process"] == FINAL_PROCESS][
        ["date", "line", "good_qty", "defect_qty"]
    ].rename(columns={"good_qty": "final_good_qty", "defect_qty": "final_defect_qty"})

    line_daily = line_level.merge(finishing, on=["date", "line"], how="left")
    line_daily["production_attainment"] = line_daily["actual_qty"] / line_daily["planned_qty"]
    return line_daily


def calculate_overview_kpis(df: pd.DataFrame) -> dict:
    """The 4 top-level KPI card values for the Production Overview screen.

    production_attainment / final_good_uph / final_defect_rate come from the
    deduplicated Line-level view. avg_downtime_minutes is a process-level average
    over every process row in `df` (all 4 processes), since downtime is only
    meaningful at the process grain.
    """
    line_daily = create_line_daily_data(df)
    scheduled_hours = len(line_daily) * SHIFT_MINUTES / 60.0
    good = line_daily["final_good_qty"].sum()
    defect = line_daily["final_defect_qty"].sum()
    return {
        "production_attainment": line_daily["actual_qty"].sum() / line_daily["planned_qty"].sum(),
        "final_good_uph": good / scheduled_hours,
        "final_defect_rate": defect / (good + defect),
        "avg_downtime_minutes": df["downtime_minutes"].mean(),
    }


def calculate_line_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Line-level KPI table (index: line), from the deduplicated (date, line) view.

    production_attainment: sum(actual_qty) / sum(planned_qty) per line.
    final_good_uph / final_defect_rate: based on the Finishing process's good/defect
    output, treated as the Line's final shippable quality/output.
    """
    line_daily = create_line_daily_data(df)
    rows = []
    for line, g in line_daily.groupby("line"):
        scheduled_hours = len(g) * SHIFT_MINUTES / 60.0
        good = g["final_good_qty"].sum()
        defect = g["final_defect_qty"].sum()
        rows.append({
            "line": line,
            "production_attainment": g["actual_qty"].sum() / g["planned_qty"].sum(),
            "final_good_uph": good / scheduled_hours,
            "final_defect_rate": defect / (good + defect),
        })
    return pd.DataFrame(rows).set_index("line")


def calculate_process_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Process-level KPI table (index: process), from the raw (date, line, process)
    rows -- no deduplication, since cycle time / downtime / defects are genuine
    per-process measurements. Callers should pass in a single-Line subset for a
    like-for-like process comparison (the UPH figures sum good_qty across whatever
    rows are passed in).
    """
    rows = []
    for process, g in df.groupby("process"):
        scheduled_hours = len(g) * SHIFT_MINUTES / 60.0
        operating_hours = g["operating_minutes"].sum() / 60.0
        rows.append({
            "process": process,
            "scheduled_good_uph": g["good_qty"].sum() / scheduled_hours,
            "runtime_good_uph": g["good_qty"].sum() / operating_hours,
            "avg_cycle_time_sec": g["cycle_time_sec"].mean(),
            "avg_downtime_minutes": g["downtime_minutes"].mean(),
            "defect_rate": g["defect_qty"].sum() / g["actual_qty"].sum(),
        })
    return pd.DataFrame(rows).set_index("process")
