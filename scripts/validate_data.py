"""Validation and KPI sanity-check script for data/production_data.csv.

Run after src/generate_data.py:
  1. Row-level data consistency checks (Section 9 of the update instructions)
  2. Derived KPI computation with sum-based (weighted) aggregation
  3. Scenario sanity checks, enforced as explicit assertions where meaningful
     (bottleneck line/process identification, before/after improvement, Pareto
     downtime causes) -- kept loose enough to not be brittle against daily noise.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.generate_data import (  # noqa: E402
    DOWNTIME_REASON_COLUMNS, LINES, PROCESSES, SHIFT_MINUTES, START_DATE, period_for_day,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "production_data.csv"

EXPECTED_ROWS = 360
PERIODS = ["Before", "Improvement", "After"]

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)


def load() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


def validate_consistency(df: pd.DataFrame) -> None:
    errors = []

    if len(df) != EXPECTED_ROWS:
        errors.append(f"row count {len(df)} != {EXPECTED_ROWS}")

    dup = df.duplicated(subset=["date", "line", "process"])
    if dup.any():
        errors.append(f"{dup.sum()} duplicate (date, line, process) combinations")

    if df.isna().any().any():
        errors.append("missing values found")

    if not (df["good_qty"] + df["defect_qty"] == df["actual_qty"]).all():
        errors.append("good_qty + defect_qty != actual_qty for some rows")

    if not (df["actual_qty"] <= df["planned_qty"]).all():
        errors.append("actual_qty > planned_qty for some rows")

    if not (df["planned_qty"] > 0).all():
        errors.append("planned_qty must be > 0")

    if not (df["worker_count"] > 0).all():
        errors.append("worker_count must be > 0")

    for col in ["actual_qty", "good_qty", "defect_qty", "downtime_minutes"]:
        if (df[col] < 0).any():
            errors.append(f"{col} has negative values")

    if not (df["operating_minutes"] > 0).all():
        errors.append("operating_minutes must be > 0 for all rows")

    shift_sum = df["operating_minutes"] + df["downtime_minutes"]
    if not np.isclose(shift_sum, SHIFT_MINUTES, atol=0.05).all():
        errors.append("operating_minutes + downtime_minutes != shift minutes for some rows")

    reason_sum = df[DOWNTIME_REASON_COLUMNS].sum(axis=1)
    if not np.isclose(reason_sum, df["downtime_minutes"], atol=0.05).all():
        errors.append("downtime reason minutes do not sum to downtime_minutes for some rows")

    capacity_qty = df["operating_minutes"] * 60.0 / df["cycle_time_sec"]
    if (capacity_qty < 0).any():
        errors.append("negative process capacity found")

    min_capacity = df.assign(capacity_qty=capacity_qty).groupby(["date", "line"])["capacity_qty"].transform("min")
    if not (df["actual_qty"] <= min_capacity + 1.0).all():
        errors.append("actual_qty exceeds the bottleneck process capacity for some (date, line)")

    for col in ["planned_qty", "actual_qty", "good_qty", "defect_qty", "worker_count"]:
        if not pd.api.types.is_integer_dtype(df[col]):
            errors.append(f"{col} is not an integer dtype ({df[col].dtype})")

    if not df["line"].isin(LINES).all():
        errors.append("unexpected value(s) in 'line'")
    if not df["process"].isin(PROCESSES).all():
        errors.append("unexpected value(s) in 'process'")
    if not df["period"].isin(PERIODS).all():
        errors.append("unexpected value(s) in 'period'")

    day_idx = (df["date"] - pd.Timestamp(START_DATE)).dt.days + 1
    expected_period = day_idx.apply(period_for_day)
    if not (df["period"] == expected_period).all():
        errors.append("period does not match the expected date -> period mapping")

    if errors:
        raise AssertionError("Data consistency validation failed:\n" + "\n".join(errors))

    print(f"[OK] consistency validation passed ({len(df)} rows)")


def add_row_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["production_attainment"] = df["actual_qty"] / df["planned_qty"]
    df["defect_rate"] = df["defect_qty"] / df["actual_qty"]
    df["scheduled_good_uph"] = df["good_qty"] / (SHIFT_MINUTES / 60.0)
    df["runtime_good_uph"] = df["good_qty"] / (df["operating_minutes"] / 60.0)
    df["downtime_rate"] = df["downtime_minutes"] / SHIFT_MINUTES
    return df


def weighted_kpis(sub: pd.DataFrame) -> pd.Series:
    """Sum-based (weighted) KPI aggregation -- not a mean of daily ratios."""
    scheduled_hours = len(sub) * SHIFT_MINUTES / 60.0
    operating_hours = sub["operating_minutes"].sum() / 60.0
    return pd.Series({
        "production_attainment": sub["actual_qty"].sum() / sub["planned_qty"].sum(),
        "defect_rate": sub["defect_qty"].sum() / sub["actual_qty"].sum(),
        "scheduled_good_uph": sub["good_qty"].sum() / scheduled_hours,
        "runtime_good_uph": sub["good_qty"].sum() / operating_hours,
        "downtime_minutes_avg": sub["downtime_minutes"].mean(),
        "cycle_time_sec_avg": sub["cycle_time_sec"].mean(),
    })


def determine_bottleneck(df: pd.DataFrame) -> pd.DataFrame:
    """Per (date, line): which process has the lowest capacity, and whether the plan
    itself (rather than any process) was the binding constraint that day."""
    cap = df.assign(capacity_qty=df["operating_minutes"] * 60.0 / df["cycle_time_sec"])
    idx = cap.groupby(["date", "line"])["capacity_qty"].idxmin()
    bottleneck = cap.loc[idx, ["date", "line", "period", "process", "capacity_qty", "planned_qty"]].copy()
    bottleneck = bottleneck.rename(columns={"process": "bottleneck_process"})
    bottleneck["plan_limited"] = bottleneck["planned_qty"] <= bottleneck["capacity_qty"] + 0.5
    return bottleneck.reset_index(drop=True)


def print_line_kpi(df: pd.DataFrame) -> None:
    print("\n=== A. 정상 Line별 production attainment (합계 기준) ===")
    g = df.groupby("line").apply(weighted_kpis, include_groups=False).round(4)
    print(g)


def print_process_line_kpi(df: pd.DataFrame) -> None:
    print("\n=== B. Process x Line KPI (합계 기준) ===")
    g = df.groupby(["process", "line"]).apply(weighted_kpis, include_groups=False).round(4)
    print(g)

    print("\n--- Stitching only (LINE-A vs B vs C) ---")
    print(g.loc["Stitching"])


def print_line_b_stitching_before_after(df: pd.DataFrame) -> None:
    print("\n=== C. LINE-B Stitching Before vs After (Day 16 Improvement 제외) ===")
    sub = df[(df["line"] == "LINE-B") & (df["process"] == "Stitching") & (df["period"].isin(["Before", "After"]))]
    g = sub.groupby("period").apply(weighted_kpis, include_groups=False).round(3)
    print(g.reindex(["Before", "After"]))

    print("\n--- Before/After 분포 overlap 확인 (scheduled_good_uph, 일자별) ---")
    for period in ["Before", "After"]:
        vals = sub[sub["period"] == period]["scheduled_good_uph"].sort_values()
        print(f"  {period:<6} min={vals.min():.2f} p25={vals.quantile(.25):.2f} "
              f"median={vals.median():.2f} p75={vals.quantile(.75):.2f} max={vals.max():.2f}")
    before_vals = sub[sub["period"] == "Before"]["scheduled_good_uph"]
    after_vals = sub[sub["period"] == "After"]["scheduled_good_uph"]
    overlap = before_vals.max() >= after_vals.min()
    print(f"  Before/After 값 범위 overlap 여부: {overlap} "
          f"(Before max={before_vals.max():.2f}, After min={after_vals.min():.2f})")


def print_line_b_stitching_downtime_reason(df: pd.DataFrame) -> None:
    print("\n=== D. LINE-B Stitching Before Downtime Reason 합계 (내림차순) ===")
    sub = df[(df["line"] == "LINE-B") & (df["process"] == "Stitching") & (df["period"] == "Before")]
    totals = sub[DOWNTIME_REASON_COLUMNS].sum().sort_values(ascending=False)
    grand_total = totals.sum()
    for reason, minutes in totals.items():
        print(f"  {reason:<24} {minutes:8.1f} min  ({minutes / grand_total:.1%})")


def print_bottleneck_determination(df: pd.DataFrame) -> None:
    print("\n=== E. 날짜별 Line Bottleneck Process 판정 ===")
    bottleneck = determine_bottleneck(df)

    print("\n--- Line별 bottleneck process 발생 횟수 (전체 30일) ---")
    print(bottleneck.groupby(["line", "bottleneck_process"]).size().unstack(fill_value=0))

    lb_before = bottleneck[(bottleneck["line"] == "LINE-B") & (bottleneck["period"] == "Before")]
    stitching_days = (lb_before["bottleneck_process"] == "Stitching").sum()
    print(f"\nLINE-B Before {len(lb_before)}일 중 Stitching이 bottleneck process였던 날: {stitching_days}일")

    return bottleneck


def validate_scenario(df: pd.DataFrame, bottleneck: pd.DataFrame) -> None:
    print("\n=== F. 시나리오 sanity check (assertion) ===")
    kpi = df.groupby(["line", "process", "period"]).apply(weighted_kpis, include_groups=False)

    def get(line, process, period, col):
        return kpi.loc[(line, process, period), col]

    lb_before_uph = get("LINE-B", "Stitching", "Before", "scheduled_good_uph")
    la_before_uph = get("LINE-A", "Stitching", "Before", "scheduled_good_uph")
    lc_before_uph = get("LINE-C", "Stitching", "Before", "scheduled_good_uph")
    assert lb_before_uph < la_before_uph and lb_before_uph < lc_before_uph, (
        f"LINE-B Stitching Before scheduled UPH({lb_before_uph:.2f}) should be lower than "
        f"LINE-A({la_before_uph:.2f}) / LINE-C({lc_before_uph:.2f})"
    )
    print(f"[OK] LINE-B Stitching Before scheduled UPH가 LINE-A/C보다 낮음 "
          f"({lb_before_uph:.2f} < {la_before_uph:.2f}, {lc_before_uph:.2f})")

    lb_before_dt = get("LINE-B", "Stitching", "Before", "downtime_minutes_avg")
    la_before_dt = get("LINE-A", "Stitching", "Before", "downtime_minutes_avg")
    lc_before_dt = get("LINE-C", "Stitching", "Before", "downtime_minutes_avg")
    assert lb_before_dt > la_before_dt and lb_before_dt > lc_before_dt, (
        f"LINE-B Stitching Before downtime({lb_before_dt:.1f}) should be higher than "
        f"LINE-A({la_before_dt:.1f}) / LINE-C({lc_before_dt:.1f})"
    )
    print(f"[OK] LINE-B Stitching Before downtime가 LINE-A/C보다 높음 "
          f"({lb_before_dt:.1f} > {la_before_dt:.1f}, {lc_before_dt:.1f})")

    lb_after_uph = get("LINE-B", "Stitching", "After", "scheduled_good_uph")
    lb_after_dt = get("LINE-B", "Stitching", "After", "downtime_minutes_avg")
    assert lb_after_uph > lb_before_uph, (
        f"LINE-B Stitching After scheduled UPH({lb_after_uph:.2f}) should improve over "
        f"Before({lb_before_uph:.2f})"
    )
    assert lb_after_dt < lb_before_dt, (
        f"LINE-B Stitching After downtime({lb_after_dt:.1f}) should be lower than "
        f"Before({lb_before_dt:.1f})"
    )
    print(f"[OK] LINE-B Stitching After가 Before보다 개선됨 "
          f"(UPH {lb_before_uph:.2f}->{lb_after_uph:.2f}, downtime {lb_before_dt:.1f}->{lb_after_dt:.1f}min)")

    sub = df[(df["line"] == "LINE-B") & (df["process"] == "Stitching") & (df["period"] == "Before")]
    totals = sub[DOWNTIME_REASON_COLUMNS].sum()
    top_two = totals[["changeover_minutes", "machine_stop_minutes"]].sum()
    rest = totals.drop(["changeover_minutes", "machine_stop_minutes"]).sum()
    assert top_two > rest, (
        f"Changeover+Machine Stop({top_two:.1f}min) should exceed the other three reasons "
        f"combined({rest:.1f}min) for LINE-B Stitching Before"
    )
    print(f"[OK] Changeover+Machine Stop이 LINE-B Stitching Before의 주요 downtime loss "
          f"({top_two:.1f}min > 나머지 {rest:.1f}min)")

    lb_before_bn = bottleneck[(bottleneck["line"] == "LINE-B") & (bottleneck["period"] == "Before")]
    stitching_share = (lb_before_bn["bottleneck_process"] == "Stitching").mean()
    assert stitching_share >= 0.6, (
        f"Stitching should be the bottleneck process on most LINE-B Before days "
        f"(actual share={stitching_share:.0%})"
    )
    print(f"[OK] LINE-B Before {len(lb_before_bn)}일 중 {stitching_share:.0%}에서 Stitching이 bottleneck process")


def main():
    df = load()
    validate_consistency(df)
    df = add_row_kpis(df)
    print_line_kpi(df)
    print_process_line_kpi(df)
    print_line_b_stitching_before_after(df)
    print_line_b_stitching_downtime_reason(df)
    bottleneck = print_bottleneck_determination(df)
    validate_scenario(df, bottleneck)


if __name__ == "__main__":
    main()
