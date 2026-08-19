"""Synthetic production dataset generator for the Production LEAN Improvement Analyzer.

30 days x 3 lines x 4 processes = 360 rows.
Day 1-15 = Before, Day 16 = Improvement (transition), Day 17-30 = After.
LINE-B / Stitching is the intentional bottleneck before the Day 16 LEAN improvement
(changeover standardization, material/tool pre-staging, line balancing).

Simplified steady-state throughput model (not a discrete-event / WIP simulation):
for each (date, line) a Line-level planned_qty is generated independently, each of the
4 processes gets its own cycle_time_sec / downtime_minutes -> production capacity
(capacity = operating_minutes * 60 / cycle_time_sec, no hidden efficiency factor), and
the Line's actual output that day is capped by min(planned_qty, *process capacities).
The slowest process (the bottleneck) therefore limits every process row of that
Line/day, matching how a bottleneck process constrains a real production line.

All generation parameters live in this file's config section so the scenario can be
tuned from one place instead of being scattered through the generation logic.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

START_DATE = "2026-01-01"
NUM_DAYS = 30
IMPROVEMENT_DAY = 16  # day of the LEAN improvement activity (transition day)
RAMP_DAYS_AFTER = 2   # day 17-18 gradually ramp up toward the full After profile

LINES = ["LINE-A", "LINE-B", "LINE-C"]
PROCESSES = ["Cutting", "Stitching", "Assembly", "Finishing"]

SHIFT_MINUTES = 480.0  # standard 8h shift; basis for scheduled time and downtime_rate

DOWNTIME_REASONS = ["Changeover", "Machine Stop", "Material Delay", "Quality Issue", "Worker Absence"]
DOWNTIME_REASON_COLUMNS = [
    "changeover_minutes", "machine_stop_minutes", "material_delay_minutes",
    "quality_issue_minutes", "worker_absence_minutes",
]
DEFAULT_DOWNTIME_PROPORTIONS = [0.20, 0.20, 0.25, 0.20, 0.15]
LINE_B_STITCHING_BEFORE_DOWNTIME_PROPORTIONS = [0.40, 0.30, 0.15, 0.10, 0.05]
LINE_B_STITCHING_AFTER_DOWNTIME_PROPORTIONS = [0.18, 0.24, 0.24, 0.20, 0.14]
# concentration of the daily Dirichlet draw around the target proportions above
# (higher = less day-to-day variation in the reason split)
DOWNTIME_PROPORTIONS_CONCENTRATION = 20.0

DEFECT_REASONS = ["Stitch Defect", "Material Defect", "Assembly Error", "Finishing Defect"]
PROCESS_DEFECT_WEIGHTS = {
    "Cutting":   [0.10, 0.70, 0.10, 0.10],
    "Stitching": [0.70, 0.15, 0.10, 0.05],
    "Assembly":  [0.10, 0.10, 0.70, 0.10],
    "Finishing": [0.10, 0.10, 0.10, 0.70],
}

# actual day-to-day performance profile per process (used unless overridden below)
PROCESS_PROFILE = {
    "Cutting":   {"cycle_mean": 52.0, "cycle_sd": 1.5, "downtime_mean": 25.0, "downtime_sd": 3.0, "defect_mean": 0.0175, "defect_sd": 0.0015, "worker_min": 6, "worker_max": 7},
    "Stitching": {"cycle_mean": 60.0, "cycle_sd": 1.5, "downtime_mean": 35.0, "downtime_sd": 3.0, "defect_mean": 0.0240, "defect_sd": 0.0020, "worker_min": 8, "worker_max": 10},
    "Assembly":  {"cycle_mean": 61.0, "cycle_sd": 1.5, "downtime_mean": 32.0, "downtime_sd": 4.0, "defect_mean": 0.0225, "defect_sd": 0.0015, "worker_min": 7, "worker_max": 9},
    "Finishing": {"cycle_mean": 55.0, "cycle_sd": 1.5, "downtime_mean": 22.0, "downtime_sd": 3.0, "defect_mean": 0.0140, "defect_sd": 0.0015, "worker_min": 5, "worker_max": 6},
}

# small (+-1~3%) per-line variation on cycle time / downtime so LINE-A/B/C aren't drawn
# from an identical distribution; deliberately kept small so it never masks the
# LINE-B / Stitching bottleneck signal. Does not apply to the Stitching override below.
LINE_CYCLE_FACTOR = {"LINE-A": 1.00, "LINE-B": 1.01, "LINE-C": 0.99}
LINE_DOWNTIME_FACTOR = {"LINE-A": 0.97, "LINE-B": 1.02, "LINE-C": 1.01}

# LINE-B / Stitching bottleneck: Before (steady-state) and After (steady-state) profiles.
# Day 16 (Improvement) and day 17-18 (ramp-up) interpolate between these two, and a
# fraction of the remaining After days get an occasional rougher day (see BAD_DAY_*).
LINE_B_STITCHING_BEFORE = {"cycle_mean": 71.0, "cycle_sd": 3.5, "downtime_mean": 68.0, "downtime_sd": 7.0, "defect_mean": 0.0330, "defect_sd": 0.0025}
LINE_B_STITCHING_AFTER = {"cycle_mean": 62.0, "cycle_sd": 2.0, "downtime_mean": 40.0, "downtime_sd": 4.5, "defect_mean": 0.0275, "defect_sd": 0.0012}

BAD_DAY_PROB_AFTER = 0.20  # ~2 rougher days over the 12 steady-state After days
BAD_DAY_CYCLE_ADD = (3.0, 6.0)
BAD_DAY_DOWNTIME_ADD = (10.0, 20.0)

# Line-level planned_qty: an independent planning baseline, NOT derived from that day's
# process capacities. Chosen so a healthy line's typical capacity (~440-450 units/day,
# limited by its slowest normal process) sits slightly below plan most days.
PLANNED_QTY_MEAN = 455.0
PLANNED_QTY_SD_PCT = 0.03

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "production_data.csv"


def period_for_day(day: int) -> str:
    if day < IMPROVEMENT_DAY:
        return "Before"
    if day == IMPROVEMENT_DAY:
        return "Improvement"
    return "After"


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _interpolate_proportions(before: list, after: list, t: float) -> np.ndarray:
    arr = np.array(before) + (np.array(after) - np.array(before)) * t
    return arr / arr.sum()


def line_b_stitching_params(day_idx: int, rng: np.random.Generator):
    """Cycle/downtime/defect parameters and downtime-reason proportions for
    LINE-B / Stitching on a given day: Before -> Improvement -> ramp -> After."""
    if day_idx < IMPROVEMENT_DAY:
        t = 0.0
    elif day_idx == IMPROVEMENT_DAY:
        t = 0.3
    elif day_idx <= IMPROVEMENT_DAY + RAMP_DAYS_AFTER:
        t = 0.6 + 0.25 * (day_idx - IMPROVEMENT_DAY - 1)  # day 17 -> 0.6, day 18 -> 0.85
    else:
        t = 1.0

    before, after = LINE_B_STITCHING_BEFORE, LINE_B_STITCHING_AFTER
    cycle_mean = _lerp(before["cycle_mean"], after["cycle_mean"], t)
    cycle_sd = _lerp(before["cycle_sd"], after["cycle_sd"], t)
    downtime_mean = _lerp(before["downtime_mean"], after["downtime_mean"], t)
    downtime_sd = _lerp(before["downtime_sd"], after["downtime_sd"], t)
    defect_mean = _lerp(before["defect_mean"], after["defect_mean"], t)
    defect_sd = _lerp(before["defect_sd"], after["defect_sd"], t)
    proportions = _interpolate_proportions(
        LINE_B_STITCHING_BEFORE_DOWNTIME_PROPORTIONS, LINE_B_STITCHING_AFTER_DOWNTIME_PROPORTIONS, t
    )

    # only the fully-settled After days (day >= IMPROVEMENT_DAY + RAMP_DAYS_AFTER + 1)
    # can roll a rougher day, so the ramp-up itself stays monotonic
    if day_idx > IMPROVEMENT_DAY + RAMP_DAYS_AFTER and rng.random() < BAD_DAY_PROB_AFTER:
        cycle_mean += rng.uniform(*BAD_DAY_CYCLE_ADD)
        downtime_mean += rng.uniform(*BAD_DAY_DOWNTIME_ADD)

    return cycle_mean, cycle_sd, downtime_mean, downtime_sd, defect_mean, defect_sd, proportions


def _split_downtime_minutes(downtime_minutes: float, proportions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Split downtime_minutes into 5 reason-minute values (with daily random variation
    around `proportions`) that sum exactly to downtime_minutes."""
    fracs = rng.dirichlet(proportions * DOWNTIME_PROPORTIONS_CONCENTRATION)
    minutes = np.round(downtime_minutes * fracs, 1)
    diff = round(downtime_minutes - minutes.sum(), 1)
    minutes[np.argmax(fracs)] += diff
    return minutes


def generate(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=NUM_DAYS, freq="D")

    rows = []
    for day_idx, date in enumerate(dates, start=1):
        period = period_for_day(day_idx)
        for line in LINES:
            planned_qty = int(round(rng.normal(PLANNED_QTY_MEAN, PLANNED_QTY_MEAN * PLANNED_QTY_SD_PCT)))

            process_data = {}
            for process in PROCESSES:
                profile = PROCESS_PROFILE[process]
                is_bottleneck = line == "LINE-B" and process == "Stitching"

                if is_bottleneck:
                    cycle_mean, cycle_sd, downtime_mean, downtime_sd, defect_mean, defect_sd, proportions = \
                        line_b_stitching_params(day_idx, rng)
                else:
                    cycle_mean = profile["cycle_mean"] * LINE_CYCLE_FACTOR[line]
                    cycle_sd = profile["cycle_sd"]
                    downtime_mean = profile["downtime_mean"] * LINE_DOWNTIME_FACTOR[line]
                    downtime_sd = profile["downtime_sd"]
                    defect_mean, defect_sd = profile["defect_mean"], profile["defect_sd"]
                    proportions = np.array(DEFAULT_DOWNTIME_PROPORTIONS)

                # observed cycle time / downtime for this process on this day (no hidden
                # efficiency factor is applied anywhere downstream of these two values)
                cycle_time_sec = round(float(np.clip(
                    rng.normal(cycle_mean, cycle_sd), cycle_mean - 3 * cycle_sd, cycle_mean + 3 * cycle_sd
                )), 1)
                downtime_minutes = round(float(np.clip(
                    rng.normal(downtime_mean, downtime_sd), 5.0, SHIFT_MINUTES - 60.0
                )), 1)
                operating_minutes = round(SHIFT_MINUTES - downtime_minutes, 1)
                capacity_qty = operating_minutes * 60.0 / cycle_time_sec

                defect_rate = float(np.clip(rng.normal(defect_mean, defect_sd), 0.0, 0.15))
                reason_minutes = _split_downtime_minutes(downtime_minutes, proportions, rng)
                worker_count = int(rng.integers(profile["worker_min"], profile["worker_max"] + 1))
                defect_reason = rng.choice(DEFECT_REASONS, p=PROCESS_DEFECT_WEIGHTS[process])

                process_data[process] = {
                    "cycle_time_sec": cycle_time_sec,
                    "downtime_minutes": downtime_minutes,
                    "operating_minutes": operating_minutes,
                    "capacity_qty": capacity_qty,
                    "defect_rate": defect_rate,
                    "reason_minutes": reason_minutes,
                    "worker_count": worker_count,
                    "primary_defect_reason": defect_reason,
                }

            # the Line's actual throughput is capped by its plan AND its slowest process
            capacities = [process_data[p]["capacity_qty"] for p in PROCESSES]
            line_actual_qty = int(np.floor(min(planned_qty, min(capacities))))

            for process in PROCESSES:
                pdata = process_data[process]
                actual_qty = line_actual_qty
                defect_qty = min(int(round(actual_qty * pdata["defect_rate"])), actual_qty)
                good_qty = actual_qty - defect_qty

                reason_minutes = pdata["reason_minutes"]
                primary_downtime_reason = DOWNTIME_REASONS[int(np.argmax(reason_minutes))]

                row = {
                    "date": date.strftime("%Y-%m-%d"),
                    "line": line,
                    "process": process,
                    "period": period,
                    "planned_qty": planned_qty,
                    "actual_qty": actual_qty,
                    "good_qty": good_qty,
                    "defect_qty": defect_qty,
                    "operating_minutes": pdata["operating_minutes"],
                    "downtime_minutes": pdata["downtime_minutes"],
                    "cycle_time_sec": pdata["cycle_time_sec"],
                    "worker_count": pdata["worker_count"],
                }
                for col, val in zip(DOWNTIME_REASON_COLUMNS, reason_minutes):
                    row[col] = round(float(val), 1)
                row["primary_downtime_reason"] = primary_downtime_reason
                row["primary_defect_reason"] = pdata["primary_defect_reason"]
                rows.append(row)

    return pd.DataFrame(rows)


def main():
    df = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
