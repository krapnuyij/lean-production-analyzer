"""Regression test for the Capacity-based Bottleneck consistency bug.

app.py determines the Primary Bottleneck once via `identify_daily_bottleneck()` /
`summarize_bottlenecks()` (lowest Process Capacity = operating_minutes * 60 /
cycle_time_sec) and reuses that same process for the warning banner, the Pareto
chart, and `build_analysis_summary()`. `build_analysis_summary()` must never
re-derive its own bottleneck from `scheduled_good_uph` -- that KPI is also pulled
down by defect_rate, so on some Line/Period combinations its minimum can point at a
different process than the Capacity-based bottleneck.
"""

import unittest

import pandas as pd

from src import analysis


class BuildAnalysisSummaryBottleneckConsistencyTest(unittest.TestCase):
    def test_uses_passed_in_capacity_based_bottleneck_not_uph_min(self) -> None:
        # Process "B" has the lowest scheduled_good_uph, but only because of a much
        # higher defect_rate -- its Cycle Time/Downtime (and therefore Capacity) are
        # actually fine. Process "A" is the Capacity-based bottleneck (longest Cycle
        # Time, highest Downtime). The pre-fix code picked "B" via
        # `process_kpis["scheduled_good_uph"].idxmin()`; the fix must honor whatever
        # `bottleneck_process` is passed in instead.
        process_kpis = pd.DataFrame(
            {
                "scheduled_good_uph": [40.0, 30.0, 50.0, 55.0],
                "runtime_good_uph": [45.0, 42.0, 52.0, 58.0],
                "avg_cycle_time_sec": [75.0, 60.0, 58.0, 55.0],
                "avg_downtime_minutes": [70.0, 30.0, 28.0, 25.0],
                "defect_rate": [0.02, 0.15, 0.02, 0.02],
            },
            index=["A", "B", "C", "D"],
        )
        pareto = pd.DataFrame(
            {
                "downtime_minutes": [40.0, 30.0],
                "share": [0.6, 0.4],
                "cumulative_share": [0.6, 1.0],
            },
            index=["Changeover", "Machine Stop"],
        )

        # sanity check: the old, buggy selection criterion would point at "B", not "A"
        self.assertEqual(process_kpis["scheduled_good_uph"].idxmin(), "B")

        summary = analysis.build_analysis_summary("LINE-X", "Before", "A", process_kpis, pareto)

        self.assertIn("A의 Process Capacity가 가장 낮게 나타나", summary)
        self.assertNotIn("B의 Process Capacity", summary)
        self.assertNotIn("B의 주요 지표", summary)


if __name__ == "__main__":
    unittest.main()
