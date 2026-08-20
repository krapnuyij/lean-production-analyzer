"""Streamlit dashboard for the Production LEAN Improvement Analyzer.

Two screens: Production Overview (factory-wide KPIs, Line comparison) and
Bottleneck Analysis (process drill-down, bottleneck determination, downtime
Pareto). Read-only against data/production_data.csv -- this app never writes to
or regenerates the dataset. KPI/analysis logic lives in src/metrics.py and
src/analysis.py; this file only wires that logic to the UI.

UI copy is Korean-first; LEAN/manufacturing terms commonly used in English on
the shop floor (Line, Process, Cycle Time, UPH, Capacity, Bottleneck, Downtime,
Changeover, Loss, Pareto, KPI) are kept as-is. Sentences never attach an
alternating Korean particle (은/는/이/가/을/를/와/과) directly to a value that
varies at runtime (a Process name, a joined list of Downtime reasons) --
those are always anchored to a fixed Korean noun or the invariant "의"
particle, so the wording stays grammatically correct no matter which Process
or reason combination the data produces.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analysis, metrics

DATA_PATH = Path(__file__).resolve().parent / "data" / "production_data.csv"

# fixed categorical color per entity (never re-assigned by sort order or filter)
LINE_COLORS = {"LINE-A": "#2a78d6", "LINE-B": "#eb6834", "LINE-C": "#1baf7a"}
PROCESS_ORDER = ["Cutting", "Stitching", "Assembly", "Finishing"]
PROCESS_COLORS = {"Cutting": "#2a78d6", "Stitching": "#eb6834", "Assembly": "#1baf7a", "Finishing": "#eda100"}
NEUTRAL_COLOR = "#52514e"
# Pareto chart: one hue for every bar so the cumulative-% line reads as the focal
# series; the line color is chosen for strong contrast against both a light and a
# dark chart surface (Streamlit adapts Plotly charts to the active theme).
PARETO_BAR_COLOR = "#2a78d6"
PARETO_LINE_COLOR = "#fab219"

PAGE_OVERVIEW = "생산 현황"
PAGE_BOTTLENECK = "Bottleneck 분석"

st.set_page_config(page_title="Production LEAN Improvement Analyzer", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


def filter_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "All":
        return df
    return df[df["period"] == period]


def render_header() -> None:
    st.markdown(
        "<h1 style='font-size:1.9rem; margin-bottom:0.3rem;'>Production LEAN Improvement Analyzer</h1>",
        unsafe_allow_html=True,
    )
    st.caption("생산 데이터를 기반으로 Line 성과를 모니터링하고 Bottleneck 및 주요 Loss를 탐색하는 LEAN 분석 대시보드")
    st.info("이 대시보드는 실제 기업 데이터가 아닌 synthetic(합성) 제조 데이터를 사용한다.", icon="ℹ️")


def render_overview(df: pd.DataFrame) -> None:
    st.header(PAGE_OVERVIEW)

    col_f1, col_f2 = st.columns(2)
    period = col_f1.selectbox("기간", ["Before", "After", "All"], index=0, key="ov_period")
    line_choice = col_f2.selectbox("Line", ["All", "LINE-A", "LINE-B", "LINE-C"], index=0, key="ov_line")
    if period == "All":
        st.caption("기간='All'은 Before + Improvement(전환일 1일) + After를 포함한 전체 30일을 의미한다.")

    period_df = filter_period(df, period)
    kpi_df = period_df if line_choice == "All" else period_df[period_df["line"] == line_choice]

    kpis = metrics.calculate_overview_kpis(kpi_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("생산계획 달성률", f"{kpis['production_attainment']:.1%}")
    c2.metric("최종 양품 UPH", f"{kpis['final_good_uph']:.1f}",
              help="Finishing Process 기준 Scheduled 양품 UPH (good_qty / 8시간 shift)")
    c3.metric("최종 불량률", f"{kpis['final_defect_rate']:.1%}", help="Finishing Process 기준 불량률")
    c4.metric("평균 Process Downtime", f"{kpis['avg_downtime_minutes']:.1f}분",
              help="현재 선택 범위 내 4개 Process row의 평균 Downtime")

    st.subheader("A. Line별 생산계획 달성률")
    line_kpis = metrics.calculate_line_kpis(period_df)  # always all lines: this chart is a comparison view
    avg_attainment = line_kpis["production_attainment"].mean()
    fig_a = go.Figure(go.Bar(
        x=line_kpis.index, y=line_kpis["production_attainment"],
        marker_color=[LINE_COLORS[l] for l in line_kpis.index],
        text=[f"{v:.1%}" for v in line_kpis["production_attainment"]],
        textposition="outside",
        hovertemplate="%{x}<br>생산계획 달성률: %{y:.1%}<extra></extra>",
    ))
    fig_a.add_hline(
        y=avg_attainment, line_dash="dot", line_color=NEUTRAL_COLOR, line_width=1.5,
        annotation_text=f"전체 평균 {avg_attainment:.1%}",
        annotation_position="bottom right",
        annotation_font=dict(size=11, color=NEUTRAL_COLOR),
    )
    fig_a.update_layout(
        xaxis_title="Line", yaxis_title="생산계획 달성률",
        yaxis_tickformat=".0%", showlegend=False, height=380,
    )
    st.plotly_chart(fig_a, width="stretch")
    if line_choice != "All":
        st.caption(f"이 비교 차트는 Line 필터와 무관하게 항상 전체 Line을 표시한다. (KPI 카드는 {line_choice} 기준)")

    st.subheader("B. 일별 생산계획 달성률 추이")
    line_daily = metrics.create_line_daily_data(period_df)
    fig_b = go.Figure()
    for line in ["LINE-A", "LINE-B", "LINE-C"]:
        sub = line_daily[line_daily["line"] == line].sort_values("date")
        fig_b.add_scatter(
            x=sub["date"], y=sub["production_attainment"], mode="lines+markers", name=line,
            line=dict(color=LINE_COLORS[line], width=2), marker=dict(size=5),
            hovertemplate="%{x|%Y-%m-%d}<br>" + line + ": %{y:.1%}<extra></extra>",
        )
    fig_b.update_layout(
        xaxis_title="날짜", yaxis_title="생산계획 달성률",
        yaxis_tickformat=".0%", height=400, legend_title_text="Line",
    )
    st.plotly_chart(fig_b, width="stretch")

    st.subheader("C. Line KPI 요약")
    summary_table = line_kpis.copy()
    summary_table["production_attainment"] = summary_table["production_attainment"].map(lambda v: f"{v:.1%}")
    summary_table["final_good_uph"] = summary_table["final_good_uph"].map(lambda v: f"{v:.1f}")
    summary_table["final_defect_rate"] = summary_table["final_defect_rate"].map(lambda v: f"{v:.1%}")
    summary_table.columns = ["생산계획 달성률", "최종 양품 UPH", "최종 불량률"]
    st.dataframe(summary_table, width="stretch")


def render_bottleneck(df: pd.DataFrame) -> None:
    st.header(PAGE_BOTTLENECK)

    col_f1, col_f2 = st.columns(2)
    line_choice = col_f1.selectbox("Line", ["LINE-A", "LINE-B", "LINE-C"], index=1, key="bn_line")
    period = col_f2.selectbox("기간", ["Before", "After", "All"], index=0, key="bn_period")

    filtered = filter_period(df, period)
    filtered = filtered[filtered["line"] == line_choice]
    if filtered.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없다.")
        return

    process_kpis = metrics.calculate_process_kpis(filtered).reindex(PROCESS_ORDER)
    capacity_by_process = (
        analysis.add_capacity(filtered).groupby("process")["capacity_qty"].mean().reindex(PROCESS_ORDER)
    )

    st.subheader("Process 성능 비교")
    col1, col2, col3 = st.columns(3)

    fig_capacity = go.Figure(go.Bar(
        x=capacity_by_process.index, y=capacity_by_process.values,
        marker_color=[PROCESS_COLORS[p] for p in capacity_by_process.index],
        hovertemplate="%{x}<br>평균 Capacity: %{y:.1f}개/일<extra></extra>",
    ))
    fig_capacity.update_layout(title="Process Capacity", xaxis_title="Process",
                                yaxis_title="평균 Capacity (개/일)", showlegend=False, height=340)
    col1.plotly_chart(fig_capacity, width="stretch")

    fig_cycle = go.Figure(go.Bar(
        x=process_kpis.index, y=process_kpis["avg_cycle_time_sec"],
        marker_color=[PROCESS_COLORS[p] for p in process_kpis.index],
        hovertemplate="%{x}<br>평균 Cycle Time: %{y:.1f}초<extra></extra>",
    ))
    fig_cycle.update_layout(title="Cycle Time", xaxis_title="Process",
                             yaxis_title="평균 Cycle Time (초)", showlegend=False, height=340)
    col2.plotly_chart(fig_cycle, width="stretch")

    fig_dt = go.Figure(go.Bar(
        x=process_kpis.index, y=process_kpis["avg_downtime_minutes"],
        marker_color=[PROCESS_COLORS[p] for p in process_kpis.index],
        hovertemplate="%{x}<br>평균 Downtime: %{y:.1f}분<extra></extra>",
    ))
    fig_dt.update_layout(title="Downtime", xaxis_title="Process",
                          yaxis_title="평균 Downtime (분)", showlegend=False, height=340)
    col3.plotly_chart(fig_dt, width="stretch")

    st.subheader("Bottleneck 판정")
    bottleneck_df = analysis.identify_daily_bottleneck(filtered)
    bn_summary = analysis.summarize_bottlenecks(bottleneck_df)
    top_process = bn_summary.index[0]
    top_days = int(bn_summary.iloc[0]["days"])
    top_share = bn_summary.iloc[0]["share"]
    total_days = len(bottleneck_df)

    st.warning(
        f"**주요 Bottleneck: {top_process}**\n\n"
        f"선택한 기간 {total_days}일 중 {top_days}일({top_share:.0%})에서 가장 낮은 Process Capacity를 기록했다.",
        icon="⚠️",
    )
    st.caption("Capacity(operating_minutes × 60 / cycle_time_sec) 기준으로, 날짜별 4개 Process 중 "
               "가장 낮은 값을 그날의 Bottleneck으로 판정한다.")
    display_summary = bn_summary.rename(columns={"days": "일수", "share": "비율"}).copy()
    display_summary["비율"] = display_summary["비율"].map(lambda v: f"{v:.0%}")
    st.dataframe(display_summary, width="stretch")

    st.subheader("Downtime Pareto")
    st.caption(f"Bottleneck Process: {top_process} / Downtime 원인별 분석 — {line_choice} / {period}")
    bottleneck_process_df = filtered[filtered["process"] == top_process]
    pareto = analysis.calculate_downtime_pareto(bottleneck_process_df)

    fig_pareto = go.Figure()
    fig_pareto.add_bar(
        x=pareto.index, y=pareto["downtime_minutes"], name="Downtime(분)",
        marker_color=PARETO_BAR_COLOR,
        hovertemplate="%{x}<br>Downtime: %{y:.1f}분<extra></extra>",
    )
    fig_pareto.add_scatter(
        x=pareto.index, y=pareto["cumulative_share"] * 100, name="누적 비율",
        mode="lines+markers",
        line=dict(color=PARETO_LINE_COLOR, width=3),
        marker=dict(size=8, color=PARETO_LINE_COLOR),
        yaxis="y2", hovertemplate="%{x}<br>누적: %{y:.1f}%<extra></extra>",
    )
    fig_pareto.update_layout(
        title=f"{top_process} Downtime Pareto",
        xaxis_title="Downtime 원인",
        yaxis=dict(title="Downtime(분)"),
        yaxis2=dict(title="누적 비율(%)", overlaying="y", side="right", range=[0, 100], ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
    )
    st.plotly_chart(fig_pareto, width="stretch")

    top_n = min(2, len(pareto))
    top_reasons = pareto.iloc[:top_n]
    top_share_cum = pareto["cumulative_share"].iloc[top_n - 1]
    st.success(
        f"Downtime Loss 중 상위 {top_n}개 원인({', '.join(top_reasons.index)})이 전체의 {top_share_cum:.1%}를 차지한다.",
        icon="📌",
    )

    with st.expander("Process KPI 상세"):
        details = process_kpis.copy()
        details.insert(0, "avg_capacity_qty", capacity_by_process)
        details = details.rename(columns={
            "avg_capacity_qty": "평균 Capacity (개/일)",
            "scheduled_good_uph": "Scheduled 양품 UPH",
            "runtime_good_uph": "Runtime 양품 UPH",
            "avg_cycle_time_sec": "평균 Cycle Time (초)",
            "avg_downtime_minutes": "평균 Downtime (분)",
            "defect_rate": "불량률",
        })
        details["불량률"] = details["불량률"].map(lambda v: f"{v:.1%}")
        for col in ["평균 Capacity (개/일)", "Scheduled 양품 UPH", "Runtime 양품 UPH",
                    "평균 Cycle Time (초)", "평균 Downtime (분)"]:
            details[col] = details[col].map(lambda v: f"{v:.1f}")
        st.dataframe(details, width="stretch")

    st.subheader("분석 요약")
    summary_text = analysis.build_analysis_summary(line_choice, period, process_kpis, pareto)
    with st.container(border=True):
        st.markdown(summary_text.replace("\n", "  \n"))


def main() -> None:
    df = load_data()

    with st.sidebar:
        st.markdown("### 메뉴")
        page = st.radio("화면 선택", [PAGE_OVERVIEW, PAGE_BOTTLENECK], label_visibility="collapsed")

    render_header()

    if page == PAGE_OVERVIEW:
        render_overview(df)
    else:
        render_bottleneck(df)


if __name__ == "__main__":
    main()
