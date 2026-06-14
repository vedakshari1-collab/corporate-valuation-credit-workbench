from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.credit_score import calculate_credit_score
from src.data_loader import load_financials
from src.financial_ratios import RATIO_EXPLANATIONS, calculate_financial_ratios
from src.peer_analysis import latest_peer_snapshot
from src.report_generator import generate_investment_memo, save_report
from src.utils import OUTPUTS_DIR, is_valid_number, money, multiple, parse_tickers, pct, plain_number, safe_float
from src.valuation import DCFInputs, dcf_inputs_from_row, run_dcf


st.set_page_config(
    page_title="Corporate Valuation & Credit Risk Workbench",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --workbench-border: #d7dde8;
        --workbench-blue: #1f4e79;
        --workbench-ink: #172033;
        --workbench-muted: #5d6878;
        --workbench-bg: #f7f9fc;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }
    h1, h2, h3 {
        color: var(--workbench-ink);
        letter-spacing: 0;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--workbench-border);
        border-radius: 8px;
        padding: 14px 16px;
        min-height: 116px;
    }
    [data-testid="stMetricLabel"] {
        color: var(--workbench-muted);
    }
    .method-note {
        border-left: 4px solid var(--workbench-blue);
        background: #f4f7fb;
        padding: 0.75rem 1rem;
        color: #334155;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0 1rem 0;
    }
    .small-muted {
        color: #667085;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PAGE_OPTIONS = [
    "Home / Project Overview",
    "Single Company Analysis",
    "DCF Valuation",
    "Credit Risk Score",
    "Peer Comparison",
    "Investment Memo Export",
]

PERCENT_COLUMNS = [
    "revenue_growth",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "roa",
    "roe",
    "debt_to_assets",
    "fcf_margin",
]

MULTIPLE_COLUMNS = ["debt_to_equity", "current_ratio", "quick_ratio", "cash_ratio", "interest_coverage"]


@st.cache_data(show_spinner=False, ttl=3600)
def cached_financials(tickers_text: str, prefer_sec: bool, years: int):
    return load_financials(parse_tickers(tickers_text), prefer_sec=prefer_sec, years=years)


def _metric_delta(value: object, is_percent: bool = True) -> str | None:
    if not is_valid_number(value):
        return None
    return pct(value) if is_percent else plain_number(value)


def _format_ratio_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    formats: dict[str, str] = {}
    for column in PERCENT_COLUMNS:
        if column in df.columns:
            formats[column] = "{:.1%}"
    for column in MULTIPLE_COLUMNS:
        if column in df.columns:
            formats[column] = "{:.2f}x"
    for column in ["dso", "dio", "dpo", "cash_conversion_cycle"]:
        if column in df.columns:
            formats[column] = "{:.1f}"
    for column in ["revenue", "net_income", "free_cash_flow", "operating_cash_flow"]:
        if column in df.columns:
            formats[column] = "${:,.1f}m"
    return df.style.format(formats, na_rep="n/a")


def _selected_company(ratio_df: pd.DataFrame, label: str = "Company") -> tuple[str, pd.DataFrame, pd.Series]:
    tickers = sorted(ratio_df["ticker"].dropna().unique().tolist())
    ticker = st.selectbox(label, tickers)
    history = ratio_df[ratio_df["ticker"] == ticker].sort_values("fiscal_year")
    latest = history.iloc[-1]
    return ticker, history, latest


def _plot_statement_trends(history: pd.DataFrame) -> go.Figure:
    cols = [column for column in ["revenue", "net_income", "free_cash_flow"] if column in history.columns]
    fig = px.line(history, x="fiscal_year", y=cols, markers=True, title="Revenue, Net Income, and Free Cash Flow")
    fig.update_layout(yaxis_title="USD millions", xaxis_title="Fiscal year", legend_title_text="")
    return fig


def _plot_margin_trends(history: pd.DataFrame) -> go.Figure:
    cols = [column for column in ["gross_margin", "operating_margin", "net_margin", "fcf_margin"] if column in history.columns]
    fig = px.line(history, x="fiscal_year", y=cols, markers=True, title="Margin Trend")
    fig.update_layout(yaxis_title="Margin", xaxis_title="Fiscal year", legend_title_text="")
    fig.update_yaxes(tickformat=".0%")
    return fig


def _plot_credit_gauge(score: float, rating: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"Internal Credit Score: {rating}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f4e79"},
                "steps": [
                    {"range": [0, 45], "color": "#f8d7da"},
                    {"range": [45, 65], "color": "#fff3cd"},
                    {"range": [65, 80], "color": "#dbeafe"},
                    {"range": [80, 100], "color": "#d1e7dd"},
                ],
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=24, r=24, t=56, b=16))
    return fig


def _build_dcf_inputs_from_sidebar(latest: pd.Series, assumptions: dict[str, float]) -> DCFInputs:
    net_debt_default = safe_float(latest.get("total_debt"), 0.0) - safe_float(latest.get("cash"), 0.0)
    shares_default = max(safe_float(latest.get("shares_outstanding"), 1.0), 0.01)

    col1, col2, col3 = st.columns(3)
    with col1:
        base_revenue = st.number_input(
            "Base revenue (USD millions)",
            min_value=0.01,
            value=float(max(safe_float(latest.get("revenue"), 0.01), 0.01)),
            step=1000.0,
        )
    with col2:
        net_debt = st.number_input("Net debt (USD millions)", value=float(net_debt_default), step=1000.0)
    with col3:
        shares = st.number_input("Shares outstanding (millions)", min_value=0.01, value=float(shares_default), step=100.0)

    return DCFInputs(
        base_revenue=base_revenue,
        revenue_growth=assumptions["revenue_growth"],
        operating_margin=assumptions["operating_margin"],
        tax_rate=assumptions["tax_rate"],
        wacc=assumptions["wacc"],
        terminal_growth=assumptions["terminal_growth"],
        net_debt=net_debt,
        shares_outstanding=shares,
        years=int(assumptions["years"]),
        fcf_margin=assumptions.get("fcf_margin"),
        reinvestment_rate=assumptions["reinvestment_rate"],
        current_price=safe_float(latest.get("current_price")),
    )


def render_home(financials: pd.DataFrame, ratio_df: pd.DataFrame, messages: list[str]) -> None:
    st.title("Corporate Valuation & Credit Risk Workbench")
    st.markdown(
        """
        <div class="method-note">
        A corporate finance dashboard for public-company analysis using annual financial statement data,
        ratio diagnostics, a transparent internal credit score, DCF valuation, peer comparison, and memo export.
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest = ratio_df.sort_values("fiscal_year").groupby("ticker", as_index=False).tail(1)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Companies loaded", f"{latest['ticker'].nunique():,.0f}")
    col2.metric("Fiscal years", f"{financials['fiscal_year'].nunique():,.0f}")
    col3.metric("Latest fiscal year", f"{int(financials['fiscal_year'].max())}")
    col4.metric("Live data used", "Yes" if "SEC EDGAR companyfacts" in financials.get("data_source", pd.Series()).unique() else "No")

    st.subheader("Loaded Universe")
    display_cols = ["ticker", "company_name", "sector", "fiscal_year", "revenue", "net_income", "data_source"]
    st.dataframe(_format_ratio_table(latest[display_cols]), use_container_width=True, hide_index=True)

    with st.expander("Finance methodology notes", expanded=True):
        st.markdown(
            """
            - Financial statement analysis computes growth, profitability, liquidity, leverage, cash flow, and DuPont-style ROE metrics.
            - Working capital analysis estimates DSO, DIO, DPO, and the cash conversion cycle from average balance sheet accounts.
            - The credit score is a transparent 0-100 educational framework, not a rating-agency credit rating.
            - The DCF uses explicit revenue growth, margin, tax, WACC, terminal growth, net debt, and share-count assumptions.
            - SEC EDGAR data is used when reachable and mappable; sample fallback data keeps the project demo stable.
            """
        )

    if messages:
        with st.expander("Data load log"):
            for message in messages:
                st.write(message)


def render_single_company(ratio_df: pd.DataFrame) -> None:
    st.title("Single Company Analysis")
    ticker, history, latest = _selected_company(ratio_df)
    st.caption(f"{latest.get('company_name', ticker)} | Source: {latest.get('data_source', 'Unknown')}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Revenue", money(latest.get("revenue")), delta=_metric_delta(latest.get("revenue_growth")))
    col2.metric("Net margin", pct(latest.get("net_margin")), help=RATIO_EXPLANATIONS["net_margin"])
    col3.metric("ROE", pct(latest.get("roe")), help=RATIO_EXPLANATIONS["roe"])
    col4.metric("FCF margin", pct(latest.get("fcf_margin")), help=RATIO_EXPLANATIONS["fcf_margin"])

    chart_col1, chart_col2 = st.columns(2)
    chart_col1.plotly_chart(_plot_statement_trends(history), use_container_width=True)
    chart_col2.plotly_chart(_plot_margin_trends(history), use_container_width=True)

    st.subheader("Ratio Summary")
    ratio_cols = [
        "fiscal_year",
        "revenue_growth",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "roa",
        "roe",
        "debt_to_equity",
        "debt_to_assets",
        "current_ratio",
        "quick_ratio",
        "interest_coverage",
        "fcf_margin",
        "asset_turnover",
        "equity_multiplier",
        "dupont_roe",
    ]
    st.dataframe(_format_ratio_table(history[[column for column in ratio_cols if column in history.columns]]), use_container_width=True, hide_index=True)

    st.subheader("Working Capital and Cash Flow Quality")
    wc_cols = ["fiscal_year", "dso", "dio", "dpo", "cash_conversion_cycle", "net_working_capital", "operating_cash_flow", "net_income", "cfo_to_net_income"]
    st.dataframe(_format_ratio_table(history[[column for column in wc_cols if column in history.columns]]), use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=history["fiscal_year"], y=history["current_assets"], name="Current assets"))
    fig.add_trace(go.Bar(x=history["fiscal_year"], y=history["current_liabilities"], name="Current liabilities"))
    fig.update_layout(barmode="group", title="Current Assets vs Current Liabilities", yaxis_title="USD millions", xaxis_title="Fiscal year")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How to read these metrics"):
        st.markdown(
            """
            - Margins show whether growth is translating into operating and after-tax profit.
            - ROA and asset turnover help distinguish capital efficiency from pure margin strength.
            - Debt-to-equity and debt-to-assets show how much balance sheet risk sits ahead of shareholders.
            - CFO versus net income highlights whether accounting earnings are being converted into cash.
            """
        )


def render_dcf(ratio_df: pd.DataFrame, assumptions: dict[str, float]) -> None:
    st.title("DCF Valuation")
    ticker, _, latest = _selected_company(ratio_df)
    st.caption(f"{latest.get('company_name', ticker)} | Fiscal year {int(latest.get('fiscal_year'))}")

    st.markdown(
        """
        <div class="method-note">
        This DCF is intentionally simple and auditable: revenue is forecast from a base year,
        free cash flow is estimated from either an FCF margin or NOPAT less reinvestment,
        and terminal value is discounted at WACC.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        inputs = _build_dcf_inputs_from_sidebar(latest, assumptions)
        result = run_dcf(inputs)
    except Exception as exc:
        st.error(f"DCF could not be calculated: {exc}")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Enterprise value", money(result.enterprise_value))
    col2.metric("Equity value", money(result.equity_value))
    col3.metric("Fair value / share", f"${result.fair_value_per_share:,.2f}")
    col4.metric("Upside / downside", pct(result.upside_downside) if result.upside_downside is not None else "n/a")

    chart_col1, chart_col2 = st.columns(2)
    fig = px.line(result.forecast, x="year", y=["revenue", "free_cash_flow"], markers=True, title="DCF Forecast")
    fig.update_layout(yaxis_title="USD millions", xaxis_title="Forecast year", legend_title_text="")
    chart_col1.plotly_chart(fig, use_container_width=True)

    bridge = pd.DataFrame(
        {
            "Component": ["PV of explicit FCF", "PV of terminal value", "Net debt", "Equity value"],
            "Value": [result.pv_fcf, result.pv_terminal_value, -inputs.net_debt, result.equity_value],
        }
    )
    bridge_fig = px.bar(bridge, x="Component", y="Value", title="Valuation Bridge")
    bridge_fig.update_layout(yaxis_title="USD millions", xaxis_title="")
    chart_col2.plotly_chart(bridge_fig, use_container_width=True)

    st.subheader("Forecast Detail")
    st.dataframe(_format_ratio_table(result.forecast), use_container_width=True, hide_index=True)

    st.subheader("Sensitivity: Fair Value Per Share")
    st.dataframe(result.sensitivity.style.format("${:,.2f}", na_rep="n/a"), use_container_width=True)


def render_credit(ratio_df: pd.DataFrame) -> None:
    st.title("Credit Risk Score")
    ticker, _, latest = _selected_company(ratio_df)
    credit = calculate_credit_score(latest)

    col1, col2 = st.columns([1, 1])
    col1.plotly_chart(_plot_credit_gauge(credit.score, credit.rating), use_container_width=True)

    categories = pd.DataFrame(
        {"Category": list(credit.category_scores.keys()), "Score": list(credit.category_scores.values())}
    )
    bar = px.bar(categories, x="Category", y="Score", range_y=[0, 20], title="Category Contribution")
    bar.update_layout(yaxis_title="Points out of 20", xaxis_title="")
    col2.plotly_chart(bar, use_container_width=True)

    st.markdown(f"**Rating bucket:** {credit.rating}")
    st.info(credit.disclaimer)
    st.write(credit.explanation)

    driver_col1, driver_col2 = st.columns(2)
    with driver_col1:
        st.subheader("Positive Drivers")
        if credit.strengths:
            for item in credit.strengths:
                st.write(f"- {item}")
        else:
            st.write("No major positive drivers were flagged by the rule set.")
    with driver_col2:
        st.subheader("Watch Items")
        if credit.weaknesses:
            for item in credit.weaknesses:
                st.write(f"- {item}")
        else:
            st.write("No major watch items were flagged by the rule set.")

    credit_cols = [
        "current_ratio",
        "quick_ratio",
        "cash_ratio",
        "debt_to_equity",
        "debt_to_assets",
        "interest_coverage",
        "operating_cash_flow_to_debt",
        "net_margin",
        "roa",
        "roe",
        "fcf_margin",
        "cfo_to_net_income",
    ]
    st.subheader("Underlying Credit Metrics")
    st.dataframe(_format_ratio_table(pd.DataFrame([latest[credit_cols]])), use_container_width=True, hide_index=True)


def render_peer_comparison(ratio_df: pd.DataFrame, assumptions: dict[str, float]) -> pd.DataFrame:
    st.title("Peer Comparison")
    peer_df = latest_peer_snapshot(ratio_df, dcf_assumptions=assumptions)

    st.dataframe(_format_ratio_table(peer_df), use_container_width=True, hide_index=True)

    chart_col1, chart_col2 = st.columns(2)
    credit_fig = px.bar(peer_df, x="ticker", y="credit_score", color="credit_bucket", title="Internal Credit Score by Company")
    credit_fig.update_layout(yaxis_title="Score", xaxis_title="")
    chart_col1.plotly_chart(credit_fig, use_container_width=True)

    scatter = px.scatter(
        peer_df,
        x="debt_to_equity",
        y="net_margin",
        size=np.maximum(ratio_df.groupby("ticker")["revenue"].last().reindex(peer_df["ticker"]).fillna(1), 1),
        color="ticker",
        hover_name="company_name",
        title="Profitability vs Leverage",
    )
    scatter.update_layout(xaxis_title="Debt / equity", yaxis_title="Net margin")
    scatter.update_yaxes(tickformat=".0%")
    chart_col2.plotly_chart(scatter, use_container_width=True)

    margin_fig = px.bar(peer_df, x="ticker", y=["revenue_growth", "net_margin", "fcf_margin"], barmode="group", title="Growth and Margin Comparison")
    margin_fig.update_layout(yaxis_title="Ratio", xaxis_title="", legend_title_text="")
    margin_fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(margin_fig, use_container_width=True)

    return peer_df


def render_memo(ratio_df: pd.DataFrame, assumptions: dict[str, float]) -> None:
    st.title("Investment Memo Export")
    ticker, history, latest = _selected_company(ratio_df)
    credit = calculate_credit_score(latest)
    peer_df = latest_peer_snapshot(ratio_df, dcf_assumptions=assumptions)

    try:
        dcf = run_dcf(dcf_inputs_from_row(latest, assumptions))
    except Exception:
        dcf = None

    memo = generate_investment_memo(latest, history, credit, dcf, peer_df)
    st.download_button(
        "Download Markdown Memo",
        memo,
        file_name=f"{ticker.lower()}_investment_memo.md",
        mime="text/markdown",
    )

    if st.button("Save memo to outputs folder"):
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        path = save_report(memo, OUTPUTS_DIR / f"{ticker.lower()}_investment_memo.md")
        st.success(f"Saved memo to {path}")

    st.markdown(memo)


with st.sidebar:
    st.header("Workbench Inputs")
    tickers_text = st.text_input("Tickers", value="AAPL, MSFT, AMZN, JPM")
    data_mode = st.radio("Data mode", ["SEC EDGAR + sample fallback", "Sample data only"], index=0)
    years = st.slider("Annual periods", min_value=3, max_value=8, value=5)
    page = st.selectbox("Dashboard page", PAGE_OPTIONS)

    st.divider()
    st.subheader("DCF Assumptions")
    revenue_growth = st.slider("Revenue growth", -10.0, 20.0, 5.0, 0.5) / 100
    operating_margin = st.slider("Operating margin", -5.0, 45.0, 18.0, 0.5) / 100
    tax_rate = st.slider("Tax rate", 0.0, 35.0, 21.0, 0.5) / 100
    wacc = st.slider("WACC", 4.0, 16.0, 9.0, 0.25) / 100
    terminal_growth = st.slider("Terminal growth", 0.0, 5.0, 2.5, 0.25) / 100
    use_fcf_margin = st.checkbox("Use FCF margin assumption", value=True)
    if use_fcf_margin:
        fcf_margin = st.slider("FCF margin", -5.0, 35.0, 14.0, 0.5) / 100
        reinvestment_rate = 0.25
    else:
        fcf_margin = None
        reinvestment_rate = st.slider("Reinvestment rate on incremental revenue", 0.0, 80.0, 25.0, 1.0) / 100

assumptions = {
    "revenue_growth": revenue_growth,
    "operating_margin": operating_margin,
    "tax_rate": tax_rate,
    "wacc": wacc,
    "terminal_growth": terminal_growth,
    "years": years,
    "fcf_margin": fcf_margin,
    "reinvestment_rate": reinvestment_rate,
}

prefer_sec = data_mode.startswith("SEC")

with st.spinner("Loading financial statement data..."):
    data_result = cached_financials(tickers_text, prefer_sec=prefer_sec, years=years)

if data_result.financials.empty:
    st.error("No financial data could be loaded. Try the default sample tickers: AAPL, MSFT, AMZN, JPM.")
    st.stop()

ratio_df = calculate_financial_ratios(data_result.financials)
if ratio_df.empty:
    st.error("Financial data loaded, but ratio calculation returned no rows.")
    st.stop()

if page == "Home / Project Overview":
    render_home(data_result.financials, ratio_df, data_result.messages)
elif page == "Single Company Analysis":
    render_single_company(ratio_df)
elif page == "DCF Valuation":
    render_dcf(ratio_df, assumptions)
elif page == "Credit Risk Score":
    render_credit(ratio_df)
elif page == "Peer Comparison":
    render_peer_comparison(ratio_df, assumptions)
elif page == "Investment Memo Export":
    render_memo(ratio_df, assumptions)
