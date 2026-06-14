from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .credit_score import CreditScoreResult
from .utils import money, multiple, pct, plain_number, write_text
from .valuation import DCFResult


def infer_analyst_conclusion(row: pd.Series, credit: CreditScoreResult, dcf: DCFResult | None) -> str:
    valuation_view = "Fairly Valued"
    if dcf and dcf.upside_downside is not None:
        if dcf.upside_downside >= 0.15:
            valuation_view = "Undervalued"
        elif dcf.upside_downside <= -0.15:
            valuation_view = "Overvalued"

    if credit.rating in {"Watchlist", "Stressed"}:
        return f"{valuation_view} / Credit Watchlist"
    if credit.rating == "Strong":
        return f"{valuation_view} / Financially Strong"
    return f"{valuation_view} / Credit Stable"


def _markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    available = [column for column in columns if column in df.columns]
    if not available or df.empty:
        return "n/a"
    table = df[available].copy()
    header = "| " + " | ".join(available) + " |"
    divider = "| " + " | ".join(["---"] * len(available)) + " |"
    rows = []
    for _, row in table.iterrows():
        rows.append("| " + " | ".join(str(row.get(column, "")) for column in available) + " |")
    return "\n".join([header, divider, *rows])


def generate_investment_memo(
    company_row: pd.Series,
    history: pd.DataFrame,
    credit: CreditScoreResult,
    dcf: DCFResult | None,
    peer_table: pd.DataFrame | None = None,
) -> str:
    ticker = str(company_row.get("ticker", "")).upper()
    company_name = company_row.get("company_name", ticker)
    conclusion = infer_analyst_conclusion(company_row, credit, dcf)

    latest_year = company_row.get("fiscal_year", "latest")
    revenue = money(company_row.get("revenue"))
    revenue_growth = pct(company_row.get("revenue_growth"))
    net_margin = pct(company_row.get("net_margin"))
    roa = pct(company_row.get("roa"))
    roe = pct(company_row.get("roe"))
    current_ratio = multiple(company_row.get("current_ratio"))
    debt_to_equity = multiple(company_row.get("debt_to_equity"))
    fcf_margin = pct(company_row.get("fcf_margin"))
    ccc = plain_number(company_row.get("cash_conversion_cycle"))

    if dcf:
        dcf_summary = (
            f"The DCF produces an estimated enterprise value of {money(dcf.enterprise_value)} "
            f"and equity value of {money(dcf.equity_value)}, implying fair value per share of "
            f"${dcf.fair_value_per_share:,.2f}."
        )
        if dcf.upside_downside is not None:
            dcf_summary += f" This compares with the available price at an implied {pct(dcf.upside_downside)} upside/downside."
    else:
        dcf_summary = "DCF valuation could not be completed because required inputs were unavailable."

    peer_section = "Peer comparison was not available."
    if peer_table is not None and not peer_table.empty:
        peer_section = _markdown_table(
            peer_table,
            [
                "ticker",
                "revenue_growth",
                "net_margin",
                "roe",
                "debt_to_equity",
                "current_ratio",
                "fcf_margin",
                "credit_score",
                "credit_bucket",
                "dcf_fair_value",
            ],
        )

    risks = []
    if credit.rating in {"Watchlist", "Stressed"}:
        risks.append("Accounting-based credit score indicates balance sheet or cash flow pressure.")
    if pd.notna(company_row.get("revenue_growth")) and company_row.get("revenue_growth") < 0:
        risks.append("Revenue contracted in the latest fiscal year.")
    if pd.notna(company_row.get("fcf_margin")) and company_row.get("fcf_margin") < 0:
        risks.append("Free cash flow margin is negative.")
    if not risks:
        risks.append("DCF sensitivity remains high to WACC, terminal growth, and margin assumptions.")
        risks.append("SEC tags can differ by issuer, so missing fields should be reviewed against the filing.")
    risk_lines = "".join(f"- {risk}\n" for risk in risks)

    memo = f"""# Investment Memo: {company_name} ({ticker})

## Company Overview
{company_name} is analyzed using the latest available annual financial statement data in the workbench. The current snapshot uses fiscal year {latest_year} and reports revenue of {revenue}.

## Financial Performance
Revenue growth was {revenue_growth}. Net margin was {net_margin}, with ROA of {roa} and ROE of {roe}. These metrics frame the company's growth, operating quality, and ability to convert its asset and equity base into earnings.

## Profitability Analysis
The latest free cash flow margin was {fcf_margin}. The DuPont view decomposes ROE into margin, asset turnover, and leverage, helping separate operating performance from balance sheet leverage.

## Liquidity and Working Capital Analysis
The current ratio was {current_ratio}. The cash conversion cycle was approximately {ccc} days where receivables, inventory, and payable data were available. Operating cash flow is compared with net income to highlight accrual quality.

## Leverage and Credit View
Debt-to-equity was {debt_to_equity}. The internal educational credit score is {credit.score:.1f}/100, mapped to a **{credit.rating}** bucket.

{credit.explanation}

> {credit.disclaimer}

## DCF Valuation Summary
{dcf_summary}

## Peer Comparison
{peer_section}

## Key Risks
{risk_lines}
## Analyst-Style Conclusion
**{conclusion}.** This conclusion is generated from the workbench's valuation and internal credit framework and should be reviewed against primary filings, management commentary, and market conditions before any real-world decision.
"""
    return memo


def save_report(markdown: str, path: Path) -> Path:
    write_text(path, markdown)
    return path
