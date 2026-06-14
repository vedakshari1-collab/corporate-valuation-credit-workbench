from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .utils import ensure_numeric_columns


BASE_NUMERIC_COLUMNS = [
    "fiscal_year",
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "total_debt",
    "cash",
    "current_assets",
    "current_liabilities",
    "accounts_receivable",
    "inventory",
    "accounts_payable",
    "cost_of_revenue",
    "operating_cash_flow",
    "capital_expenditures",
    "interest_expense",
    "income_tax_expense",
    "shares_outstanding",
    "current_price",
]


RATIO_EXPLANATIONS: Mapping[str, str] = {
    "revenue_growth": "Year-over-year revenue growth shows top-line expansion or contraction.",
    "gross_margin": "Gross margin measures profit after direct costs, before operating expenses.",
    "operating_margin": "Operating margin shows how much revenue remains after core operating costs.",
    "net_margin": "Net margin measures after-tax earnings power as a percentage of revenue.",
    "roa": "Return on assets measures earnings generated per dollar of assets.",
    "roe": "Return on equity measures earnings generated per dollar of book equity.",
    "debt_to_equity": "Debt-to-equity compares funded debt with the accounting equity cushion.",
    "debt_to_assets": "Debt-to-assets shows the share of assets financed by debt.",
    "current_ratio": "Current ratio compares short-term assets with short-term liabilities.",
    "quick_ratio": "Quick ratio excludes inventory and focuses on cash plus receivables.",
    "cash_ratio": "Cash ratio is the most conservative liquidity measure.",
    "fcf_margin": "FCF margin measures free cash flow after capital expenditures as a share of revenue.",
    "interest_coverage": "Interest coverage compares operating income with interest expense.",
    "asset_turnover": "Asset turnover measures revenue generated per dollar of assets.",
    "equity_multiplier": "Equity multiplier captures balance sheet leverage in the DuPont framework.",
    "dupont_roe": "DuPont ROE approximates net margin times asset turnover times equity multiplier.",
    "dso": "Days Sales Outstanding estimates how long receivables take to collect.",
    "dio": "Days Inventory Outstanding estimates how long inventory is held before sale.",
    "dpo": "Days Payable Outstanding estimates how long the company takes to pay suppliers.",
    "cash_conversion_cycle": "Cash conversion cycle approximates days cash is tied up in working capital.",
}


def _safe_divide_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan})
    return numerator.astype(float).div(denominator.astype(float))


def _average_with_prior(series: pd.Series) -> pd.Series:
    average = (series + series.shift(1)) / 2
    return average.fillna(series)


def calculate_financial_ratios(financials: pd.DataFrame) -> pd.DataFrame:
    """Calculate financial, working-capital, and DuPont ratios.

    Input monetary values are expected in a consistent unit, usually USD millions.
    Ratios are unitless, so the same logic works for SEC data and the sample data.
    """
    if financials.empty:
        return financials.copy()

    df = financials.copy()
    if "ticker" not in df.columns:
        raise ValueError("financials must include a ticker column")

    df = ensure_numeric_columns(df, BASE_NUMERIC_COLUMNS)
    if "company_name" not in df.columns:
        df["company_name"] = df["ticker"]
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    if "data_source" not in df.columns:
        df["data_source"] = "Unknown"

    df["capital_expenditures"] = df["capital_expenditures"].abs()
    df = df.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for _, group in df.groupby("ticker", sort=False):
        g = group.copy()
        avg_assets = _average_with_prior(g["total_assets"])
        avg_equity = _average_with_prior(g["total_equity"])
        avg_receivables = _average_with_prior(g["accounts_receivable"])
        avg_inventory = _average_with_prior(g["inventory"])
        avg_payables = _average_with_prior(g["accounts_payable"])

        g["revenue_growth"] = g["revenue"].pct_change()
        g["gross_margin"] = _safe_divide_series(g["gross_profit"], g["revenue"])
        g["operating_margin"] = _safe_divide_series(g["operating_income"], g["revenue"])
        g["net_margin"] = _safe_divide_series(g["net_income"], g["revenue"])
        g["roa"] = _safe_divide_series(g["net_income"], avg_assets)
        g["roe"] = _safe_divide_series(g["net_income"], avg_equity)
        g["debt_to_equity"] = _safe_divide_series(g["total_debt"], g["total_equity"])
        g["debt_to_assets"] = _safe_divide_series(g["total_debt"], g["total_assets"])
        g["current_ratio"] = _safe_divide_series(g["current_assets"], g["current_liabilities"])
        g["quick_ratio"] = _safe_divide_series(g["cash"] + g["accounts_receivable"], g["current_liabilities"])
        g["cash_ratio"] = _safe_divide_series(g["cash"], g["current_liabilities"])
        g["free_cash_flow"] = g["operating_cash_flow"] - g["capital_expenditures"]
        g["fcf_margin"] = _safe_divide_series(g["free_cash_flow"], g["revenue"])
        g["interest_coverage"] = _safe_divide_series(g["operating_income"], g["interest_expense"].abs())
        g["asset_turnover"] = _safe_divide_series(g["revenue"], avg_assets)
        g["equity_multiplier"] = _safe_divide_series(avg_assets, avg_equity)
        g["dupont_roe"] = g["net_margin"] * g["asset_turnover"] * g["equity_multiplier"]
        g["operating_cash_flow_to_debt"] = _safe_divide_series(g["operating_cash_flow"], g["total_debt"])
        g["cfo_to_net_income"] = _safe_divide_series(g["operating_cash_flow"], g["net_income"])

        # Working capital metrics use average balances to approximate period activity.
        g["dso"] = _safe_divide_series(avg_receivables, g["revenue"]) * 365
        g["dio"] = _safe_divide_series(avg_inventory, g["cost_of_revenue"]) * 365
        g["dpo"] = _safe_divide_series(avg_payables, g["cost_of_revenue"]) * 365
        g["cash_conversion_cycle"] = g["dso"] + g["dio"] - g["dpo"]
        g["net_working_capital"] = g["current_assets"] - g["current_liabilities"]
        g["cfo_minus_net_income"] = g["operating_cash_flow"] - g["net_income"]

        frames.append(g)

    return pd.concat(frames, ignore_index=True)


def latest_ratio_row(ratio_df: pd.DataFrame, ticker: str) -> pd.Series:
    rows = ratio_df[ratio_df["ticker"].str.upper() == ticker.upper()].sort_values("fiscal_year")
    if rows.empty:
        raise ValueError(f"No ratio data available for {ticker}")
    return rows.iloc[-1]


def ratio_display_table(ratio_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
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
        "fcf_margin",
        "interest_coverage",
        "cash_conversion_cycle",
    ]
    available = [column for column in columns if column in ratio_df.columns]
    return ratio_df[available].copy()

