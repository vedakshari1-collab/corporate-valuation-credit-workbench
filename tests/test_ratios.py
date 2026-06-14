from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.financial_ratios import calculate_financial_ratios


def test_ratio_calculations_and_working_capital() -> None:
    data = pd.DataFrame(
        [
            {
                "ticker": "TST",
                "company_name": "Test Co",
                "sector": "Industrial",
                "fiscal_year": 2023,
                "revenue": 1000,
                "gross_profit": 400,
                "operating_income": 200,
                "net_income": 100,
                "total_assets": 1000,
                "total_liabilities": 600,
                "total_equity": 400,
                "total_debt": 300,
                "cash": 100,
                "current_assets": 500,
                "current_liabilities": 250,
                "accounts_receivable": 100,
                "inventory": 80,
                "accounts_payable": 60,
                "cost_of_revenue": 600,
                "operating_cash_flow": 150,
                "capital_expenditures": 50,
                "interest_expense": 20,
                "income_tax_expense": 30,
                "shares_outstanding": 100,
                "current_price": 10,
            },
            {
                "ticker": "TST",
                "company_name": "Test Co",
                "sector": "Industrial",
                "fiscal_year": 2024,
                "revenue": 1100,
                "gross_profit": 462,
                "operating_income": 220,
                "net_income": 121,
                "total_assets": 1100,
                "total_liabilities": 650,
                "total_equity": 450,
                "total_debt": 330,
                "cash": 120,
                "current_assets": 550,
                "current_liabilities": 275,
                "accounts_receivable": 110,
                "inventory": 88,
                "accounts_payable": 66,
                "cost_of_revenue": 638,
                "operating_cash_flow": 170,
                "capital_expenditures": 55,
                "interest_expense": 22,
                "income_tax_expense": 35,
                "shares_outstanding": 100,
                "current_price": 12,
            },
        ]
    )

    ratios = calculate_financial_ratios(data)
    latest = ratios.iloc[-1]

    assert round(latest["revenue_growth"], 4) == 0.1
    assert round(latest["gross_margin"], 4) == 0.42
    assert round(latest["current_ratio"], 2) == 2.0
    assert latest["free_cash_flow"] == 115
    assert round(latest["interest_coverage"], 2) == 10.0
    assert round(latest["cash_conversion_cycle"], 1) == round(latest["dso"] + latest["dio"] - latest["dpo"], 1)

