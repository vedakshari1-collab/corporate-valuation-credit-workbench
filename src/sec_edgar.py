from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
STOOQ_PRICE_URL = "https://stooq.com/q/l/?s={ticker}.us&f=sd2t2ohlcv&h&e=csv"

DEFAULT_USER_AGENT = (
    "Corporate Valuation Credit Risk Workbench research app "
    "(set SEC_USER_AGENT env var with your contact email)"
)


CONCEPT_TAGS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "accounts_receivable": ["AccountsReceivableNetCurrent", "AccountsReceivableNet"],
    "inventory": ["InventoryNet", "InventoryFinishedGoodsNetOfReserves"],
    "accounts_payable": ["AccountsPayableCurrent", "AccountsPayableTradeCurrent"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capital_expenditures": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "interest_expense": ["InterestExpenseNonOperating", "InterestExpense"],
    "income_tax_expense": ["IncomeTaxExpenseBenefit"],
    "shares_outstanding": [
        "EntityCommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
}

DEBT_COMPONENT_TAGS = [
    "ShortTermBorrowings",
    "ShortTermDebt",
    "LongTermDebtCurrent",
    "LongTermDebtAndFinanceLeaseObligationsCurrent",
    "LongTermDebtNoncurrent",
    "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
    "FinanceLeaseLiabilityCurrent",
    "FinanceLeaseLiabilityNoncurrent",
]

DEBT_FALLBACK_TAGS = [
    "LongTermDebt",
    "LongTermDebtAndFinanceLeaseObligations",
    "DebtCurrent",
]

MONETARY_COLUMNS = [
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
]


class EdgarDataError(RuntimeError):
    """Raised when SEC data cannot be fetched or mapped into the project schema."""


def _headers() -> dict[str, str]:
    return {"User-Agent": os.getenv("SEC_USER_AGENT", DEFAULT_USER_AGENT)}


def _get_json(url: str, timeout: int = 20) -> dict[str, Any]:
    response = requests.get(url, headers=_headers(), timeout=timeout)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=1)
def ticker_cik_map() -> dict[str, str]:
    data = _get_json(SEC_TICKERS_URL)
    mapping: dict[str, str] = {}
    for item in data.values():
        ticker = str(item.get("ticker", "")).upper()
        cik = str(item.get("cik_str", "")).zfill(10)
        if ticker and cik:
            mapping[ticker] = cik
    return mapping


def cik_for_ticker(ticker: str) -> str:
    mapping = ticker_cik_map()
    cik = mapping.get(ticker.upper())
    if not cik:
        raise EdgarDataError(f"No SEC CIK found for ticker {ticker}.")
    return cik


def fetch_company_facts(ticker: str) -> dict[str, Any]:
    cik = cik_for_ticker(ticker)
    return _get_json(SEC_FACTS_URL.format(cik=cik))


def _unit_items(tag_payload: dict[str, Any], preferred_units: Iterable[str]) -> list[dict[str, Any]]:
    units = tag_payload.get("units", {})
    for unit in preferred_units:
        if unit in units:
            return units[unit]
    for unit_items in units.values():
        return unit_items
    return []


def _annual_series(company_facts: dict[str, Any], tag: str, units: Iterable[str] = ("USD",)) -> pd.Series:
    facts = company_facts.get("facts", {}).get("us-gaap", {})
    tag_payload = facts.get(tag)
    if not tag_payload:
        return pd.Series(dtype=float)

    records: list[dict[str, Any]] = []
    for item in _unit_items(tag_payload, units):
        form = str(item.get("form", ""))
        fp = str(item.get("fp", ""))
        fy = item.get("fy")
        value = item.get("val")
        filed = item.get("filed", "")

        if value is None or fy is None:
            continue
        if form not in {"10-K", "10-K/A", "20-F", "40-F"}:
            continue
        if fp not in {"FY", ""} and form in {"10-K", "10-K/A"}:
            continue
        try:
            fiscal_year = int(fy)
            records.append({"fiscal_year": fiscal_year, "value": float(value), "filed": filed})
        except (TypeError, ValueError):
            continue

    if not records:
        return pd.Series(dtype=float)

    frame = pd.DataFrame(records).sort_values(["fiscal_year", "filed"])
    latest = frame.groupby("fiscal_year", as_index=False).tail(1)
    return latest.set_index("fiscal_year")["value"].sort_index()


def _series_for_any_tag(company_facts: dict[str, Any], tags: Iterable[str], units: Iterable[str] = ("USD",)) -> pd.Series:
    for tag in tags:
        series = _annual_series(company_facts, tag, units=units)
        if not series.dropna().empty:
            return series
    return pd.Series(dtype=float)


def _sum_series_for_tags(company_facts: dict[str, Any], tags: Iterable[str]) -> pd.Series:
    series_list = [_annual_series(company_facts, tag, units=("USD",)) for tag in tags]
    series_list = [series for series in series_list if not series.dropna().empty]
    if not series_list:
        return pd.Series(dtype=float)
    combined = pd.concat(series_list, axis=1).fillna(0).sum(axis=1)
    combined = combined.replace({0: np.nan})
    return combined


def fetch_latest_price(ticker: str) -> float | None:
    """Best-effort public price lookup. Returns None if unavailable."""
    try:
        response = requests.get(STOOQ_PRICE_URL.format(ticker=ticker.lower()), timeout=10)
        response.raise_for_status()
        lines = response.text.strip().splitlines()
        if len(lines) < 2:
            return None
        parts = lines[1].split(",")
        close = parts[6] if len(parts) > 6 else ""
        value = float(close)
        if value > 0:
            return value
    except Exception:
        return None
    return None


def company_facts_to_financials(ticker: str, company_facts: dict[str, Any], years: int = 5) -> pd.DataFrame:
    facts = company_facts.get("facts", {}).get("us-gaap", {})
    if not facts:
        raise EdgarDataError(f"No us-gaap facts found for {ticker}.")

    series_by_column: dict[str, pd.Series] = {}
    for column, tags in CONCEPT_TAGS.items():
        units = ("shares",) if column == "shares_outstanding" else ("USD",)
        series_by_column[column] = _series_for_any_tag(company_facts, tags, units=units)

    debt_components = _sum_series_for_tags(company_facts, DEBT_COMPONENT_TAGS)
    if debt_components.dropna().empty:
        debt_components = _series_for_any_tag(company_facts, DEBT_FALLBACK_TAGS, units=("USD",))
    series_by_column["total_debt"] = debt_components

    fiscal_years = sorted(set().union(*(series.index.tolist() for series in series_by_column.values())))
    if not fiscal_years:
        raise EdgarDataError(f"No annual facts mapped for {ticker}.")
    fiscal_years = fiscal_years[-years:]

    rows: list[dict[str, Any]] = []
    company_name = company_facts.get("entityName", ticker.upper())
    current_price = fetch_latest_price(ticker)

    for fiscal_year in fiscal_years:
        row: dict[str, Any] = {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "sector": "Public company",
            "fiscal_year": fiscal_year,
            "data_source": "SEC EDGAR companyfacts",
            "current_price": current_price,
        }
        for column, series in series_by_column.items():
            value = series.get(fiscal_year, np.nan)
            row[column] = value

        for column in MONETARY_COLUMNS:
            if pd.notna(row.get(column)):
                row[column] = float(row[column]) / 1_000_000
        if pd.notna(row.get("shares_outstanding")):
            row["shares_outstanding"] = float(row["shares_outstanding"]) / 1_000_000

        if pd.isna(row.get("gross_profit")) and pd.notna(row.get("revenue")) and pd.notna(row.get("cost_of_revenue")):
            row["gross_profit"] = row["revenue"] - row["cost_of_revenue"]
        if pd.isna(row.get("total_liabilities")) and pd.notna(row.get("total_assets")) and pd.notna(row.get("total_equity")):
            row["total_liabilities"] = row["total_assets"] - row["total_equity"]
        if pd.notna(row.get("capital_expenditures")):
            row["capital_expenditures"] = abs(float(row["capital_expenditures"]))

        rows.append(row)

    df = pd.DataFrame(rows)
    if df["revenue"].dropna().empty and df["net_income"].dropna().empty:
        raise EdgarDataError(f"SEC facts for {ticker} did not include enough statement data.")
    return df


def fetch_company_financials(ticker: str, years: int = 5) -> pd.DataFrame:
    company_facts = fetch_company_facts(ticker)
    return company_facts_to_financials(ticker, company_facts, years=years)

