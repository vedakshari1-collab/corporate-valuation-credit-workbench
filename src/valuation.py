from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import safe_float


@dataclass(frozen=True)
class DCFInputs:
    base_revenue: float
    revenue_growth: float
    operating_margin: float
    tax_rate: float
    wacc: float
    terminal_growth: float
    net_debt: float
    shares_outstanding: float
    years: int = 5
    fcf_margin: float | None = None
    reinvestment_rate: float = 0.25
    current_price: float | None = None


@dataclass
class DCFResult:
    inputs: DCFInputs
    forecast: pd.DataFrame
    terminal_value: float
    pv_terminal_value: float
    pv_fcf: float
    enterprise_value: float
    equity_value: float
    fair_value_per_share: float
    upside_downside: float | None
    sensitivity: pd.DataFrame


def validate_dcf_inputs(inputs: DCFInputs) -> None:
    if inputs.base_revenue <= 0:
        raise ValueError("Base revenue must be positive.")
    if inputs.shares_outstanding <= 0:
        raise ValueError("Shares outstanding must be positive.")
    if inputs.years < 1:
        raise ValueError("Forecast period must be at least one year.")
    if inputs.wacc <= inputs.terminal_growth:
        raise ValueError("WACC must be greater than terminal growth.")
    if inputs.tax_rate < 0 or inputs.tax_rate > 1:
        raise ValueError("Tax rate must be between 0% and 100%.")


def build_dcf_forecast(inputs: DCFInputs) -> pd.DataFrame:
    validate_dcf_inputs(inputs)
    rows: list[dict[str, float]] = []
    prior_revenue = inputs.base_revenue

    for year in range(1, inputs.years + 1):
        revenue = prior_revenue * (1 + inputs.revenue_growth)
        ebit = revenue * inputs.operating_margin
        nopat = ebit * (1 - inputs.tax_rate)

        if inputs.fcf_margin is not None:
            free_cash_flow = revenue * inputs.fcf_margin
            reinvestment = np.nan
        else:
            # Reinvestment is modeled as a percentage of incremental revenue.
            reinvestment = max(revenue - prior_revenue, 0) * inputs.reinvestment_rate
            free_cash_flow = nopat - reinvestment

        discount_factor = 1 / ((1 + inputs.wacc) ** year)
        rows.append(
            {
                "year": year,
                "revenue": revenue,
                "ebit": ebit,
                "nopat": nopat,
                "reinvestment": reinvestment,
                "free_cash_flow": free_cash_flow,
                "discount_factor": discount_factor,
                "pv_free_cash_flow": free_cash_flow * discount_factor,
            }
        )
        prior_revenue = revenue

    return pd.DataFrame(rows)


def run_dcf(inputs: DCFInputs, build_sensitivity: bool = True) -> DCFResult:
    forecast = build_dcf_forecast(inputs)
    final_fcf = float(forecast.iloc[-1]["free_cash_flow"])
    terminal_value = final_fcf * (1 + inputs.terminal_growth) / (inputs.wacc - inputs.terminal_growth)
    pv_terminal_value = terminal_value / ((1 + inputs.wacc) ** inputs.years)
    pv_fcf = float(forecast["pv_free_cash_flow"].sum())
    enterprise_value = pv_fcf + pv_terminal_value
    equity_value = enterprise_value - inputs.net_debt
    fair_value_per_share = equity_value / inputs.shares_outstanding

    current_price = safe_float(inputs.current_price)
    upside_downside = None
    if not np.isnan(current_price) and current_price > 0:
        upside_downside = fair_value_per_share / current_price - 1

    sensitivity = sensitivity_table(inputs) if build_sensitivity else pd.DataFrame()
    return DCFResult(
        inputs=inputs,
        forecast=forecast,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        pv_fcf=pv_fcf,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        fair_value_per_share=fair_value_per_share,
        upside_downside=upside_downside,
        sensitivity=sensitivity,
    )


def sensitivity_table(
    inputs: DCFInputs,
    wacc_values: Iterable[float] | None = None,
    terminal_growth_values: Iterable[float] | None = None,
) -> pd.DataFrame:
    if wacc_values is None:
        wacc_values = [inputs.wacc - 0.01, inputs.wacc - 0.005, inputs.wacc, inputs.wacc + 0.005, inputs.wacc + 0.01]
    if terminal_growth_values is None:
        terminal_growth_values = [
            inputs.terminal_growth - 0.005,
            inputs.terminal_growth,
            inputs.terminal_growth + 0.005,
        ]

    rows: dict[str, dict[str, float]] = {}
    for wacc in wacc_values:
        row_label = f"{wacc:.1%}"
        rows[row_label] = {}
        for terminal_growth in terminal_growth_values:
            column_label = f"{terminal_growth:.1%}"
            if wacc <= terminal_growth:
                rows[row_label][column_label] = np.nan
                continue
            scenario_inputs = DCFInputs(
                base_revenue=inputs.base_revenue,
                revenue_growth=inputs.revenue_growth,
                operating_margin=inputs.operating_margin,
                tax_rate=inputs.tax_rate,
                wacc=wacc,
                terminal_growth=terminal_growth,
                net_debt=inputs.net_debt,
                shares_outstanding=inputs.shares_outstanding,
                years=inputs.years,
                fcf_margin=inputs.fcf_margin,
                reinvestment_rate=inputs.reinvestment_rate,
                current_price=inputs.current_price,
            )
            scenario = run_dcf(scenario_inputs, build_sensitivity=False)
            rows[row_label][column_label] = scenario.fair_value_per_share

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "WACC"
    table.columns.name = "Terminal growth"
    return table


def dcf_inputs_from_row(row: pd.Series, assumptions: dict[str, float]) -> DCFInputs:
    net_debt = safe_float(row.get("total_debt"), 0.0) - safe_float(row.get("cash"), 0.0)
    return DCFInputs(
        base_revenue=max(safe_float(row.get("revenue"), 0.0), 0.01),
        revenue_growth=assumptions.get("revenue_growth", 0.04),
        operating_margin=assumptions.get("operating_margin", safe_float(row.get("operating_margin"), 0.15)),
        tax_rate=assumptions.get("tax_rate", 0.21),
        wacc=assumptions.get("wacc", 0.09),
        terminal_growth=assumptions.get("terminal_growth", 0.025),
        net_debt=assumptions.get("net_debt", net_debt),
        shares_outstanding=max(assumptions.get("shares_outstanding", safe_float(row.get("shares_outstanding"), 0.0)), 0.01),
        years=int(assumptions.get("years", 5)),
        fcf_margin=assumptions.get("fcf_margin"),
        reinvestment_rate=assumptions.get("reinvestment_rate", 0.25),
        current_price=safe_float(row.get("current_price")),
    )

