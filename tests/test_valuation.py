from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.valuation import DCFInputs, run_dcf


def test_dcf_outputs_are_positive_for_profitable_case() -> None:
    inputs = DCFInputs(
        base_revenue=1000,
        revenue_growth=0.05,
        operating_margin=0.20,
        tax_rate=0.21,
        wacc=0.09,
        terminal_growth=0.025,
        net_debt=100,
        shares_outstanding=100,
        years=5,
        fcf_margin=0.12,
        current_price=10,
    )

    result = run_dcf(inputs)

    assert result.enterprise_value > 0
    assert result.equity_value > 0
    assert result.fair_value_per_share > 0
    assert result.upside_downside is not None
    assert result.sensitivity.shape == (5, 3)


def test_dcf_rejects_terminal_growth_above_wacc() -> None:
    inputs = DCFInputs(
        base_revenue=1000,
        revenue_growth=0.05,
        operating_margin=0.20,
        tax_rate=0.21,
        wacc=0.025,
        terminal_growth=0.03,
        net_debt=100,
        shares_outstanding=100,
    )

    with pytest.raises(ValueError):
        run_dcf(inputs)

