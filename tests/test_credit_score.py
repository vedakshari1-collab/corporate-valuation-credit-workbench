from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.credit_score import calculate_credit_score


def test_credit_score_strong_company() -> None:
    row = {
        "current_ratio": 2.1,
        "quick_ratio": 1.6,
        "cash_ratio": 0.8,
        "debt_to_equity": 0.4,
        "debt_to_assets": 0.2,
        "interest_coverage": 12,
        "operating_cash_flow_to_debt": 0.7,
        "net_margin": 0.22,
        "roa": 0.13,
        "roe": 0.27,
        "fcf_margin": 0.16,
        "cfo_to_net_income": 1.3,
    }

    result = calculate_credit_score(row)

    assert result.score >= 80
    assert result.rating == "Strong"
    assert result.strengths


def test_credit_score_stressed_company() -> None:
    row = {
        "current_ratio": 0.6,
        "quick_ratio": 0.4,
        "cash_ratio": 0.05,
        "debt_to_equity": 4.0,
        "debt_to_assets": 0.85,
        "interest_coverage": 0.8,
        "operating_cash_flow_to_debt": 0.03,
        "net_margin": -0.05,
        "roa": -0.02,
        "roe": -0.10,
        "fcf_margin": -0.08,
        "cfo_to_net_income": 0.4,
    }

    result = calculate_credit_score(row)

    assert result.score < 45
    assert result.rating == "Stressed"
    assert result.weaknesses
