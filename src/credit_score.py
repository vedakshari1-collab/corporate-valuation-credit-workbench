from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .utils import is_valid_number, safe_float


DISCLAIMER = (
    "This is an internal educational credit score built from transparent accounting ratios. "
    "It is not a rating-agency credit rating and should not be used as investment advice."
)


@dataclass
class CreditScoreResult:
    score: float
    rating: str
    category_scores: dict[str, float]
    strengths: list[str]
    weaknesses: list[str]
    explanation: str
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "rating": self.rating,
            "category_scores": self.category_scores,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "explanation": self.explanation,
            "disclaimer": self.disclaimer,
        }


def _score_high(value: object, bands: list[tuple[float, float]], missing_score: float = 0.5) -> float:
    if not is_valid_number(value):
        return missing_score
    x = float(value)
    for threshold, score in bands:
        if x >= threshold:
            return score
    return bands[-1][1]


def _score_low(value: object, bands: list[tuple[float, float]], missing_score: float = 0.5) -> float:
    if not is_valid_number(value):
        return missing_score
    x = float(value)
    for threshold, score in bands:
        if x <= threshold:
            return score
    return bands[-1][1]


def _weighted(parts: list[tuple[float, float]]) -> float:
    return sum(score * weight for score, weight in parts) / sum(weight for _, weight in parts)


def calculate_credit_score(row: Mapping[str, object] | pd.Series) -> CreditScoreResult:
    """Build a 0-100 accounting-based credit score from ratio categories."""
    liquidity = _weighted(
        [
            (_score_high(row.get("current_ratio"), [(2.0, 1.0), (1.5, 0.8), (1.0, 0.6), (0.75, 0.35), (-np.inf, 0.1)]), 0.45),
            (_score_high(row.get("quick_ratio"), [(1.5, 1.0), (1.0, 0.8), (0.75, 0.55), (0.5, 0.3), (-np.inf, 0.1)]), 0.35),
            (_score_high(row.get("cash_ratio"), [(0.75, 1.0), (0.4, 0.8), (0.2, 0.55), (0.1, 0.3), (-np.inf, 0.1)]), 0.20),
        ]
    )
    leverage = _weighted(
        [
            (_score_low(row.get("debt_to_equity"), [(0.5, 1.0), (1.0, 0.8), (2.0, 0.55), (3.0, 0.30), (np.inf, 0.1)]), 0.50),
            (_score_low(row.get("debt_to_assets"), [(0.25, 1.0), (0.4, 0.8), (0.6, 0.55), (0.75, 0.30), (np.inf, 0.1)]), 0.50),
        ]
    )
    coverage = _weighted(
        [
            (_score_high(row.get("interest_coverage"), [(10.0, 1.0), (5.0, 0.8), (2.5, 0.55), (1.5, 0.35), (-np.inf, 0.1)]), 0.50),
            (_score_high(row.get("operating_cash_flow_to_debt"), [(0.6, 1.0), (0.35, 0.8), (0.2, 0.55), (0.1, 0.35), (-np.inf, 0.1)]), 0.50),
        ]
    )
    profitability = _weighted(
        [
            (_score_high(row.get("net_margin"), [(0.20, 1.0), (0.12, 0.8), (0.06, 0.55), (0.02, 0.35), (-np.inf, 0.1)]), 0.40),
            (_score_high(row.get("roa"), [(0.12, 1.0), (0.07, 0.8), (0.03, 0.55), (0.0, 0.35), (-np.inf, 0.1)]), 0.30),
            (_score_high(row.get("roe"), [(0.25, 1.0), (0.15, 0.8), (0.08, 0.55), (0.0, 0.35), (-np.inf, 0.1)]), 0.30),
        ]
    )
    cash_flow_quality = _weighted(
        [
            (_score_high(row.get("fcf_margin"), [(0.15, 1.0), (0.08, 0.8), (0.03, 0.55), (0.0, 0.35), (-np.inf, 0.1)]), 0.50),
            (_score_high(row.get("cfo_to_net_income"), [(1.2, 1.0), (1.0, 0.8), (0.75, 0.55), (0.5, 0.35), (-np.inf, 0.1)]), 0.50),
        ]
    )

    category_scores = {
        "Liquidity": round(liquidity * 20, 1),
        "Leverage": round(leverage * 20, 1),
        "Coverage": round(coverage * 20, 1),
        "Profitability": round(profitability * 20, 1),
        "Cash flow quality": round(cash_flow_quality * 20, 1),
    }
    score = round(sum(category_scores.values()), 1)
    rating = rating_bucket(score)
    strengths, weaknesses = _score_drivers(row)

    if strengths and weaknesses:
        explanation = f"Strengths include {', '.join(strengths[:3])}. Watch items include {', '.join(weaknesses[:3])}."
    elif strengths:
        explanation = f"Strengths include {', '.join(strengths[:3])}. No major weaknesses were flagged by the rule set."
    elif weaknesses:
        explanation = f"Watch items include {', '.join(weaknesses[:3])}. Positive drivers were limited or unavailable."
    else:
        explanation = "The score is driven by a mixed or incomplete set of accounting ratios."

    return CreditScoreResult(score, rating, category_scores, strengths, weaknesses, explanation)


def rating_bucket(score: float) -> str:
    if score >= 80:
        return "Strong"
    if score >= 65:
        return "Stable"
    if score >= 45:
        return "Watchlist"
    return "Stressed"


def _score_drivers(row: Mapping[str, object] | pd.Series) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    weaknesses: list[str] = []

    current_ratio = safe_float(row.get("current_ratio"))
    debt_to_assets = safe_float(row.get("debt_to_assets"))
    interest_coverage = safe_float(row.get("interest_coverage"))
    net_margin = safe_float(row.get("net_margin"))
    fcf_margin = safe_float(row.get("fcf_margin"))
    cfo_to_net_income = safe_float(row.get("cfo_to_net_income"))

    if is_valid_number(current_ratio):
        if current_ratio >= 1.5:
            strengths.append("solid short-term liquidity")
        elif current_ratio < 1.0:
            weaknesses.append("thin current ratio")

    if is_valid_number(debt_to_assets):
        if debt_to_assets <= 0.35:
            strengths.append("moderate debt burden")
        elif debt_to_assets >= 0.60:
            weaknesses.append("elevated debt-to-assets")

    if is_valid_number(interest_coverage):
        if interest_coverage >= 5:
            strengths.append("strong interest coverage")
        elif interest_coverage < 2:
            weaknesses.append("weak interest coverage")

    if is_valid_number(net_margin):
        if net_margin >= 0.12:
            strengths.append("healthy net margin")
        elif net_margin < 0.03:
            weaknesses.append("limited profitability")

    if is_valid_number(fcf_margin):
        if fcf_margin >= 0.08:
            strengths.append("positive free cash flow margin")
        elif fcf_margin < 0:
            weaknesses.append("negative free cash flow margin")

    if is_valid_number(cfo_to_net_income):
        if cfo_to_net_income >= 1.0:
            strengths.append("cash earnings conversion above net income")
        elif cfo_to_net_income < 0.75:
            weaknesses.append("cash conversion below net income")

    return strengths, weaknesses

