from __future__ import annotations

import pandas as pd

from .credit_score import calculate_credit_score
from .utils import latest_rows_by_ticker, safe_float
from .valuation import dcf_inputs_from_row, run_dcf


PEER_METRICS = [
    "ticker",
    "company_name",
    "sector",
    "fiscal_year",
    "revenue_growth",
    "net_margin",
    "roe",
    "roa",
    "debt_to_equity",
    "current_ratio",
    "fcf_margin",
    "credit_score",
    "credit_bucket",
    "dcf_fair_value",
    "current_price",
]


def latest_peer_snapshot(ratio_df: pd.DataFrame, dcf_assumptions: dict[str, float] | None = None) -> pd.DataFrame:
    if ratio_df.empty:
        return pd.DataFrame(columns=PEER_METRICS)

    latest = latest_rows_by_ticker(ratio_df)
    rows: list[dict[str, object]] = []
    for _, row in latest.iterrows():
        credit = calculate_credit_score(row)
        dcf_fair_value = float("nan")
        if dcf_assumptions is not None:
            try:
                result = run_dcf(dcf_inputs_from_row(row, dcf_assumptions))
                dcf_fair_value = result.fair_value_per_share
            except Exception:
                dcf_fair_value = float("nan")

        rows.append(
            {
                "ticker": row.get("ticker"),
                "company_name": row.get("company_name"),
                "sector": row.get("sector"),
                "fiscal_year": row.get("fiscal_year"),
                "revenue_growth": row.get("revenue_growth"),
                "net_margin": row.get("net_margin"),
                "roe": row.get("roe"),
                "roa": row.get("roa"),
                "debt_to_equity": row.get("debt_to_equity"),
                "current_ratio": row.get("current_ratio"),
                "fcf_margin": row.get("fcf_margin"),
                "credit_score": credit.score,
                "credit_bucket": credit.rating,
                "dcf_fair_value": dcf_fair_value,
                "current_price": safe_float(row.get("current_price")),
            }
        )

    return pd.DataFrame(rows)[PEER_METRICS]


def peer_metric_rank(peer_df: pd.DataFrame, metric: str, ascending: bool = False) -> pd.DataFrame:
    if peer_df.empty or metric not in peer_df.columns:
        return pd.DataFrame()
    cols = ["ticker", "company_name", metric]
    return peer_df[cols].sort_values(metric, ascending=ascending).reset_index(drop=True)

