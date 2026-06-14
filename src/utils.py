from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def parse_tickers(raw: str | Iterable[str]) -> list[str]:
    """Normalize a comma, space, or newline separated ticker list."""
    if isinstance(raw, str):
        parts = re.split(r"[\s,;]+", raw)
    else:
        parts = list(raw)

    tickers: list[str] = []
    for part in parts:
        ticker = str(part).strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def safe_divide(numerator: float | int | None, denominator: float | int | None, default: float = np.nan) -> float:
    """Divide safely and return NaN by default for missing or zero denominators."""
    try:
        if numerator is None or denominator is None:
            return default
        if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0:
            return default
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def safe_float(value: object, default: float = np.nan) -> float:
    """Convert a value to float while keeping missing values harmless."""
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def is_valid_number(value: object) -> bool:
    try:
        return value is not None and not pd.isna(value) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def ensure_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a copy with requested columns present and numeric."""
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def latest_rows_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    sort_cols = ["ticker", "fiscal_year"] if "fiscal_year" in df.columns else ["ticker"]
    sorted_df = df.sort_values(sort_cols)
    return sorted_df.groupby("ticker", as_index=False).tail(1).reset_index(drop=True)


def pct(value: object, digits: int = 1) -> str:
    if not is_valid_number(value):
        return "n/a"
    return f"{float(value) * 100:.{digits}f}%"


def multiple(value: object, digits: int = 2) -> str:
    if not is_valid_number(value):
        return "n/a"
    return f"{float(value):.{digits}f}x"


def money(value: object, digits: int = 1, suffix: str = "m") -> str:
    if not is_valid_number(value):
        return "n/a"
    return f"${float(value):,.{digits}f}{suffix}"


def plain_number(value: object, digits: int = 1) -> str:
    if not is_valid_number(value):
        return "n/a"
    return f"{float(value):,.{digits}f}"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

