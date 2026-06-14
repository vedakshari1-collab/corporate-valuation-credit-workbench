from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .sec_edgar import EdgarDataError, fetch_company_financials
from .utils import DATA_DIR, parse_tickers


@dataclass
class DataLoadResult:
    financials: pd.DataFrame
    messages: list[str]
    used_live_data: bool


def load_sample_financials(path: Path | None = None) -> pd.DataFrame:
    sample_path = path or DATA_DIR / "sample_financials.csv"
    df = pd.read_csv(sample_path)
    df["ticker"] = df["ticker"].str.upper()
    if "data_source" not in df.columns:
        df["data_source"] = "Sample fallback data"
    return df


def load_sample_peers(path: Path | None = None) -> pd.DataFrame:
    sample_path = path or DATA_DIR / "sample_peers.csv"
    df = pd.read_csv(sample_path)
    df["ticker"] = df["ticker"].str.upper()
    return df


def filter_sample_financials(tickers: list[str]) -> pd.DataFrame:
    sample = load_sample_financials()
    if not tickers:
        return sample
    return sample[sample["ticker"].isin([ticker.upper() for ticker in tickers])].copy()


def load_financials(tickers: str | list[str], prefer_sec: bool = True, years: int = 5) -> DataLoadResult:
    """Load company financials from SEC EDGAR with sample fallback.

    The fallback is intentionally explicit: every returned row carries a data_source value
    so a user can see whether they are looking at live SEC data or sample data.
    """
    parsed_tickers = parse_tickers(tickers)
    if not parsed_tickers:
        parsed_tickers = ["AAPL", "MSFT", "AMZN", "JPM"]

    frames: list[pd.DataFrame] = []
    messages: list[str] = []
    used_live_data = False

    if prefer_sec:
        for ticker in parsed_tickers:
            try:
                live = fetch_company_financials(ticker, years=years)
                frames.append(live)
                used_live_data = True
                messages.append(f"{ticker}: loaded annual financials from SEC EDGAR.")
            except Exception as exc:
                sample = filter_sample_financials([ticker])
                if not sample.empty:
                    frames.append(sample)
                    messages.append(f"{ticker}: SEC request or mapping failed; using sample fallback data. Detail: {exc}")
                else:
                    messages.append(f"{ticker}: no SEC data or sample fallback available. Detail: {exc}")
    else:
        sample = filter_sample_financials(parsed_tickers)
        frames.append(sample)
        missing = sorted(set(parsed_tickers) - set(sample["ticker"].unique()))
        messages.append("Sample data mode is active.")
        if missing:
            messages.append(f"No sample data available for: {', '.join(missing)}.")

    if not frames:
        return DataLoadResult(pd.DataFrame(), messages, used_live_data)

    financials = pd.concat(frames, ignore_index=True)
    financials = financials.drop_duplicates(subset=["ticker", "fiscal_year"], keep="first")
    financials = financials.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)
    return DataLoadResult(financials, messages, used_live_data)


def available_sample_tickers() -> list[str]:
    return sorted(load_sample_financials()["ticker"].unique().tolist())

