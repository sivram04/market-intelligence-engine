from __future__ import annotations

from datetime import date
from typing import Tuple

import pandas as pd
import yfinance as yf


def fetch_spy_vix(ticker_spy: str, ticker_vix: str, start: date, end: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two daily OHLCV frames with a Date index (timezone-naive), columns standardized.
    """
    spy = yf.download(ticker_spy, start=str(start), end=str(end), auto_adjust=False, progress=False)
    vix = yf.download(ticker_vix, start=str(start), end=str(end), auto_adjust=False, progress=False)

    if spy.empty:
        raise RuntimeError(f"yfinance returned empty data for {ticker_spy}.")
    if vix.empty:
        raise RuntimeError(f"yfinance returned empty data for {ticker_vix}.")

    spy = _standardize_yf(spy)
    vix = _standardize_yf(vix)

    return spy, vix


def _standardize_yf(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "date"

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(x).lower().strip() for x in out.columns.get_level_values(0)]
    else:
        out.columns = [str(c).lower().strip() for c in out.columns]

    return out