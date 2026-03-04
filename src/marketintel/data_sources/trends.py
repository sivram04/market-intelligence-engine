from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd


def fetch_google_trends_daily(
    keyword: str,
    start: date,
    end: date,
    geo: str = "US",
) -> pd.DataFrame:
    """
    Placeholder module to satisfy the architecture.
    Not used in the required pipeline stages yet.

    Returns schema: date, trends_value
    """
    # Keeping it deterministic + non-blocking for now.
    return pd.DataFrame(columns=["date", "trends_value"])