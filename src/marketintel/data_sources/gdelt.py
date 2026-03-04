from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
import requests


GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass(frozen=True)
class GdeltArticle:
    date: date
    title: str
    url: str
    source: str


def fetch_gdelt_headlines_daily(
    query: str,
    start: date,
    end: date,
    max_records_per_day: int,
    timeout_s: int,
    max_retries: int,
    backoff_s: float,
) -> pd.DataFrame:
    """
    Pulls GDELT DOC headlines day-by-day to reduce API fragility.
    Output columns: date, title, url, source
    """
    rows: List[Dict[str, str]] = []

    for d in pd.date_range(start=start, end=end, freq="D"):
        day = d.date()
        day_rows = _fetch_one_day(
            query=query,
            day=day,
            max_records=max_records_per_day,
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_s=backoff_s,
        )
        rows.extend(day_rows)

    df = pd.DataFrame(rows)
    if df.empty:
        # Keep deterministic schema even if GDELT is down
        return pd.DataFrame(columns=["date", "title", "url", "source"])

    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _fetch_one_day(
    query: str,
    day: date,
    max_records: int,
    timeout_s: int,
    max_retries: int,
    backoff_s: float,
) -> List[Dict[str, str]]:
    """
    GDELT DOC API supports startdatetime/enddatetime.
    We request JSON and extract articles.
    """
    start_dt = f"{day.strftime('%Y%m%d')}000000"
    end_dt = f"{day.strftime('%Y%m%d')}235959"

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": start_dt,
        "enddatetime": end_dt,
        "maxrecords": int(max_records),
        "sort": "HybridRel",
    }

    session = requests.Session()
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(GDELT_DOC_ENDPOINT, params=params, timeout=timeout_s)
            r.raise_for_status()
            payload = r.json()

            articles = payload.get("articles", [])
            out: List[Dict[str, str]] = []
            for a in articles:
                out.append(
                    {
                        "date": str(day),
                        "title": a.get("title", "") or "",
                        "url": a.get("url", "") or "",
                        "source": a.get("sourceCountry", "") or a.get("source", "") or "",
                    }
                )
            return out

        except Exception as e:
            last_err = e
            # basic exponential-ish backoff without sleeping too long
            import time
            time.sleep(backoff_s * attempt)

    # If all retries fail, return empty for that day (pipeline still completes)
    return []