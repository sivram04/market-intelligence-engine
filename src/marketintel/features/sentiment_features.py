from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..utils.text import clean_headline


_analyzer = SentimentIntensityAnalyzer()


def score_headlines(df_headlines: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns expected: date, title, url, source
    Output adds: title_clean, sentiment_compound
    """
    if df_headlines is None or df_headlines.empty:
        out = (df_headlines.copy() if df_headlines is not None else pd.DataFrame())
        out["title_clean"] = []
        out["sentiment_compound"] = []
        return out

    out = df_headlines.copy()

    # Defensive: handle NaN titles and non-strings
    out["title"] = out["title"].fillna("").astype(str)

    # Drop empty titles early (saves scoring time)
    out = out[out["title"].str.strip().ne("")].copy()

    # Normalize date type (keep as date string or date; groupby will work consistently)
    # If your pipeline already uses date strings like "YYYY-MM-DD", this is safe.
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date

    # Clean + score
    out["title_clean"] = out["title"].map(clean_headline)

    def _score(t: str) -> float:
        if not t:
            return 0.0
        return float(_analyzer.polarity_scores(t)["compound"])

    out["sentiment_compound"] = out["title_clean"].map(_score)
    return out


def aggregate_daily_sentiment(df_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Daily aggregation:
    - headlines_count
    - sentiment_mean/median/min/max
    """
    if df_scored is None or df_scored.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "headlines_count",
                "sentiment_mean",
                "sentiment_median",
                "sentiment_min",
                "sentiment_max",
            ]
        )

    # Ensure date is a proper date for grouping
    df = df_scored.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    g = df.groupby("date")["sentiment_compound"]

    out = pd.DataFrame(
        {
            "headlines_count": g.size(),
            "sentiment_mean": g.mean(),
            "sentiment_median": g.median(),
            "sentiment_min": g.min(),
            "sentiment_max": g.max(),
        }
    ).reset_index()

    # deterministic types
    out["headlines_count"] = out["headlines_count"].astype(int)
    for c in ["sentiment_mean", "sentiment_median", "sentiment_min", "sentiment_max"]:
        out[c] = out[c].astype(float)

    return out