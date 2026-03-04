from __future__ import annotations

import pandas as pd


def merge_feature_store(market_features: pd.DataFrame, daily_sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join sentiment onto market features by date.
    Missing sentiment is filled with zeros (no articles that day).
    """
    df = market_features.merge(daily_sentiment, on="date", how="left")

    # fill sentiment defaults
    df["headlines_count"] = df["headlines_count"].fillna(0).astype(int)
    for c in ["sentiment_mean", "sentiment_median", "sentiment_min", "sentiment_max"]:
        df[c] = df[c].fillna(0.0).astype(float)

    # lag sentiment by 1 day (useful to model delayed digestion)
    df["sentiment_mean_lag1"] = df["sentiment_mean"].shift(1).fillna(0.0).astype(float)
    df["headlines_count_lag1"] = df["headlines_count"].shift(1).fillna(0).astype(int)
    df["sentiment_mean_lag2"] = df["sentiment_mean"].shift(2).fillna(0)
    df["sentiment_mean_lag3"] = df["sentiment_mean"].shift(3).fillna(0)
    df["headlines_count_lag2"] = df["headlines_count"].shift(2).fillna(0)
    df["headlines_count_lag3"] = df["headlines_count"].shift(3).fillna(0)
    df = df.sort_values("date").reset_index(drop=True)
    return df