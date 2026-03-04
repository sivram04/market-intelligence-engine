from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression


ENHANCED_FEATURES = [
    "lag_return",
    "volatility_30d",
    "sentiment_mean_lag1",
    "sentiment_mean_lag2",
    "sentiment_mean_lag3",
    "headlines_count_lag1"
]


def train_enhanced(train_df: pd.DataFrame) -> LinearRegression:
    X = train_df[ENHANCED_FEATURES]
    y = train_df["target_next_return"]
    model = LinearRegression()
    model.fit(X, y)
    return model