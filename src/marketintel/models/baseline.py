from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression


BASELINE_FEATURES = ["lag_return", "volatility_30d"]


def train_baseline(train_df: pd.DataFrame) -> LinearRegression:
    X = train_df[BASELINE_FEATURES]
    y = train_df["target_next_return"]
    model = LinearRegression()
    model.fit(X, y)
    return model