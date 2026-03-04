from __future__ import annotations

import numpy as np
import pandas as pd


def build_market_features(
    spy: pd.DataFrame,
    vix: pd.DataFrame,
    volatility_window: int,
    volume_z_window: int
) -> pd.DataFrame:
    """
    Produces daily market features indexed by date:
    - close_spy, volume_spy
    - close_vix
    - log_return (SPY)
    - lag_return (SPY)
    - volatility_30d (rolling std of log returns)
    - volume_z_30d (rolling z-score of SPY volume)
    - target_next_return (next day log return)
    """
    df = pd.DataFrame(index=spy.index.copy())
    df.index.name = "date"

    df["close_spy"] = spy["close"]
    df["volume_spy"] = spy["volume"]
    df["close_vix"] = vix.reindex(df.index)["close"]

    df["log_return"] = np.log(df["close_spy"]).diff()
    df["lag_return"] = df["log_return"].shift(1)

    df["volatility_30d"] = df["log_return"].rolling(volatility_window).std()

    vol_mean = df["volume_spy"].rolling(volume_z_window).mean()
    vol_std = df["volume_spy"].rolling(volume_z_window).std()
    df["volume_z_30d"] = (df["volume_spy"] - vol_mean) / vol_std

    df["target_next_return"] = df["log_return"].shift(-1)

    df = df.dropna().copy()
    df.reset_index(inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df