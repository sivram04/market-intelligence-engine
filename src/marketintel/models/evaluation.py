from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass(frozen=True)
class Metrics:
    r2: float
    mae: float
    directional_accuracy: float


def time_split(df: pd.DataFrame, test_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) <= test_days + 5:
        raise ValueError(f"Not enough rows ({len(df)}) for test_days={test_days}. Need more history.")
    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()
    return train, test


def evaluate_regression(model, test_df: pd.DataFrame, feature_cols: list[str]) -> Metrics:
    X = test_df[feature_cols]
    y_true = test_df["target_next_return"].astype(float).values
    y_pred = model.predict(X).astype(float)

    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))

    # Directional accuracy: sign(pred) == sign(actual)
    # (treat 0 as 0)
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    return Metrics(r2=r2, mae=mae, directional_accuracy=dir_acc)


def regime_directional_accuracy(test_df: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Volatility regime segmentation (median split) computed on TEST SET ONLY (no leakage).
    Returns directional accuracy for low/high/all.
    """
    vol = test_df["volatility_30d"].astype(float).values
    y_true = test_df["target_next_return"].astype(float).values

    med = float(np.median(vol))
    low_mask = vol <= med
    high_mask = vol > med

    def _acc(mask) -> float:
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))

    return {
        "all": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
        "low_vol": _acc(low_mask),
        "high_vol": _acc(high_mask),
    }