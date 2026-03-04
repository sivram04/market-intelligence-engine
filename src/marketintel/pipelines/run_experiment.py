from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from ..io_paths import Paths
from ..models.baseline import train_baseline, BASELINE_FEATURES
from ..models.enhanced import train_enhanced, ENHANCED_FEATURES
from ..models.evaluation import time_split, evaluate_regression, regime_directional_accuracy


def main() -> None:
    s = Settings()
    p = Paths.from_project_root()
    p.ensure_dirs()

    if not p.feature_store_csv.exists():
        raise FileNotFoundError(f"Missing feature store: {p.feature_store_csv}. Run Stage 1 first.")

    df = pd.read_csv(p.feature_store_csv)
    if df.empty:
        raise ValueError("feature_store.csv is empty.")

    # enforce types
    df["date"] = pd.to_datetime(df["date"]).dt.date

    train_df, test_df = time_split(df, test_days=s.test_days)

    # --- Baseline ---
    baseline_model = train_baseline(train_df)
    baseline_metrics = evaluate_regression(baseline_model, test_df, BASELINE_FEATURES)
    baseline_pred = baseline_model.predict(test_df[BASELINE_FEATURES]).astype(float)
    baseline_regimes = regime_directional_accuracy(test_df, baseline_pred)

    # --- Enhanced ---
    enhanced_model = train_enhanced(train_df)
    enhanced_metrics = evaluate_regression(enhanced_model, test_df, ENHANCED_FEATURES)
    enhanced_pred = enhanced_model.predict(test_df[ENHANCED_FEATURES]).astype(float)
    enhanced_regimes = regime_directional_accuracy(test_df, enhanced_pred)

    # Required structured outputs (OUT-OF-SAMPLE ONLY)
    print("=== OUT-OF-SAMPLE EXPERIMENT RESULTS ===")
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,} (last {s.test_days} days)")
    print(f"Test date range: {test_df['date'].min()} -> {test_df['date'].max()}")
    print("")

    print("BASELINE MODEL (LinearRegression)")
    print(f"Features: {BASELINE_FEATURES}")
    print(f"R²:  {baseline_metrics.r2:.6f}")
    print(f"MAE: {baseline_metrics.mae:.6f}")
    print(f"Directional accuracy: {baseline_metrics.directional_accuracy:.4f}")
    print("Volatility regime directional accuracy (median split on TEST):")
    print(f"  Low vol:  {baseline_regimes['low_vol']:.4f}")
    print(f"  High vol: {baseline_regimes['high_vol']:.4f}")
    print(f"  All:      {baseline_regimes['all']:.4f}")
    print("")

    print("ENHANCED MODEL (LinearRegression + Sentiment)")
    print(f"Features: {ENHANCED_FEATURES}")
    print(f"R²:  {enhanced_metrics.r2:.6f}")
    print(f"MAE: {enhanced_metrics.mae:.6f}")
    print(f"Directional accuracy: {enhanced_metrics.directional_accuracy:.4f}")
    print("Volatility regime directional accuracy (median split on TEST):")
    print(f"  Low vol:  {enhanced_regimes['low_vol']:.4f}")
    print(f"  High vol: {enhanced_regimes['high_vol']:.4f}")
    print(f"  All:      {enhanced_regimes['all']:.4f}")


if __name__ == "__main__":
    main()