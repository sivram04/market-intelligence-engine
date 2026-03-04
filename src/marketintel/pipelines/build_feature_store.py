from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from pathlib import Path
from ..config import Settings
from ..io_paths import Paths
from ..data_sources.market_yf import fetch_spy_vix
from ..data_sources.gdelt import fetch_gdelt_headlines_daily
from ..features.market_features import build_market_features
from ..features.sentiment_features import score_headlines, aggregate_daily_sentiment
from ..features.feature_store import merge_feature_store

def _print_feature_store_summary(fs: pd.DataFrame) -> None:
    print("")
    print("=== FEATURE STORE SANITY SUMMARY ===")

    # % days with headlines
    pct_with_headlines = (fs["headlines_count"] > 0).mean() * 100.0
    print(f"% days with headlines_count > 0: {pct_with_headlines:.2f}%")

    # sentiment distribution (only where headlines exist, otherwise zeros dominate)
    nonzero = fs.loc[fs["headlines_count"] > 0, "sentiment_mean"]
    if len(nonzero) > 0:
        print(
            "sentiment_mean (days with headlines) "
            f"min/median/max: {nonzero.min():.4f} / {nonzero.median():.4f} / {nonzero.max():.4f}"
        )
    else:
        print("sentiment_mean: no days with headlines > 0 in this window.")

    # correlations (insight only)
    corr_sm = fs["sentiment_mean"].corr(fs["target_next_return"])
    corr_hc = fs["headlines_count"].corr(fs["target_next_return"])
    print(f"corr(sentiment_mean, next_return): {corr_sm:.4f}")
    print(f"corr(headlines_count, next_return): {corr_hc:.4f}")
    print("===================================")
    print("")
def main() -> None:
    s = Settings()
    p = Paths.from_project_root()
    p.ensure_dirs()

    # Dates: ensure >= 90 days of headlines
    end = date.today()
    start_market = s.market_start_date()
    start_gdelt = end - timedelta(days=s.gdelt_lookback_days)

    # 1) Market data
    spy, vix = fetch_spy_vix(s.ticker_spy, s.ticker_vix, start_market, end + timedelta(days=1))
    mf = build_market_features(spy, vix, volatility_window=s.volatility_window, volume_z_window=s.volume_z_window)
    mf.to_csv(p.market_features_csv, index=False)

      # 2) GDELT headlines (cached)
    if p.gdelt_headlines_csv.exists() and p.gdelt_headlines_csv.stat().st_size > 0:
        print("[2/4] Loading cached GDELT headlines ...")
        headlines = pd.read_csv(p.gdelt_headlines_csv)
    else:
        print("[2/4] Fetching GDELT headlines ... (first run can take a few minutes)")
        headlines = fetch_gdelt_headlines_daily(
            query=s.gdelt_query,
            start=start_gdelt,
            end=end,
            max_records_per_day=s.gdelt_max_records_per_day,
            timeout_s=s.request_timeout_s,
            max_retries=s.request_max_retries,
            backoff_s=s.request_backoff_s,
        )
        headlines.to_csv(p.gdelt_headlines_csv, index=False)

    print(f"[2/4] Headlines rows: {len(headlines):,}")
    # 3) Sentiment scoring + daily aggregates
    scored = score_headlines(headlines)
    daily = aggregate_daily_sentiment(scored)
    daily.to_csv(p.daily_sentiment_csv, index=False)

    # 4) Feature store merge
    fs = merge_feature_store(mf, daily)
    fs.to_csv(p.feature_store_csv, index=False)

    # Required console outputs
    if not fs.empty:
        dmin = fs["date"].min()
        dmax = fs["date"].max()
        print("=== FEATURE STORE BUILD COMPLETE ===")
        print(f"Rows in feature_store: {len(fs):,}")
        print(f"Date range coverage: {dmin} -> {dmax}")
        print(f"Wrote: {p.market_features_csv}")
        print(f"Wrote: {p.gdelt_headlines_csv}")
        print(f"Wrote: {p.daily_sentiment_csv}")
        print(f"Wrote: {p.feature_store_csv}")
    else:
        print("Feature store is empty. Something failed upstream (market or GDELT ingestion).")


if __name__ == "__main__":
    main()