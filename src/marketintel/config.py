from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Settings:
    # Market tickers
    ticker_spy: str = "SPY"
    ticker_vix: str = "^VIX"

    # Feature engineering
    volatility_window: int = 30
    volume_z_window: int = 30

    # GDELT ingestion
    gdelt_lookback_days: int = 180  # fetch >= 90; default higher for reliability
    gdelt_max_records_per_day: int = 250
    gdelt_query: str = (
    '("S&P 500" OR SPY OR equities OR "stock market" OR "US stocks" OR earnings '
    'OR inflation OR "Federal Reserve" OR Powell OR CPI OR jobs OR recession OR tariff)'
)
    gdelt_mode: str = "ArtList"

    # Experiment split
    test_days: int = 30

    # Requests / retry behavior
    request_timeout_s: int = 30
    request_max_retries: int = 5
    request_backoff_s: float = 1.2

    def market_start_date(self) -> date:
        # enough history for rolling windows + train/test
        return date.today() - timedelta(days=max(220, self.gdelt_lookback_days + 120))
