import numpy as np
import yfinance as yf

from tools.config import DEFAULT_FALLBACK_VOLATILITY, TRADING_DAYS_PER_YEAR


class HistoricalVolatilityEstimator:
    def __init__(
        self,
        fallback: float = DEFAULT_FALLBACK_VOLATILITY,
        trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        self.fallback = fallback
        self.trading_days_per_year = trading_days_per_year

    def estimate(self, ticker: str, lookback_window: int = 252) -> float:
        try:
            data = yf.Ticker(ticker).history(period=f"{lookback_window + 1}d")["Close"]
            returns = np.log(data / data.shift(1)).dropna()
            volatility = returns.std() * np.sqrt(self.trading_days_per_year)
            return float(volatility)
        except Exception as e:
            print(f"Error calculating historical volatility for {ticker}: {e}")
            return self.fallback
