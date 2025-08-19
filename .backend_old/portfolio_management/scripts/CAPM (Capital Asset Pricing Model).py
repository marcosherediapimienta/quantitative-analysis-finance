import numpy as np
import pandas as pd
import yfinance as yf

class CAPM:
    def __init__(self, assets, market="^GSPC", risk_free_rate=0.04, start="2024-01-01", end="2025-01-01"):
        """
        Initializes the CAPM class with assets and the market index.
        
        :param assets: List of asset tickers to analyze.
        :param market: Market index as a benchmark (default is S&P 500).
        :param risk_free_rate: Risk-free rate (default is x% annual).
        :param start: Start date for data retrieval.
        :param end: End date for data retrieval.
        """
        self.assets = assets
        self.market = market
        self.risk_free_rate = risk_free_rate
        self.start = start
        self.end = end
        self.data = None
        self.returns = None
        self.betas = {}
        self.expected_returns = {}

    def download_data(self):
        """Downloads adjusted closing price data from Yahoo Finance."""
        tickers = self.assets + [self.market]
        self.data = yf.download(tickers, start=self.start, end=self.end)["Adj Close"]

    def compute_returns(self):
        """Computes daily returns for the assets and the market."""
        self.returns = self.data.pct_change().dropna()

    def compute_capm(self):
        """Calculates betas and expected returns using the CAPM formula."""
        market_return = self.returns[self.market].mean() * 252  # Annualized market return
        
        for asset in self.assets:
            beta = self.returns.cov().loc[asset, self.market] / self.returns.cov().loc[self.market, self.market]
            expected_return = self.risk_free_rate + beta * (market_return - self.risk_free_rate)
            self.betas[asset] = beta
            self.expected_returns[asset] = expected_return

    def get_results(self):
        """Returns a DataFrame with CAPM results."""
        return pd.DataFrame({"Beta": self.betas, "Expected Return": self.expected_returns})

    def run(self):
        """Executes the full CAPM process."""
        self.download_data()
        self.compute_returns()
        self.compute_capm()
        return self.get_results()

assets = ["AAPL", "MSFT", "GOOGL", "AMZN"]
capm_model = CAPM(assets)
df_capm = capm_model.run()
print(df_capm)


