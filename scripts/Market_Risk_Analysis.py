import yfinance as yf
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

class RiskAnalysis:
    def __init__(self, tickers, start="2023-01-01", end="2024-02-01", benchmark="^GSPC"):
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.start = start
        self.end = end
        self.benchmark = benchmark
        self.data = self._download_data()

    def _download_data(self):
        """Downloads adjusted close price data for assets and benchmark."""
        prices = yf.download(self.tickers + [self.benchmark], start=self.start, end=self.end)["Adj Close"]
        return prices

    def compute_volatility(self):
        """Computes annualized volatility of the assets."""
        returns = self.data.pct_change().dropna()
        volatilities = returns.std() * np.sqrt(252)  # Annualized volatility
        return volatilities

    def compute_beta(self):
        """Calculates Beta of the assets relative to the benchmark."""
        returns = self.data.pct_change().dropna()
        market_returns = returns[self.benchmark]
        betas = {}

        for ticker in self.tickers:
            cov_matrix = np.cov(returns[ticker], market_returns)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            betas[ticker] = beta
        
        return betas

    def compute_var(self, confidence_level=0.95):
        """Computes historical Value at Risk (VaR) at a given confidence level."""
        returns = self.data.pct_change().dropna()
        var_values = {}

        for ticker in self.tickers:
            var = np.percentile(returns[ticker], 100 * (1 - confidence_level))
            var_values[ticker] = var
        
        return var_values

    def compute_cvar(self, confidence_level=0.95):
        """Computes Conditional Value at Risk (CVaR) at a given confidence level."""
        returns = self.data.pct_change().dropna()
        cvar_values = {}

        for ticker in self.tickers:
            var = np.percentile(returns[ticker], 100 * (1 - confidence_level))
            cvar = returns[ticker][returns[ticker] <= var].mean()
            cvar_values[ticker] = cvar
        
        return cvar_values

    def plot_var_cvar(self):
        """Plots Value at Risk (VaR) and Conditional VaR (CVaR) as histograms."""
        returns = self.data.pct_change().dropna()
        
        for ticker in self.tickers:
            var = self.compute_var()[ticker]
            cvar = self.compute_cvar()[ticker]

            plt.figure(figsize=(8, 5))
            plt.hist(returns[ticker], bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
            plt.axvline(var, color='blue', linestyle='dashed', linewidth=2, label=f"VaR 95%: {var:.4f}")
            plt.axvline(cvar, color='red', linestyle='dashed', linewidth=2, label=f"CVaR 95%: {cvar:.4f}")
            plt.title(f"Risk Distribution: {ticker}")
            plt.xlabel("Daily Returns")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

    def summary(self):
        """Displays a summary of computed risk metrics and generates plots."""
        print(f"\nRisk Analysis for {', '.join(self.tickers)}")
        print("-" * 50)
        print("Annualized Volatility:")
        print(self.compute_volatility())

        print("\nBeta relative to the market:")
        print(self.compute_beta())

        print("\nValue at Risk (VaR) 95%:")
        print(self.compute_var())

        print("\nConditional VaR (CVaR) 95%:")
        print(self.compute_cvar())
        self.plot_var_cvar()