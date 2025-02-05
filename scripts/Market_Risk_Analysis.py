import yfinance as yf
import numpy as np
import pandas as pd

class RiskAnalysis:
    def __init__(self, tickers, weights, start="2023-01-01", end="2024-01-01", benchmark="^GSPC", risk_free_rate=0.03):
        if end is None:
            end = pd.to_datetime("today").strftime("%Y-%m-%d")
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.weights = np.array(weights)
        if not np.isclose(self.weights.sum(), 1):
            self.weights = self.weights / self.weights.sum()
        self.start = start
        self.end = end
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate / 252
        self.data = self._download_data()

    def _download_data(self):
        """Descarga precios ajustados de cierre."""
        tickers = self.tickers + [self.benchmark]
        try:
            prices = yf.download(tickers, start=self.start, end=self.end)["Adj Close"]
            return prices
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None
    
    def compute_correlation_matrix(self):
        """Calcula la matriz de correlación de los activos en la cartera."""
        returns = self.data[self.tickers].pct_change().dropna()
        return returns.corr()
    
    def compute_volatility(self):
        """Calcula la volatilidad anualizada de la cartera."""
        returns = self.data[self.tickers].pct_change().dropna()
        return np.sqrt(np.dot(self.weights.T, np.dot(returns.cov(), self.weights))) * np.sqrt(252)

    def compute_var(self, confidence_level=0.95, horizon=1):
        """Calcula el Value at Risk (VaR) para un nivel de confianza y horizonte dados."""
        returns = self.data[self.tickers].pct_change().dropna()
        portfolio_returns = returns.dot(self.weights)
        return np.percentile(portfolio_returns, 100 * (1 - confidence_level)) * np.sqrt(horizon)

    def compute_cvar(self, confidence_level=0.95, horizon=1):
        """Calcula el Conditional Value at Risk (CVaR) para un nivel de confianza y horizonte dados."""
        returns = self.data[self.tickers].pct_change().dropna()
        portfolio_returns = returns.dot(self.weights)
        var = self.compute_var(confidence_level, horizon)
        return portfolio_returns[portfolio_returns <= var].mean()

    def compute_beta(self):
        """Calcula la beta de la cartera respecto al benchmark."""
        returns = self.data.pct_change().dropna()
        portfolio_returns = returns[self.tickers].dot(self.weights)
        market_returns = returns[self.benchmark]
        cov_matrix = np.cov(portfolio_returns, market_returns)
        return cov_matrix[0, 1] / cov_matrix[1, 1]

    def compute_sharpe_ratio(self):
        """Calcula el ratio de Sharpe anualizado de la cartera."""
        returns = self.data[self.tickers].pct_change().dropna()
        portfolio_returns = returns.dot(self.weights)
        excess_return = portfolio_returns.mean() - self.risk_free_rate
        return (excess_return * 252) / self.compute_volatility()

    def compute_sortino_ratio(self):
        """Calcula el ratio de Sortino anualizado de la cartera."""
        returns = self.data[self.tickers].pct_change().dropna()
        portfolio_returns = returns.dot(self.weights)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        excess_return = portfolio_returns.mean() - self.risk_free_rate
        return (excess_return * 252) / downside_vol

    def compute_max_drawdown(self):
        """Calcula el máximo drawdown de la cartera."""
        cumulative_returns = (1 + self.data[self.tickers].pct_change()).cumprod()
        portfolio_value = cumulative_returns.dot(self.weights)
        return (portfolio_value / portfolio_value.cummax() - 1).min()

    def compute_treynor_ratio(self):
        """Calcula el ratio de Treynor de la cartera."""
        annual_return = self.data[self.tickers].pct_change().dot(self.weights).mean() * 252
        return (annual_return - self.risk_free_rate * 252) / self.compute_beta()

    def summary(self):
        """Muestra un resumen de las métricas de riesgo de la cartera."""
        print(f"\nRisk Analysis for Portfolio: {', '.join(self.tickers)}")
        print("-" * 50)
        print(f"Annualized Volatility: {self.compute_volatility():.4f}")
        print(f"Beta relative to {self.benchmark}: {self.compute_beta():.4f}")
        print(f"Value at Risk (VaR 95%): {self.compute_var():.4%}")
        print(f"Conditional VaR (CVaR 95%): {self.compute_cvar():.4%}")
        print(f"Sharpe Ratio: {self.compute_sharpe_ratio():.4f}")
        print(f"Sortino Ratio: {self.compute_sortino_ratio():.4f}")
        print(f"Max Drawdown: {self.compute_max_drawdown():.4%}")
        print(f"Treynor Ratio: {self.compute_treynor_ratio():.4f}")
        print(f"Correlation Matrix:{self.compute_correlation_matrix()}")