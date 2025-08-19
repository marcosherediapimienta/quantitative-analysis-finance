import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

class MeanVarianceOptimization:
    def __init__(self, tickers, start="2023-01-01", end="2024-01-01"):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.data = self._download_data()
        self.returns = self._calculate_returns()
        self.mean_returns = self._calculate_mean_returns()
        # Si hay más de un activo, calculamos la matriz de covarianza
        if len(self.tickers) > 1:
            self.cov_matrix = self._calculate_cov_matrix()
        else:
            self.cov_matrix = None 
        
        # Mostrar los rendimientos promedio y la matriz de covarianza
        print("Mean Returns:")
        print(self.mean_returns)

        print("\nCovariance Matrix:")
        print(self.cov_matrix)
    
    def _download_data(self):
        """Descargar los datos de precios ajustados"""
        data = yf.download(self.tickers, start=self.start, end=self.end)['Adj Close']
        return data
    
    def _calculate_returns(self):
        """Calcular los rendimientos diarios"""
        returns = self.data.pct_change(fill_method=None).dropna()
        return returns
    
    def _calculate_mean_returns(self):
        """Estimar los rendimientos esperados (promedio de los rendimientos diarios)"""
        return self.returns.mean()
    
    def _calculate_cov_matrix(self):
        """Estimar la matriz de covarianza de los rendimientos"""
        return self.returns.cov()
    
    def objective(self, weights):
        """Función objetivo (minimizar la volatilidad)"""
        # Si hay más de un activo, usamos la covarianza
        if len(self.tickers) > 1:
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        else:
            # Si hay solo un activo, simplemente calculamos su volatilidad
            return np.std(self.returns) * np.sqrt(252)
    
    def optimize_portfolio(self):
        """Optimización de la cartera"""
        if len(self.tickers) == 1:
            return np.array([1])  # Si solo hay un activo, el peso es 1
        else:
            # Optimización de media-varianza para más de un activo
            constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
            bounds = tuple((0, 1) for asset in range(len(self.tickers)))
            initial_weights = [1.0 / len(self.tickers)] * len(self.tickers)

            result = minimize(self.objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            return result.x
    
    def portfolio_performance(self):
        """Rendimiento esperado y volatilidad de la cartera óptima"""
        portfolio_return = np.sum(self.mean_returns * self.optimize_portfolio()) * 252  # Anualizado
        portfolio_volatility = self.objective(self.optimize_portfolio())  # Volatilidad anualizada
        return portfolio_return, portfolio_volatility

    def summary(self):
        """Mostrar resumen de la optimización"""
        portfolio_return, portfolio_volatility = self.portfolio_performance()
        optimal_weights = self.optimize_portfolio()
        
        print(f"Expected Portfolio Return: {portfolio_return:.2%}")
        print(f"Portfolio Volatility: {portfolio_volatility:.2%}")
        
        # Mostrar los pesos de cada activo en la cartera óptima
        print("\nOptimal Portfolio Weights:")
        for ticker, weight in zip(self.tickers, optimal_weights):
            print(f"{ticker}: {weight:.2%}")

# Ejemplo de uso
tickers = ['ACWI','META','AMZN','BTC-USD', '^TNX']
optimizer = MeanVarianceOptimization(tickers)
optimizer.summary()

