import numpy as np
import cvxpy as cp
import yfinance as yf

class MarkowitzPortfolio:
    def __init__(self, tickers, start="2020-01-01", end="2024-01-01"):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.returns = None
        self.expected_returns = None
        self.cov_matrix = None
        self.weights = None

    def download_data(self):
        """Descarga los datos históricos y calcula los retornos."""
        data = yf.download(self.tickers, start=self.start, end=self.end)["Adj Close"]
        self.returns = data.pct_change().dropna()
        self.expected_returns = self.returns.mean().values
        self.cov_matrix = self.returns.cov().values

        # Asegurar que la matriz de covarianza sea semidefinida positiva
        eigvals = np.linalg.eigvals(self.cov_matrix)
        if np.any(eigvals < 0):
            print("Warning: Covariance matrix adjusted to be PSD.")
            self.cov_matrix = (self.cov_matrix + self.cov_matrix.T) / 2
            self.cov_matrix += np.eye(self.cov_matrix.shape[0]) * 1e-6

    def optimize_portfolio(self, max_risk=0.02, max_weight=0.5):
        """Optimiza la cartera maximizando el retorno esperado con restricciones."""
        n = len(self.tickers)
        w = cp.Variable(n)  # Pesos de los activos

        # Restricciones
        constraints = [
            cp.sum(w) == 1,  # Suma de pesos = 1
            w >= 0,  # Sin ventas en corto
            w <= max_weight  # Máximo peso por activo
        ]

        # Función de riesgo (Varianza del portafolio)
        risk = cp.quad_form(w, self.cov_matrix)
        constraints.append(risk <= max_risk)

        # Maximizar retorno esperado
        objective = cp.Maximize(self.expected_returns @ w)
        problem = cp.Problem(objective, constraints)
        problem.solve()

        self.weights = w.value
        return self.weights

    def display_results(self):
        """Muestra la asignación óptima del portafolio."""
        print("\n✅ Optimal Portfolio Allocation:")
        for asset, weight in zip(self.tickers, self.weights):
            print(f"{asset}: {weight:.4f}")

        expected_return = np.dot(self.expected_returns, self.weights)
        risk_value = np.sqrt(self.weights @ self.cov_matrix @ self.weights)

        print(f"\n📊 Expected Return: {expected_return:.4%}")
        print(f"⚠️ Portfolio Risk: {risk_value:.4%}")


# 📌 Ejemplo de uso
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
portfolio = MarkowitzPortfolio(tickers)
portfolio.download_data()
portfolio.optimize_portfolio()
portfolio.display_results()
