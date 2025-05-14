import numpy as np
from scipy.stats import norm
import yfinance as yf

def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def binomial_call_price(S, K, T, r, sigma, N=100):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    ST = S * u**np.arange(N, -1, -1) * d**np.arange(0, N+1, 1)
    payoff = np.maximum(ST - K, 0)
    for i in range(N, 0, -1):
        payoff = np.exp(-r * dt) * (p * payoff[:-1] + (1 - p) * payoff[1:])
    return payoff[0]

if __name__ == "__main__":
    # Parámetros reales desde Yahoo Finance
    ticker = "AAPL"  # Puedes cambiar por cualquier ticker válido
    S = yf.Ticker(ticker).history(period="1d")['Close'][-1]
    K = round(S)  # Strike cercano al spot
    T = 30/365    # 30 días a vencimiento
    r = 0.05      # Tasa libre de riesgo (puedes ajustar)
    sigma = 0.2   # Volatilidad inicial (puedes ajustar)
    N = 100       # Pasos binomiales

    print(f"Spot (S) obtenido de Yahoo Finance para {ticker}: {S:.2f}")
    print(f"Strike (K) usado: {K}")

    bs_price = black_scholes_call_price(S, K, T, r, sigma)
    binom_price = binomial_call_price(S, K, T, r, sigma, N)

    print(f"Precio Black-Scholes: {bs_price:.4f}")
    print(f"Precio Binomial (N={N}): {binom_price:.4f}")
