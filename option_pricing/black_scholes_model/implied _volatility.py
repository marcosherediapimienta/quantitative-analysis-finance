from scipy.stats import norm
import numpy as np
from scipy.optimize import brentq

def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def vega(S, K, T, r, sigma):
    # Derivada del precio respecto a sigma
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_volatility_newton(market_price, S, K, T, r, tol=1e-6, max_iter=100):
    sigma = 0.2  # Suposición inicial (20%)
    for i in range(max_iter):
        price = black_scholes_call_price(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma)
        if abs(v) < 1e-8:  # Si vega es muy baja, usar Brent
            return implied_volatility_brent(market_price, S, K, T, r)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / v
    return None  # No converge

def implied_volatility_brent(market_price, S, K, T, r, tol=1e-6):
    # Usamos el método de Brent para encontrar la raíz de la función de precio
    def objective(sigma):
        return black_scholes_call_price(S, K, T, r, sigma) - market_price
    
    # Definir un rango de volatilidades para el método de Brent
    return brentq(objective, 1e-6, 5, xtol=tol)

# Parámetros
S = 206.48
K = 140
T = 35/365
r = 0.0420
C_market = 67.93

# Calcular volatilidad implícita
iv = implied_volatility_newton(C_market, S, K, T, r)

# Calcular vega
sigma = iv if iv is not None else 0.2  # Usar iv calculada o valor inicial
vega_value = vega(S, K, T, r, sigma)

# Mostrar resultados
print(f"Implied Volatility: {iv:.2%}" if iv is not None else "Implied Volatility: No converge")
print(f"Vega: {vega_value:.8f}")
