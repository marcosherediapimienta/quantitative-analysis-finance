import scipy.stats as si
import numpy as np

# Parámetros dados
S = 205.35      # Precio spot
K = 140         # Strike
T = 33 / 365    # Tiempo a vencimiento en años
r = 0.0432      # Tasa libre de riesgo
C_market = 67.93  # Precio de mercado de la call

# Función de precio de Black-Scholes para una call europea
def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)

# Función para encontrar la volatilidad implícita
def implied_volatility_call(C_market, S, K, T, r, tol=1e-6, max_iterations=100):
    sigma_low = 1e-6
    sigma_high = 5.0
    for i in range(max_iterations):
        sigma_mid = (sigma_low + sigma_high) / 2
        price = bs_call_price(S, K, T, r, sigma_mid)
        if abs(price - C_market) < tol:
            return sigma_mid
        elif price > C_market:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
    return sigma_mid  # Mejor estimación si no converge completamente

# Calcular la volatilidad implícita
iv = implied_volatility_call(C_market, S, K, T, r)
print(f"Volatilidad implícita: {iv:.4%}")
