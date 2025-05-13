import numpy as np

# Parámetros
S0 = 123        # Precio inicial
K = 100         # Strike
T = 38/365         # Tiempo hasta vencimiento (1 año)
r = 0.0421        # Tasa libre de riesgo
sigma = 0.59    # Volatilidad
N = 100000      # Número de simulaciones
option_type = 'call'  # 'put' para opción put

# Simulación de precios al vencimiento
Z = np.random.standard_normal(N)
ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

# Payoff
if option_type == 'call':
    payoff = np.maximum(ST - K, 0)
else:
    payoff = np.maximum(K - ST, 0)

# Valor presente esperado
option_price = np.exp(-r * T) * np.mean(payoff)

print(f"Precio estimado de la opción {option_type}: {option_price:.4f}")
