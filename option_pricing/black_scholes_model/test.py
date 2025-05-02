import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type debe ser 'call' o 'put'")

# Añadir función vega para calcular la sensibilidad Vega de la opción
def vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

# Añadir función para calcular la volatilidad implícita usando Newton-Raphson
def implied_volatility(price, S, K, T, r, option_type='call', initial_sigma=0.2, tol=1e-6, max_iter=100):
    sigma = initial_sigma
    for i in range(max_iter):
        price_est = black_scholes(S, K, T, r, sigma, option_type)
        v = vega(S, K, T, r, sigma)
        diff = price_est - price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / v
    return sigma

def main():
    # Parámetros
    precio_actual = 100
    K = 105
    T = 1
    r = 0.05
    sigma = 0.2

    # Calcular precios de opciones
    precio_call = black_scholes(precio_actual, K, T, r, sigma, 'call')
    precio_put = black_scholes(precio_actual, K, T, r, sigma, 'put')

    print(f"Call (K={K}, T={T:.2f} años): {precio_call:.2f}")
    print(f"Put  (K={K}, T={T:.2f} años): {precio_put:.2f}")

    # Obtener precio de mercado de la opción para calcular volatilidad implícita
    chain = activo.option_chain(expiration)
    calls = chain.calls
    puts = chain.puts

    market_call = calls[calls['strike'] == K]
    if not market_call.empty:
        mid_call = (market_call['bid'].iloc[0] + market_call['ask'].iloc[0]) / 2
        iv_call = implied_volatility(mid_call, precio_actual, K, T, r, 'call')
        print(f"Precio de mercado call: {mid_call:.2f}")
        print(f"Volatilidad implícita call: {iv_call:.2%}")
    else:
        print(f"No se encontró opción call con strike {K}")

    market_put = puts[puts['strike'] == K]
    if not market_put.empty:
        mid_put = (market_put['bid'].iloc[0] + market_put['ask'].iloc[0]) / 2
        iv_put = implied_volatility(mid_put, precio_actual, K, T, r, 'put')
        print(f"Precio de mercado put: {mid_put:.2f}")
        print(f"Volatilidad implícita put: {iv_put:.2%}")
    else:
        print(f"No se encontró opción put con strike {K}")

if __name__ == "__main__":
    main()