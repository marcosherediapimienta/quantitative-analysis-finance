"""
Monte Carlo simulation for European Put Option Pricing using Black-Scholes model
"""
import numpy as np
import yfinance as yf
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import brentq

def monte_carlo_european_put(S0, K, T, r, sigma, n_simulations=10000):
    """
    Monte Carlo pricing for a European put option under Black-Scholes assumptions.
    """
    np.random.seed(42)
    Z = np.random.standard_normal(n_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    return price

def get_option_data_yahoo(ticker, expiry, strike, option_type='put', r=0.0421):
    """
    Fetches option data from Yahoo Finance using yfinance.
    Returns dict: S0, K, T, r, market_price
    """
    stock = yf.Ticker(ticker)
    S0 = stock.history(period='1d')['Close'].iloc[-1]
    options = stock.option_chain(expiry)
    df = options.puts if option_type == 'put' else options.calls
    row = df[df['strike'] == strike]
    if row.empty:
        raise ValueError('Strike not found in options chain')
    market_price = float(row['lastPrice'].iloc[0])
    today = datetime.now().date()
    expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
    T = (expiry_date - today).days / 365
    return {
        'S0': float(S0),
        'K': strike,
        'T': T,
        'r': r,
        'market_price': market_price
    }

def implied_volatility_newton_put(market_price, S, K, T, r, tol=1e-6, max_iter=25):
    """
    Calculate implied volatility for a European put using the Newton-Raphson method.
    Returns float: Implied volatility or None if it doesn't converge
    """
    def black_scholes_put_price(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    def vega(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)
    sigma = 0.2
    for _ in range(max_iter):
        price = black_scholes_put_price(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma)
        if abs(v) < 1e-8:
            def objective(sigma_):
                return black_scholes_put_price(S, K, T, r, sigma_) - market_price
            try:
                return brentq(objective, 0.0001, 5.0, xtol=tol)
            except Exception:
                return None
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma = sigma - diff / v
        if sigma <= 0:
            sigma = 0.0001
    def objective(sigma_):
        return black_scholes_put_price(S, K, T, r, sigma_) - market_price
    try:
        return brentq(objective, 0.0001, 5.0, xtol=tol)
    except Exception:
        return None

if __name__ == "__main__":
    # Configuración de ejemplo
    ticker = '^SPX'
    expiry = '2025-06-13'
    option_type = 'put'
    # Mostrar strikes disponibles antes de pedir el strike
    stock = yf.Ticker(ticker)
    options = stock.option_chain(expiry)
    strikes = options.puts['strike'].values
    print("\nStrikes disponibles para este vencimiento:")
    print(strikes)
    # Permitir al usuario elegir el strike
    strike = float(input("\nIntroduce el strike que deseas usar (de la lista mostrada): "))
    data = get_option_data_yahoo(ticker, expiry, strike, option_type)
    print("\n" + "="*50)
    print(f"  DATOS DE LA OPCIÓN ({ticker} {option_type.upper()})")
    print("="*50)
    print(f"  Subyacente actual (S0):     {data['S0']:.2f}")
    print(f"  Strike (K):                  {data['K']:.2f}")
    print(f"  Tiempo a vencimiento (T):    {data['T']*365:.0f} días ({data['T']:.4f} años)")
    print(f"  Tasa libre de riesgo (r):    {data['r']*100:.2f}%")
    print(f"  Precio de mercado opción:    {data['market_price']:.4f}")
    print("\nCalculando volatilidad implícita...")
    iv = implied_volatility_newton_put(
        data['market_price'], data['S0'], data['K'], data['T'], data['r']
    )
    if iv is not None:
        print(f"  Volatilidad implícita (IV):   {iv*100:.2f}%")
    else:
        print("  Volatilidad implícita (IV):   No convergió")
    print("\nCalculando precio con Monte Carlo...")
    n_sim = 10000
    price = monte_carlo_european_put(
        data['S0'], data['K'], data['T'], data['r'], iv, n_sim
    )
    print(f"  Precio opción put europea (Monte Carlo, IV): {price:.4f}")
    print("="*50 + "\n")
