"""
Monte Carlo simulation for European Call Option Pricing using Black-Scholes model
"""
import numpy as np
import yfinance as yf
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import brentq

def monte_carlo_european_call(S0, K, T, r, sigma, n_simulations=10000):
    """
    Monte Carlo pricing for a European call option under Black-Scholes assumptions.
    """
    np.random.seed(42)
    Z = np.random.standard_normal(n_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    return price

def get_option_data_yahoo(ticker, expiry, strike, option_type='call', r=0.0421):
    """
    Fetches option data from Yahoo Finance using yfinance.
    Returns dict: S0, K, T, r, market_price
    """
    stock = yf.Ticker(ticker)
    S0 = stock.history(period='1d')['Close'].iloc[-1]
    options = stock.option_chain(expiry)
    df = options.calls if option_type == 'call' else options.puts
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

def implied_volatility_newton(market_price, S, K, T, r, tol=1e-6, max_iter=25):
    """
    Calculate implied volatility using the Newton-Raphson method for calls.
    Returns float: Implied volatility or None if it doesn't converge
    """
    
    def black_scholes_call_price(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    def vega(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)
    sigma = 0.2
    for _ in range(max_iter):
        price = black_scholes_call_price(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma)
        if abs(v) < 1e-8:
            def objective(sigma_):
                return black_scholes_call_price(S, K, T, r, sigma_) - market_price
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
        return black_scholes_call_price(S, K, T, r, sigma_) - market_price
    try:
        return brentq(objective, 0.0001, 5.0, xtol=tol)
    except Exception:
        return None

if __name__ == "__main__":
    print("\nEuropean Call Option Pricing (Monte Carlo + Implied Volatility)")
    print("="*60)
    ticker = input("Enter the ticker (e.g., ^SPX): ").strip()
    stock = yf.Ticker(ticker)
    expirations = stock.options
    print("\nAvailable expiration dates:")
    for i, exp in enumerate(expirations):
        print(f"  [{i}] {exp}")
    idx = int(input("\nSelect the number of the expiration date you want to use: "))
    expiry = expirations[idx]
    options = stock.option_chain(expiry)
    strikes = options.calls['strike'].values
    print("\nAvailable strikes for this expiration:")
    print(strikes)
    strike = float(input("\nEnter the strike you want to use (from the list above): "))
    r = input("\nEnter the risk-free rate (e.g., 0.0421 for 4.21%) [default 0.0421]: ").strip()
    r = float(r) if r else 0.0421
    data = get_option_data_yahoo(ticker, expiry, strike, 'call', r)
    print("\n" + "="*50)
    print(f"  OPTION DATA ({ticker} CALL)")
    print("="*50)
    print(f"  Current underlying (S0):     {data['S0']:.2f}")
    print(f"  Strike (K):                  {data['K']:.2f}")
    print(f"  Time to maturity (T):         {data['T']*365:.0f} days ({data['T']:.4f} years)")
    print(f"  Risk-free rate (r):           {data['r']*100:.2f}%")
    print(f"  Option market price:          {data['market_price']:.4f}")
    print("\nCalculating implied volatility...")
    iv = implied_volatility_newton(
        data['market_price'], data['S0'], data['K'], data['T'], data['r']
    )
    if iv is not None:
        print(f"  Implied volatility (IV):       {iv*100:.2f}%")
    else:
        print("  Implied volatility (IV):       Did not converge")
    print("\nCalculating price with Monte Carlo...")
    n_sim = 10000
    price = monte_carlo_european_call(
        data['S0'], data['K'], data['T'], data['r'], iv, n_sim
    )
    print(f"  European call option price (Monte Carlo, IV): {price:.4f}")
    print("="*50 + "\n")
