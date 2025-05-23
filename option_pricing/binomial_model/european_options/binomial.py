import numpy as np
import yfinance as yf
from datetime import datetime
import scipy.stats as stats
from scipy.optimize import brentq

def binomial_european_option_price(S, K, T, r, sigma, N=100, option_type='call'):
    """
    Price a European option using the Cox-Ross-Rubinstein binomial tree (vectorized, numerically stable).
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        N: Number of time steps
        option_type: 'call' or 'put'
    Returns:
        Option price (float)
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Vectorized calculation of terminal stock prices
    j = np.arange(N + 1)
    ST = S * (u ** j) * (d ** (N - j))
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    # Backward induction (vectorized)
    for _ in range(N):
        payoff = discount * (p * payoff[1:] + (1 - p) * payoff[:-1])
    return payoff[0]

def binomial_greeks_european_option(S, K, T, r, sigma, N=100, option_type='call', h=1e-2):
    """
    Estimate Delta, Gamma, Vega, Theta, Rho for a European option using the binomial model.
    Uses finite differences (bumping) on the binomial price.
    """
    # Delta
    price_up = binomial_european_option_price(S + h, K, T, r, sigma, N, option_type)
    price_down = binomial_european_option_price(S - h, K, T, r, sigma, N, option_type)
    price = binomial_european_option_price(S, K, T, r, sigma, N, option_type)
    delta = (price_up - price_down) / (2 * h)
    gamma = (price_up - 2 * price + price_down) / (h ** 2)
    # Vega
    price_vega_up = binomial_european_option_price(S, K, T, r, sigma + h, N, option_type)
    price_vega_down = binomial_european_option_price(S, K, T, r, sigma - h, N, option_type)
    vega = (price_vega_up - price_vega_down) / (2 * h)
    # Theta (backward difference)
    price_theta = binomial_european_option_price(S, K, T - h, r, sigma, N, option_type)
    theta = (price_theta - price) / h
    # Rho
    price_rho_up = binomial_european_option_price(S, K, T, r + h, sigma, N, option_type)
    price_rho_down = binomial_european_option_price(S, K, T, r - h, sigma, N, option_type)
    rho = (price_rho_up - price_rho_down) / (2 * h)
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }

def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

def black_scholes_put_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * stats.norm.pdf(d1) * np.sqrt(T)

def implied_volatility_option(market_price, S, K, T, r, option_type='call', tol=1e-6, max_iter=100):
    sigma = 0.2
    for _ in range(max_iter):
        if option_type == 'call':
            price = black_scholes_call_price(S, K, T, r, sigma)
        else:
            price = black_scholes_put_price(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma)
        diff = price - market_price
        if abs(v) < 1e-8:
            break
        if abs(diff) < tol:
            return sigma
        sigma = sigma - diff / v
        if sigma <= 0:
            sigma = 1e-6
    # If Newton-Raphson fails, use Brent's method
    def objective(sigma_):
        if option_type == 'call':
            return black_scholes_call_price(S, K, T, r, sigma_) - market_price
        else:
            return black_scholes_put_price(S, K, T, r, sigma_) - market_price
    try:
        return brentq(objective, 1e-6, 5.0, xtol=tol, maxiter=max_iter)
    except Exception:
        return None

def get_input(prompt, default, cast_func, validator=None):
    while True:
        try:
            user_input = input(f"{prompt} [default: {default}]: ").strip()
            if user_input == '':
                value = default
            else:
                value = cast_func(user_input)
            if validator and not validator(value):
                print("Invalid value. Please try again.")
                continue
            return value
        except Exception:
            print("Invalid input. Please try again.")

if __name__ == "__main__":
    print("\nBinomial Model European Option Pricing (Cox-Ross-Rubinstein)")
    # 1. Option type first
    option_type = input("Option type ('call' or 'put') [call]: ").strip().lower() or 'call'
    # 2. Ticker input
    while True:
        ticker = input("Enter stock ticker (e.g., ^SPX) [default: ^SPX]: ").strip().upper()
        if ticker == '':
            ticker = "^SPX"
        try:
            ticker_obj = yf.Ticker(ticker)
            S_default = ticker_obj.history(period="1d")['Close'].iloc[-1]
            print(f"Latest available spot price for {ticker}: {S_default:.2f}")
            expirations = ticker_obj.options
            if not expirations:
                print("No options data available for this ticker. Please try another.")
                continue
            print("\nAvailable expirations:")
            for i, exp in enumerate(expirations):
                print(f"  {i+1}. {exp}")
            while True:
                exp_input = input(f"Select expiration by number [default: 1]: ").strip()
                if exp_input == '':
                    exp_idx = 0
                else:
                    try:
                        exp_idx = int(exp_input) - 1
                        if exp_idx < 0 or exp_idx >= len(expirations):
                            print("Invalid selection. Try again.")
                            continue
                    except Exception:
                        print("Invalid input. Try again.")
                        continue
                expiration = expirations[exp_idx]
                break
            opt_chain = ticker_obj.option_chain(expiration)
            if option_type == 'call':
                strikes = opt_chain.calls['strike'].values
            else:
                strikes = opt_chain.puts['strike'].values
            print(f"\nAvailable strikes for {expiration} ({option_type}s):")
            for i, strike in enumerate(strikes):
                print(f"  {i+1}. {strike}")
            while True:
                strike_input = input(f"Select strike by number [default: closest to spot]: ").strip()
                if strike_input == '':
                    closest_idx = (np.abs(strikes - S_default)).argmin()
                    K = float(strikes[closest_idx])
                    print(f"Selected strike: {K}")
                    break
                else:
                    try:
                        strike_idx = int(strike_input) - 1
                        if strike_idx < 0 or strike_idx >= len(strikes):
                            print("Invalid selection. Try again.")
                            continue
                        K = float(strikes[strike_idx])
                        print(f"Selected strike: {K}")
                        break
                    except Exception:
                        print("Invalid input. Try again.")
                        continue
            today = datetime.now().date()
            expiry_date = datetime.strptime(expiration, "%Y-%m-%d").date()
            T = (expiry_date - today).days / 365.0
            if T <= 0:
                print("Selected expiration is not valid (already expired). Please try again.")
                continue
            print(f"Time to expiration (T): {T:.4f} years")
            break
        except Exception as e:
            print(f"Could not fetch data for this ticker. Error: {e}\nPlease try again.")
    S = S_default
    r = get_input("Enter risk-free rate (e.g., 0.0421 for 4.21%)", 0.0421, float, lambda x: x >= 0)
    # Get market price from Yahoo for selected option
    if option_type == 'call':
        row = opt_chain.calls[opt_chain.calls['strike'] == K]
    else:
        row = opt_chain.puts[opt_chain.puts['strike'] == K]
    if not row.empty:
        market_price = float(row['lastPrice'].values[0])
        print(f"Market price for selected {option_type}: {market_price}")
    else:
        market_price = None
        print("No market price found for this option.")

    # Calculate implied volatility if possible, otherwise ask user
    if market_price is not None:
        iv = implied_volatility_option(market_price, S, K, T, r, option_type)
        if iv is not None:
            print(f"Implied volatility: {iv*100:.2f}% (used in binomial model)")
            sigma = iv
        else:
            print("Could not calculate implied volatility from market price.")
            sigma = get_input("Enter volatility (e.g., 0.2 for 20%)", 0.2, float, lambda x: x > 0)
    else:
        print("No market price available, using input volatility.")
        sigma = get_input("Enter volatility (e.g., 0.2 for 20%)", 0.2, float, lambda x: x > 0)

    N = get_input("Enter number of steps (e.g., 100)", 100, int, lambda x: x > 0)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    price = binomial_european_option_price(S, K, T, r, sigma, N, option_type)
    print("\n" + "="*50)
    print(f"European {option_type.capitalize()} Option Pricing (Binomial Model)")
    print("="*50)
    print(f"Underlying:        {ticker}")
    print(f"Spot price (S):    {S:.2f}")
    print(f"Strike (K):        {K:.2f}")
    print(f"Expiration:        {expiration}")
    print(f"Time to expiry:    {T*365:.0f} days ({T:.4f} years)")
    print(f"Risk-free rate:    {r*100:.2f}%")
    print(f"Volatility (σ):    {sigma*100:.2f}%")
    print(f"Steps (N):         {N}")
    print(f"u (up factor):     {u:.6f}")
    print(f"d (down factor):   {d:.6f}")
    print(f"p (risk-neutral):  {p:.6f}")
    if market_price is not None:
        print(f"Market price:      {market_price:.4f}")
        print(f"Model price:       {price:.4f}")
    else:
        print(f"Model price:       {price:.4f}")
    greeks = binomial_greeks_european_option(S, K, T, r, sigma, N, option_type)
    print("\nGreeks (Binomial estimates):")
    for greek, value in greeks.items():
        print(f"  {greek.capitalize():<6}: {value:.4f}")
    print("="*50)
