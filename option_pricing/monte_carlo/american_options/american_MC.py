import numpy as np
import yfinance as yf
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# Longstaff-Schwartz for American options

def american_option_longstaff_schwartz(S, K, T, r, sigma, n_sim=10000, n_steps=50, option_type='call', seed=None, regression_type='quadratic'):
    dt = T / n_steps
    discount = np.exp(-r * dt)
    S_paths = np.zeros((n_sim, n_steps + 1))
    S_paths[:, 0] = S
    rng = np.random.default_rng(seed)
    for t in range(1, n_steps + 1):
        Z = rng.standard_normal(n_sim)
        S_paths[:, t] = S_paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    if option_type == 'call':
        payoff = np.maximum(S_paths - K, 0)
    else:
        payoff = np.maximum(K - S_paths, 0)
    V = payoff[:, -1]
    for t in range(n_steps - 1, 0, -1):
        itm = payoff[:, t] > 0
        if np.any(itm):
            X = S_paths[itm, t]
            Y = V[itm] * discount
            if regression_type == 'linear':
                A = np.vstack([np.ones_like(X), X]).T
            elif regression_type == 'quadratic':
                A = np.vstack([np.ones_like(X), X, X**2]).T
            elif regression_type == 'cubic':
                A = np.vstack([np.ones_like(X), X, X**2, X**3]).T
            else:
                A = np.vstack([np.ones_like(X), X, X**2]).T
            coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
            if regression_type == 'linear':
                continuation = coeffs[0] + coeffs[1] * X
            elif regression_type == 'quadratic':
                continuation = coeffs[0] + coeffs[1] * X + coeffs[2] * X**2
            elif regression_type == 'cubic':
                continuation = coeffs[0] + coeffs[1] * X + coeffs[2] * X**2 + coeffs[3] * X**3
            else:
                continuation = coeffs[0] + coeffs[1] * X + coeffs[2] * X**2
            exercise = payoff[itm, t]
            exercise_now = exercise > continuation
            idx = np.where(itm)[0][exercise_now]
            V[idx] = exercise[exercise_now]
        V = V * discount
    price = np.mean(V)
    return price

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

def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_volatility_call(S, K, T, r, market_price, tol=1e-6, max_iter=25):
    sigma = 0.2
    for i in range(max_iter):
        price = black_scholes_call_price(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma)
        if abs(v) < 1e-8:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma = sigma - diff / v
        if sigma <= 0:
            sigma = 0.0001
    def objective(s): 
        return black_scholes_call_price(S, K, T, r, s) - market_price
    try:
        return brentq(objective, 0.0001, 5.0, xtol=tol)
    except:
        return None

def implied_volatility_put(S, K, T, r, market_price, tol=1e-6, max_iter=25):
    sigma = 0.2
    for i in range(max_iter):
        price = black_scholes_put_price(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma)
        if abs(v) < 1e-8:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma = sigma - diff / v
        if sigma <= 0:
            sigma = 0.0001
    def objective(s):
        return black_scholes_put_price(S, K, T, r, s) - market_price
    try:
        return brentq(objective, 0.0001, 5.0, xtol=tol)
    except:
        return None

def american_mc_greeks(S, K, T, r, sigma, n_sim, n_steps, option_type='call', seed=None, regression_type='quadratic'):
    base_rng = np.random.default_rng(seed)
    seeds = base_rng.integers(0, 1e9, size=7)  # 7 seeds for each calculation
    eps_S = S * 0.01
    price_up = american_option_longstaff_schwartz(S + eps_S, K, T, r, sigma, n_sim, n_steps, option_type, seed=int(seeds[0]), regression_type=regression_type)
    price_down = american_option_longstaff_schwartz(S - eps_S, K, T, r, sigma, n_sim, n_steps, option_type, seed=int(seeds[1]), regression_type=regression_type)
    price = american_option_longstaff_schwartz(S, K, T, r, sigma, n_sim, n_steps, option_type, seed=int(seeds[2]), regression_type=regression_type)
    delta = (price_up - price_down) / (2 * eps_S)
    gamma = (price_up - 2 * price + price_down) / (eps_S ** 2)
    eps_sigma = 0.01
    price_vega_up = american_option_longstaff_schwartz(S, K, T, r, sigma + eps_sigma, n_sim, n_steps, option_type, seed=int(seeds[3]), regression_type=regression_type)
    price_vega_down = american_option_longstaff_schwartz(S, K, T, r, sigma - eps_sigma, n_sim, n_steps, option_type, seed=int(seeds[4]), regression_type=regression_type)
    vega = (price_vega_up - price_vega_down) / (2 * eps_sigma)
    dt = 1/365
    if T - dt > 0:
        price_Tdt = american_option_longstaff_schwartz(S, K, T - dt, r, sigma, n_sim, n_steps, option_type, seed=int(seeds[5]), regression_type=regression_type)
        theta = (price_Tdt - price) / dt
    else:
        theta = float('nan')
    dr = 0.01
    price_rho_up = american_option_longstaff_schwartz(S, K, T, r + dr, sigma, n_sim, n_steps, option_type, seed=int(seeds[6]), regression_type=regression_type)
    price_rho_down = american_option_longstaff_schwartz(S, K, T, r - dr, sigma, n_sim, n_steps, option_type, seed=int(seeds[2]), regression_type=regression_type)
    rho = (price_rho_up - price_rho_down) / (2 * dr)
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        theta = (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1)
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }

def get_historical_volatility(ticker, window=252):
    """
    Calculates the annualized historical volatility using daily closing prices from Yahoo Finance.
    By default uses a 1-year window (252 trading days).
    """
    try:
        data = yf.Ticker(ticker).history(period=f"{window+1}d")['Close']
        returns = np.log(data / data.shift(1)).dropna()
        hist_vol = returns.std() * np.sqrt(252)
        return float(hist_vol)
    except Exception as e:
        print(f"Error calculating historical volatility for {ticker}: {e}")
        return 0.2  # fallback

def valor_intrinseco_put_desc(S, K, T, r):
    return max(K - S * np.exp(-r * T), 0)

if __name__ == "__main__":
    print("\nAmerican Option Pricing via Monte Carlo (Longstaff-Schwartz, Yahoo Finance data)")
    # Ask option type
    while True:
        option_type = input("Which type of option do you want to analyze? (call/put) [default: call]: ").strip().lower()
        if option_type == '':
            option_type = 'call'
        if option_type in ['call', 'put']:
            break
        else:
            print("Please enter 'call' or 'put'.")
    # Ticker input
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
            strikes = opt_chain.calls['strike'].values
            print(f"\nAvailable strikes for {expiration}:")
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
            from datetime import datetime
            today = pd.Timestamp.today().normalize()
            expiry_date = pd.Timestamp(expiration)
            T_days = (expiry_date - today).days
            if T_days <= 0:
                print("Selected expiration is not valid (already expired). Please try again.")
                continue
            print(f"Time to expiration: {T_days} days")
            break
        except Exception as e:
            print(f"Could not fetch data for this ticker. Error: {e}\nPlease try again.")
    S = S_default
    T = T_days / 365
    r = get_input("Enter risk-free rate (as decimal, e.g., 0.0421 for 4.21%)", 0.0421, float, lambda x: x >= 0)
    n_sim = get_input("Number of Monte Carlo simulations", 10000, int, lambda x: x > 0)
    n_steps = get_input("Number of time steps (Longstaff-Schwartz)", 50, int, lambda x: x > 1)
    regression_type = input("Regression type for continuation value ('linear', 'quadratic', or 'cubic') [quadratic]: ").strip().lower()
    if regression_type not in ['linear', 'quadratic', 'cubic']:
        regression_type = 'quadratic'
    seed = get_input("Random seed for reproducibility (int, blank for random)", 42, int)
    # Market price
    try:
        call_row = opt_chain.calls[opt_chain.calls['strike'] == K]
        put_row = opt_chain.puts[opt_chain.puts['strike'] == K]
        if not call_row.empty:
            C_market = float(call_row['lastPrice'].values[0])
        else:
            C_market = 1.0
        if not put_row.empty:
            P_market = float(put_row['lastPrice'].values[0])
        else:
            P_market = 1.0
    except Exception:
        C_market = 1.0
        P_market = 1.0
    # Implied volatility
    if option_type == 'call':
        iv = implied_volatility_call(S, K, T, r, C_market)
        market_price = C_market
    else:
        vi_put = valor_intrinseco_put_desc(S, K, T, r)
        market_price = P_market
        if market_price < vi_put:
            print(f"[WARNING] The market price of the put (${market_price:.2f}) is less than the discounted intrinsic value (${vi_put:.2f}). It is not possible to find a consistent implied volatility. Using historical volatility as fallback.")
            iv = None
        else:
            iv = implied_volatility_put(S, K, T, r, P_market)
    if iv is None:
        try:
            iv = get_historical_volatility(ticker, window=252)
            print(f"No implied volatility found, using 1y historical volatility as fallback: {iv:.2%}")
        except Exception as e:
            print(f'Could not fetch historical volatility: {e}. Using 20% as fallback.')
            iv = 0.2
    print(f"\nImplied Volatility ({option_type}): {iv:.2%}")
    # Monte Carlo pricing
    mc_price = american_option_longstaff_schwartz(S, K, T, r, iv, n_sim, n_steps, option_type=option_type, seed=seed, regression_type=regression_type)
    # Greeks calculation (analytical Black-Scholes)
    greeks = calculate_greeks(S, K, T, r, iv, option_type=option_type)
    # Greeks calculation by Monte Carlo (finite differences)
    n_sim_greeks = max(100000, n_sim)
    mc_greek_vals = american_mc_greeks(S, K, T, r, iv, n_sim_greeks, n_steps, option_type=option_type, seed=seed, regression_type=regression_type)
    # Warning message if the market price is less than the discounted intrinsic value
    if option_type == 'put':
        vi_put = valor_intrinseco_put_desc(S, K, T, r)
        if market_price < vi_put:
            print(f"[WARNING] The market price of the put (${market_price:.2f}) is less than the discounted intrinsic value (${vi_put:.2f}). It was not possible to find a consistent implied volatility. Historical volatility was used as fallback.")
    print("\n" + "="*50)
    print(f"{'RESULTS':^50}")
    print("="*50)
    print(f"{'Underlying:':<25}{ticker}")
    print(f"{'Spot Price:':<25}${S:.2f}")
    print(f"{'Strike Price:':<25}${K:.2f}")
    print(f"{'Expiration:':<25}{expiration}")
    print(f"{'Time to Expiration:':<25}{T*365:.0f} days ({T:.4f} years)")
    print(f"{'Risk-free Rate:':<25}{r:.2%}")
    print(f"{'Market Price:':<25}${C_market if option_type=='call' else P_market:.2f}")
    print(f"{'Implied Volatility:':<25}{iv:.2%}")
    print(f"{'MC Price:':<25}${mc_price:.2f}")
    print("-"*50)
    print(f"{'Greek':<10}{'Value':>15}{'Description':>25}")
    print("-"*50)
    print(f"{'Delta':<10}{greeks['delta']:>15.4f}{'per $1 change in spot':>25}")
    print(f"{'Gamma':<10}{greeks['gamma']:>15.4f}{'per $1 change in spot':>25}")
    print(f"{'Theta':<10}{greeks['theta']:>15.4f}{'per day':>25}")
    print(f"{'Vega':<10}{greeks['vega']:>15.4f}{'per 1 vol point':>25}")
    print(f"{'Rho':<10}{greeks['rho']:>15.4f}{'per 1% rate change':>25}")
    print("-"*50)
    print(f"{'MC Greeks':^50}")
    print("-"*50)
    print(f"{'Delta':<10}{mc_greek_vals['delta']:>15.4f}{'per $1 change in spot':>25}")
    print(f"{'Gamma':<10}{mc_greek_vals['gamma']:>15.4f}{'per $1 change in spot':>25}")
    print(f"{'Theta':<10}{mc_greek_vals['theta']:>15.4f}{'per day':>25}")
    print(f"{'Vega':<10}{mc_greek_vals['vega']:>15.4f}{'per 1 vol point':>25}")
    print(f"{'Rho':<10}{mc_greek_vals['rho']:>15.4f}{'per 1% rate change':>25}")
    print("="*50)
