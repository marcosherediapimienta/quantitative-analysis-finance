"""
American Option Pricing using Least Squares Monte Carlo (Longstaff-Schwartz)
Permite elegir tipo (call/put), ticker, vencimiento, strike, y todos los parámetros desde CLI, mostrando listas numeradas para expiries y strikes.
"""
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
from scipy.optimize import bisect
import copy

def american_option_lsmc(S0, K, T, r, sigma, N=50, M=10000, option_type='put'):
    """
    Price an American option using the Longstaff-Schwartz Monte Carlo method.
    Args:
        S0: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        N: Number of time steps
        M: Number of simulated paths
        option_type: 'call' or 'put'
    Returns:
        Estimated option price
    """
    dt = T / N
    discount = np.exp(-r * dt)
    # Simulate paths
    S = np.zeros((M, N+1))
    S[:,0] = S0
    for t in range(1, N+1):
        z = np.random.randn(M)
        S[:,t] = S[:,t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    # Payoff matrix
    if option_type == 'call':
        payoff = np.maximum(S - K, 0)
    else:
        payoff = np.maximum(K - S, 0)
    V = payoff[:,-1].copy()
    # Backward induction
    for t in range(N-1, 0, -1):
        itm = payoff[:,t] > 0
        if np.any(itm):
            X = S[itm, t]
            Y = V[itm] * discount
            # Regression (basis: 1, S, S^2)
            A = np.vstack([np.ones_like(X), X, X**2]).T
            coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
            continuation = coeffs[0] + coeffs[1]*X + coeffs[2]*X**2
            exercise = payoff[itm, t]
            exercise_now = exercise > continuation
            V[itm] = np.where(exercise_now, exercise, Y)
        V[~itm] = V[~itm] * discount
    price = np.mean(V) * np.exp(-r * dt)
    return price

def implied_vol_lsmc(market_price, S0, K, T, r, N, M, option_type, sigma_bounds=(0.01, 3.0), tol=1e-4, maxiter=50):
    """
    Calcula la volatilidad implícita usando LSMC y método de bisección.
    Si no converge, devuelve None.
    """
    def objective(sigma):
        price = american_option_lsmc(S0, K, T, r, sigma, N, M, option_type)
        return price - market_price
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imp_vol = bisect(objective, sigma_bounds[0], sigma_bounds[1], xtol=tol, maxiter=maxiter)
        return imp_vol
    except Exception:
        return None

def find_optimal_steps(S0, K, T, r, sigma, M, option_type, tol=1e-3, max_N=500, step=50):
    """
    Automatically find the optimal number of time steps N for LSMC.
    Increases N until price difference is below tol (relative) or max_N is reached.
    Returns (optimal_N, price).
    """
    N = step
    last_price = american_option_lsmc(S0, K, T, r, sigma, N, M, option_type)
    while N + step <= max_N:
        N_new = N + step
        price = american_option_lsmc(S0, K, T, r, sigma, N_new, M, option_type)
        if abs(price - last_price) / max(abs(price), 1e-8) < tol:
            return N_new, price
        N = N_new
        last_price = price
    return N, last_price

def lsmc_greeks(S0, K, T, r, sigma, N, M, option_type):
    """
    Estimate option Greeks (Delta, Gamma, Vega, Theta, Rho) using finite differences and LSMC.
    Uses robust step sizes for more stable results.
    """
    dS = max(0.5, S0 * 0.01)
    d_sigma = max(0.01, sigma * 0.05)
    dT = 1/365
    dr = 0.001
    # DELTA
    price_up = american_option_lsmc(S0 + dS, K, T, r, sigma, N, M, option_type)
    price_down = american_option_lsmc(S0 - dS, K, T, r, sigma, N, M, option_type)
    price = american_option_lsmc(S0, K, T, r, sigma, N, M, option_type)
    delta = (price_up - price_down) / (2 * dS)
    gamma = (price_up - 2*price + price_down) / (dS**2)
    # VEGA
    price_vega = american_option_lsmc(S0, K, T, r, sigma + d_sigma, N, M, option_type)
    vega = (price_vega - price) / d_sigma
    # THETA
    price_theta = american_option_lsmc(S0, K, T - dT, r, sigma, N, M, option_type)
    theta = (price_theta - price) / dT  # Per day
    # RHO
    price_rho = american_option_lsmc(S0, K, T, r + dr, sigma, N, M, option_type)
    rho = (price_rho - price) / dr
    return {
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Theta': theta,
        'Rho': rho
    }

if __name__ == "__main__":
    print("--- American Option LSMC (Longstaff-Schwartz) ---")
    ticker = input("Ticker symbol (e.g. AAPL): ").strip().upper()
    option_type = input("Option type ('call' or 'put') [put]: ").strip().lower() or 'put'
    stock = yf.Ticker(ticker)
    expiries = stock.options
    print("Available expiries:")
    for i, exp in enumerate(expiries):
        print(f"  [{i}] {exp}")
    expiry_idx = int(input("Select expiry by number: "))
    expiry = expiries[expiry_idx]
    opt_chain = stock.option_chain(expiry)
    opt_df = opt_chain.puts if option_type == 'put' else opt_chain.calls
    strikes = list(opt_df['strike'])
    print("Available strikes:")
    for i, strike in enumerate(strikes):
        print(f"  [{i}] {strike}")
    strike_idx = int(input("Select strike by number: "))
    K = float(strikes[strike_idx])
    S0 = stock.history(period='1d')['Close'].iloc[-1]
    today = datetime.now().date()
    expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
    T = (expiry_date - today).days / 365.0
    r = float(input("Risk-free rate (e.g. 0.05): "))
    M = int(input("Number of paths (e.g. 10000): "))
    N = int(input("Number of time steps (e.g. 50): "))
    # Try to get market price for the selected option
    market_price = opt_df.iloc[strike_idx]['lastPrice'] if 'lastPrice' in opt_df.columns else None
    sigma = None
    if market_price and market_price > 0:
        print(f"\n[INFO] Market price for selected option: {market_price:.4f}")
        print("[INFO] Calculating implied volatility using LSMC (may take a while)...")
        sigma = implied_vol_lsmc(market_price, S0, K, T, r, N, M//5, option_type)  # Fewer paths for IV for speed
        if sigma:
            print(f"[RESULT] Implied volatility (LSMC): {sigma:.4f}\n")
        else:
            print("[WARN] Implied volatility could not be found, using historical volatility.\n")
    if not sigma:
        hist = stock.history(period='1y')['Close']
        logret = np.log(hist / hist.shift(1)).dropna()
        sigma = np.std(logret) * np.sqrt(252)
        print(f"[INFO] Using historical volatility: {sigma:.4f}\n")
    print("========== OPTION INPUTS ==========")
    print(f"Spot price (S0):     {S0:.4f}")
    print(f"Strike (K):          {K:.4f}")
    print(f"Time to maturity:    {T:.6f} years")
    print(f"Risk-free rate (r):  {r:.4f}")
    print(f"Volatility (sigma):  {sigma:.4f}")
    print(f"Type:                {option_type}")
    print(f"Steps (N):           {N}")
    print(f"Paths (M):           {M}")
    print("===================================\n")
    price = american_option_lsmc(S0, K, T, r, sigma, N, M, option_type)
    greeks = lsmc_greeks(S0, K, T, r, sigma, N, M, option_type)
    print(f"[RESULT] American {option_type} option price (LSMC): {price:.4f}\n")
    print("========== LSMC GREEKS (finite diff) ==========")
    for greek, value in greeks.items():
        print(f"{greek:>8}: {value: .6f}")
    print("==============================================\n")
