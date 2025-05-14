import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq
from datetime import datetime
import yfinance as yf

def finite_difference_european_call(S0, K, T, r, sigma, Smax=2, M=100, N=100):
    """
    Finite difference pricing for a European call option (explicit method).
    S0: spot price
    K: strike
    T: time to maturity (years)
    r: risk-free rate
    sigma: volatility
    Smax: multiple of K for the max underlying price in the grid
    M: number of price steps
    N: number of time steps
    """
    S_max = Smax * K
    dS = S_max / M
    dt = T / N
    grid = np.zeros((M+1, N+1))
    S_values = np.linspace(0, S_max, M+1)
    # Payoff at maturity
    grid[:, -1] = np.maximum(S_values - K, 0)
    # Boundary conditions
    grid[0, :] = 0
    grid[-1, :] = S_max - K * np.exp(-r * dt * np.arange(N+1)[::-1])
    # Backward induction
    for j in reversed(range(N)):
        for i in range(1, M):
            delta = (grid[i+1, j+1] - grid[i-1, j+1]) / (2 * dS)
            gamma = (grid[i+1, j+1] - 2*grid[i, j+1] + grid[i-1, j+1]) / (dS**2)
            grid[i, j] = grid[i, j+1] + dt * (
                0.5 * sigma**2 * S_values[i]**2 * gamma +
                r * S_values[i] * delta - r * grid[i, j+1]
            )
    # Interpolate to get the price at S0
    return np.interp(S0, S_values, grid[:, 0])

def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

def implied_volatility_call(market_price, S, K, T, r, tol=1e-6, max_iter=100, track_iterations=False):
    def black_scholes_call_price_inner(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    def vega(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * stats.norm.pdf(d1) * np.sqrt(T)
    sigma = 0.2
    iterations = []
    for _ in range(max_iter):
        price = black_scholes_call_price_inner(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma)
        diff = price - market_price
        if track_iterations:
            iterations.append((sigma, price, diff))
        if abs(v) < 1e-8:
            break
        if abs(diff) < tol:
            if track_iterations:
                return sigma, iterations
            return sigma
        sigma = sigma - diff / v
        if sigma <= 0:
            sigma = 1e-6
    # If Newton-Raphson fails, use Brent
    def objective(sigma_):
        return black_scholes_call_price_inner(S, K, T, r, sigma_) - market_price
    try:
        root = brentq(objective, 1e-6, 5.0, xtol=tol, maxiter=max_iter)
        if track_iterations:
            iterations.append((root, black_scholes_call_price_inner(S, K, T, r, root), black_scholes_call_price_inner(S, K, T, r, root) - market_price))
            return root, iterations
        return root
    except Exception:
        if track_iterations:
            return None, iterations
        return None

if __name__ == "__main__":
    
    # Request ticker and option parameters
    ticker = input("Enter the ticker (e.g., ^SPX): ")
    stock = yf.Ticker(ticker)
    expirations = stock.options
    print("\nAvailable expiration dates for this ticker:")
    for i, exp in enumerate(expirations):
        print(f"  [{i}] {exp}")
    idx = int(input("\nSelect the number of the expiration date you want to use: "))
    expiry = expirations[idx]

    # Get the rest of the data
    options = stock.option_chain(expiry)
    strikes = options.calls['strike'].values
    print("\nAvailable strikes for this expiration:")
    print(strikes)
    strike = float(input("\nEnter the strike you want to use (from the list above): "))
    row = options.calls[options.calls['strike'] == strike]
    if row.empty:
        raise ValueError('Strike not found in the option chain')
    market_price = float(row['lastPrice'].iloc[0])
    S0 = stock.history(period='1d')['Close'].iloc[-1]
    r = 0.0421  # You can change this or request it from the user

    # Calculate T automatically

    today = datetime.now().date()
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    T = (expiry_date - today).days / 365.0
    print(f"\nTime to expiration (T): {T:.4f} years")
    print(f"Current underlying price: {S0:.2f}")
    print(f"Market price of the option: {market_price:.4f}")

    # Calculate implied volatility and price

    sigma, iterations = implied_volatility_call(market_price, S0, strike, T, r, track_iterations=True)
    if sigma is not None:
        print(f"Implied volatility: {sigma*100:.2f}%")
    else:
        print("Could not calculate implied volatility")
    price = finite_difference_european_call(S0, strike, T, r, sigma)
    print(f"European call option price (Finite Differences, IV): {price:.4f}")
