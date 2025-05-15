import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq
from datetime import datetime
import yfinance as yf

def finite_difference_european_put(S0, K, T, r, sigma, Smax=2, M=100, N=100):
    """
    Finite difference pricing for a European put option (explicit method).
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
    grid[:, -1] = np.maximum(K - S_values, 0)
    # Boundary conditions
    grid[0, :] = K * np.exp(-r * dt * np.arange(N+1)[::-1])
    grid[-1, :] = 0
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

def black_scholes_put_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

def implied_volatility_put(market_price, S, K, T, r, tol=1e-6, max_iter=100, track_iterations=False):
    def black_scholes_put_price_inner(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    def vega(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * stats.norm.pdf(d1) * np.sqrt(T)
    sigma = 0.2
    iterations = []
    for _ in range(max_iter):
        price = black_scholes_put_price_inner(S, K, T, r, sigma)
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
        return black_scholes_put_price_inner(S, K, T, r, sigma_) - market_price
    try:
        root = brentq(objective, 1e-6, 5.0, xtol=tol, maxiter=max_iter)
        if track_iterations:
            iterations.append((root, black_scholes_put_price_inner(S, K, T, r, root), black_scholes_put_price_inner(S, K, T, r, root) - market_price))
            return root, iterations
        return root
    except Exception:
        if track_iterations:
            return None, iterations
        return None

def finite_difference_greeks_put(S0, K, T, r, sigma, Smax=3, M=100, N=1000, bump=1e-5, debug=False):
    """
    Calculate Greeks for a European put option using finite differences (bumping) on the finite difference price.
    Returns a dict with Delta, Gamma, Vega, Theta, Rho.
    """
    # Delta
    dS = S0 * bump
    price_up = finite_difference_european_put(S0 + dS, K, T, r, sigma, Smax, M, N)
    price_down = finite_difference_european_put(S0 - dS, K, T, r, sigma, Smax, M, N)
    price = finite_difference_european_put(S0, K, T, r, sigma, Smax, M, N)
    delta = (price_up - price_down) / (2 * dS)
    gamma = (price_up - 2 * price + price_down) / (dS ** 2)
    # Vega
    dsigma = bump
    price_vega = finite_difference_european_put(S0, K, T, r, sigma + dsigma, Smax, M, N)
    vega = (price_vega - price) / dsigma
    # Theta (should be negative for puts)
    dT = min(bump, T/10) if T > 0.01 else 1e-6
    if T - dT > 0:
        price_theta = finite_difference_european_put(S0, K, T - dT, r, sigma, Smax, M, N)
        theta = (price_theta - price) / dT
    else:
        theta = float('nan')
    # Rho
    dr = bump
    price_rho = finite_difference_european_put(S0, K, T, r + dr, sigma, Smax, M, N)
    rho = (price_rho - price) / dr
    return {
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Theta': theta,
        'Rho': rho
    }

if __name__ == "__main__":
    ticker = input("Enter the ticker (e.g., ^SPX): ")
    stock = yf.Ticker(ticker)
    expirations = stock.options
    print("\nAvailable expiration dates for this ticker:")
    for i, exp in enumerate(expirations):
        print(f"  [{i}] {exp}")
    idx = int(input("\nSelect the number of the expiration date you want to use: "))
    expiry = expirations[idx]
    options = stock.option_chain(expiry)
    strikes = options.puts['strike'].values
    print("\nAvailable strikes for this expiration:")
    print(strikes)
    strike = float(input("\nEnter the strike you want to use (from the list above): "))
    row = options.puts[options.puts['strike'] == strike]
    if row.empty:
        raise ValueError('Strike not found in the option chain')
    market_price = float(row['lastPrice'].iloc[0])
    S0 = stock.history(period='1d')['Close'].iloc[-1]
    r = 0.0421  # You can change this or request it from the user
    today = datetime.now().date()
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    T = (expiry_date - today).days / 365.0
    print(f"\nTime to expiration (T): {T:.4f} years")
    print(f"Current underlying price: {S0:.2f}")
    print(f"Market price of the option: {market_price:.4f}")
    sigma, iterations = implied_volatility_put(market_price, S0, strike, T, r, track_iterations=True)
    if sigma is not None:
        print(f"Implied volatility: {sigma*100:.2f}%")
    else:
        print("Could not calculate implied volatility")
    price = finite_difference_european_put(S0, strike, T, r, sigma)
    print(f"European put option price (Finite Differences, IV): {price:.4f}")

    # Calculate and print Greeks
    greeks = finite_difference_greeks_put(S0, strike, T, r, sigma, debug=True)
    print("\nGreeks (Finite Difference, bumping):")
    for greek, value in greeks.items():
        print(f"  {greek}: {value:.4f}")
