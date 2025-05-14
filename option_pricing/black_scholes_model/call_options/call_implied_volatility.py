from scipy.stats import norm
import numpy as np
from scipy.optimize import brentq
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def calculate_historical_volatility(ticker: str, window: int = 252) -> float:
    """
    Calculate historical volatility using daily returns.
    
    Args:
        ticker: Stock ticker symbol
        window: Number of trading days to consider (default: 252 for 1 year)
    
    Returns:
        float: Annualized historical volatility
    """
    # Get historical data
    stock = yf.Ticker(ticker)
    hist = stock.history(period=f"{window+1}d")
    
    # Calculate daily returns
    returns = np.log(hist['Close'] / hist['Close'].shift(1))
    
    # Calculate annualized volatility
    return returns.std() * np.sqrt(252)

def black_scholes_call_price(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European call option.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility
    
    Returns:
        float: Call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def vega(S, K, T, r, sigma):
    """
    Calculate the vega (same for calls and puts).
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility
    
    Returns:
        float: Option vega
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_volatility_newton(market_price, S, K, T, r, tol=1e-6, max_iter=25):
    """
    Calculate implied volatility using the Newton-Raphson method for calls.
    
    Args:
        market_price: Market price of the call option
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
    
    Returns:
        float: Implied volatility or None if it doesn't converge
    """
    # Initial sigma value (20%)
    sigma = 0.2
    
    print(f"\nCalculating implied volatility with parameters:")
    print(f"Market Price: ${market_price:.2f}")
    print(f"Stock Price: ${S:.2f}")
    print(f"Strike Price: ${K:.2f}")
    print(f"Time to Expiration: {T:.4f} years")
    print(f"Risk-free Rate: {r:.2%}")
    
    for i in range(max_iter):
        # Calculate theoretical price and vega
        price = black_scholes_call_price(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma)
        
        if abs(v) < 1e-8:
            print(f"\nVega too small ({v:.8f}), switching to Brent's method")
            return implied_volatility_brent(market_price, S, K, T, r)
        
        diff = price - market_price
        
        if abs(diff) < tol:
            print(f"\nConverged after {i+1} iterations")
            print(f"Final IV: {sigma:.2%}")
            return sigma
        
        sigma = sigma - diff / v
        
        if sigma <= 0:
            sigma = 0.0001
            print(f"\nSigma became negative, resetting to {sigma}")
    
    print("\nNewton-Raphson did not converge, switching to Brent's method")
    return implied_volatility_brent(market_price, S, K, T, r)

def implied_volatility_brent(market_price, S, K, T, r, tol=1e-6):
    """
    Calculate implied volatility using Brent's method for calls.
    """
    def objective(sigma):
        return black_scholes_call_price(S, K, T, r, sigma) - market_price
    
    ranges = [(0.0001, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0)]
    
    for a, b in ranges:
        try:
            fa = objective(a)
            fb = objective(b)
            
            if fa * fb < 0:
                print(f"\nBrent's method found solution in range [{a:.4f}, {b:.4f}]")
                return brentq(objective, a, b, xtol=tol)
            else:
                print(f"\nNo solution in range [{a:.4f}, {b:.4f}]")
                print(f"f({a:.4f}) = {fa:.4f}, f({b:.4f}) = {fb:.4f}")
        except Exception as e:
            print(f"\nError in Brent's method for range [{a:.4f}, {b:.4f}]: {str(e)}")
            continue
    
    print("\nBrent's method failed to find a solution in any range")
    return None

def calculate_greeks(S, K, T, r, sigma):
    """
    Calculate all option Greeks for calls.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility
    
    Returns:
        dict: Dictionary containing all Greeks
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate Greeks for calls
    delta = norm.cdf(d1)  # Call delta
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))  # Same for calls and puts
    theta = (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega = S * np.sqrt(T) * norm.pdf(d1)  # Same for calls and puts
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)  # Call rho
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

def plot_greeks(S, K, T, r, sigma):
    """
    Plot all option Greeks against different parameters for calls.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Delta and Gamma vs Stock Price
    S_range = np.linspace(S * 0.5, S * 1.5, 100)
    deltas = []
    gammas = []
    
    for s in S_range:
        greeks = calculate_greeks(s, K, T, r, sigma)
        deltas.append(greeks['delta'])
        gammas.append(greeks['gamma'])
    
    ax1.plot(S_range, deltas, label='Delta')
    ax1.set_title('Call Delta vs Stock Price')
    ax1.set_xlabel('Stock Price')
    ax1.set_ylabel('Delta')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(S_range, gammas, label='Gamma', color='orange')
    ax2.set_title('Gamma vs Stock Price')
    ax2.set_xlabel('Stock Price')
    ax2.set_ylabel('Gamma')
    ax2.grid(True)
    ax2.legend()
    
    # 2. Theta vs Time to Expiration
    T_range = np.linspace(0.01, T * 2, 100)
    thetas = []
    
    for t in T_range:
        greeks = calculate_greeks(S, K, t, r, sigma)
        thetas.append(greeks['theta'])
    
    ax3.plot(T_range, thetas, label='Theta', color='green')
    ax3.set_title('Call Theta vs Time to Expiration')
    ax3.set_xlabel('Time to Expiration (years)')
    ax3.set_ylabel('Theta')
    ax3.grid(True)
    ax3.legend()
    
    # 3. Vega and Rho vs Volatility
    sigma_range = np.linspace(0.1, sigma * 2, 100)
    vegas = []
    rhos = []
    
    for s in sigma_range:
        greeks = calculate_greeks(S, K, T, r, s)
        vegas.append(greeks['vega'])
        rhos.append(greeks['rho'])
    
    ax4.plot(sigma_range, vegas, label='Vega', color='red')
    ax4.plot(sigma_range, rhos, label='Rho', color='purple')
    ax4.set_title('Vega and Call Rho vs Volatility')
    ax4.set_xlabel('Volatility')
    ax4.set_ylabel('Value')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('call_greeks_analysis.png')
    plt.close()

# Parameters for call option
ticker = "^SPX"  # Stock ticker
S = 5886.55  # Current stock price
K = 5730     # Strike price
T = 31/365   # Time to expiration
r = 0.0421   # Risk-free rate 
C_market = 237.50 # Market price of the call option (example value)

# Calculate historical volatility
hist_vol = calculate_historical_volatility(ticker)

# Calculate implied volatility
print("\nStarting implied volatility calculation for call option...")
iv = implied_volatility_newton(C_market, S, K, T, r)

# Calculate all Greeks
sigma = iv if iv is not None else 0.2
greeks = calculate_greeks(S, K, T, r, sigma)

# Display results
print(f"\nResults:")
print(f"Historical Volatility: {hist_vol:.2%}")
print(f"Implied Volatility: {iv:.2%}" if iv is not None else "Implied Volatility: No convergence")
print("\nGreeks:")
print(f"Delta: {greeks['delta']:.4f} (change in option price per $1 change in stock price)")
print(f"Gamma: {greeks['gamma']:.4f} (change in delta per $1 change in stock price)")
print(f"Theta: {greeks['theta']:.4f} (change in option price per day)")
print(f"Vega: {greeks['vega']:.4f} (change in option price per 1% change in volatility)")
print(f"Rho: {greeks['rho']:.4f} (change in option price per 1% change in interest rate)")

# Plot Greeks
print("\nGenerating Greeks plots for call option...")
plot_greeks(S, K, T, r, sigma)
print("Plots saved as 'call_greeks_analysis.png'")