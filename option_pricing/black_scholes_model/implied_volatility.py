from scipy.stats import norm
import numpy as np
from scipy.optimize import brentq
import yfinance as yf
import pandas as pd

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
    # Multiply by sqrt(252) to annualize (252 trading days in a year)
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
    Calculate the vega (derivative of option price with respect to volatility).
    
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
    Calculate implied volatility using the Newton-Raphson method.
    
    Args:
        market_price: Market price of the option
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
    
    # Print initial parameters for debugging
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
        
        # If vega is too small, the method might be unstable
        if abs(v) < 1e-8:
            print(f"\nVega too small ({v:.8f}), switching to Brent's method")
            return implied_volatility_brent(market_price, S, K, T, r)
        
        # Calculate difference between theoretical and market price
        diff = price - market_price
        
        # If difference is less than tolerance, we've converged
        if abs(diff) < tol:
            print(f"\nConverged after {i+1} iterations")
            print(f"Final IV: {sigma:.2%}")
            return sigma
        
        # Update sigma using Newton-Raphson formula
        sigma = sigma - diff / v
        
        # Ensure sigma is positive
        if sigma <= 0:
            sigma = 0.0001
            print(f"\nSigma became negative, resetting to {sigma}")
    
    print("\nNewton-Raphson did not converge, switching to Brent's method")
    return implied_volatility_brent(market_price, S, K, T, r)

def implied_volatility_brent(market_price, S, K, T, r, tol=1e-6):
    """
    Calculate implied volatility using Brent's method.
    This is used as a fallback when Newton-Raphson fails.
    
    Args:
        market_price: Market price of the option
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        tol: Convergence tolerance
    
    Returns:
        float: Implied volatility
    """
    def objective(sigma):
        return black_scholes_call_price(S, K, T, r, sigma) - market_price
    
    # Try different ranges for sigma
    ranges = [(0.0001, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0)]
    
    for a, b in ranges:
        try:
            # Check if the function changes sign in this interval
            fa = objective(a)
            fb = objective(b)
            
            if fa * fb < 0:  # Function changes sign
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

# Parameters
ticker = "NVDA"  # Stock ticker
S = 114.50  # Current stock price
K = 114     # Strike price
T = 33/365  # Time to expiration 
r = 0.0432  # Risk-free rate 
C_market = 7.85  # Market price of the call option

# Calculate historical volatility
hist_vol = calculate_historical_volatility(ticker)

# Calculate implied volatility
print("\nStarting implied volatility calculation...")
iv = implied_volatility_newton(C_market, S, K, T, r)

# Calculate vega
sigma = iv if iv is not None else 0.2  # Use calculated IV or initial value
vega_value = vega(S, K, T, r, sigma)

# Display results
print(f"\nResults:")
print(f"Historical Volatility: {hist_vol:.2%}")
print(f"Implied Volatility: {iv:.2%}" if iv is not None else "Implied Volatility: No convergence")
print(f"Vega: {vega_value:.8f}") 