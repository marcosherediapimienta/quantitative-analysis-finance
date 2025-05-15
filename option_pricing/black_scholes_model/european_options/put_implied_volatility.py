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

def black_scholes_put_price(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European put option.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility
    
    Returns:
        float: Put option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

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
    Calculate implied volatility using the Newton-Raphson method for puts.
    
    Args:
        market_price: Market price of the put option
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
        price = black_scholes_put_price(S, K, T, r, sigma)
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
    Calculate implied volatility using Brent's method for puts.
    """
    def objective(sigma):
        return black_scholes_put_price(S, K, T, r, sigma) - market_price
    
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
    Calculate all option Greeks for puts.
    
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
    
    # Calculate Greeks for puts
    delta = norm.cdf(d1) - 1  # Put delta
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))  # Same for calls and puts
    theta = (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    vega = S * np.sqrt(T) * norm.pdf(d1)  # Same for calls and puts
    rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)  # Put rho
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

def plot_greeks(S, K, T, r, sigma):
    """
    Plot all option Greeks against different parameters for puts.
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
    ax1.set_title('Put Delta vs Stock Price')
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
    ax3.set_title('Put Theta vs Time to Expiration')
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
    ax4.set_title('Vega and Put Rho vs Volatility')
    ax4.set_xlabel('Volatility')
    ax4.set_ylabel('Value')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('put_greeks_analysis.png')
    plt.close()

# Interactive parameter input for put option analysis
print("\nEuropean Put Option Implied Volatility & Greeks Calculator (Black-Scholes)")

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

# Ticker input
while True:
    ticker = input("Enter stock ticker (e.g., ^SPX) [default: ^SPX]: ").strip().upper()
    if ticker == '':
        ticker = "^SPX"
    try:
        ticker_obj = yf.Ticker(ticker)
        S_default = ticker_obj.history(period="1d")['Close'].iloc[-1]
        print(f"Latest available spot price for {ticker}: {S_default:.2f}")
        # Get available expirations
        expirations = ticker_obj.options
        if not expirations:
            print("No options data available for this ticker. Please try another.")
            continue
        print("\nAvailable expirations:")
        for i, exp in enumerate(expirations):
            print(f"  {i+1}. {exp}")
        # Ask user to select expiration
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
        # Get available strikes for this expiration
        opt_chain = ticker_obj.option_chain(expiration)
        strikes = opt_chain.puts['strike'].values
        print(f"\nAvailable strikes for {expiration}:")
        for i, strike in enumerate(strikes):
            print(f"  {i+1}. {strike}")
        # Ask user to select strike
        while True:
            strike_input = input(f"Select strike by number [default: closest to spot]: ").strip()
            if strike_input == '':
                # Find closest strike to spot
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
        # Calculate time to expiration in days
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

# Spot price (do not ask, use S_default from Yahoo)
S = S_default
# Time to expiration (days, do not ask, use T_days from Yahoo)
T = T_days / 365
# Risk-free rate (ask user)
r = get_input("Enter risk-free rate (as decimal, e.g., 0.0421 for 4.21%)", 0.0421, float, lambda x: x >= 0)
# Market price of the put option (do not ask, use from Yahoo)
try:
    put_row = opt_chain.puts[opt_chain.puts['strike'] == K]
    if not put_row.empty:
        P_market = float(put_row['lastPrice'].values[0])
        print(f"Market price for selected put: {P_market}")
    else:
        P_market = 1.0
except Exception:
    P_market = 1.0

# Calculate historical volatility
hist_vol = calculate_historical_volatility(ticker)

# Calculate implied volatility
print("\nStarting implied volatility calculation for put option...")
iv = implied_volatility_newton(P_market, S, K, T, r)

# Calculate all Greeks
sigma = iv if iv is not None else 0.2
greeks = calculate_greeks(S, K, T, r, sigma)

# Display results
print("\n" + "="*50)
print(f"{'RESULTS':^50}")
print("="*50)
print(f"{'Underlying:':<25}{ticker}")
print(f"{'Spot Price:':<25}${S:.2f}")
print(f"{'Strike Price:':<25}${K:.2f}")
print(f"{'Expiration:':<25}{expiration}")
print(f"{'Time to Expiration:':<25}{T*365:.0f} days ({T:.4f} years)")
print(f"{'Risk-free Rate:':<25}{r:.2%}")
print(f"{'Market Put Price:':<25}${P_market:.2f}")
print(f"{'Historical Volatility:':<25}{hist_vol:.2%}")
if iv is not None:
    print(f"{'Implied Volatility:':<25}{iv:.2%}")
else:
    print(f"{'Implied Volatility:':<25}No convergence")
print("-"*50)
print(f"{'Greek':<10}{'Value':>15}{'Description':>25}")
print("-"*50)
print(f"{'Delta':<10}{greeks['delta']:>15.4f}{'per $1 change in spot':>25}")
print(f"{'Gamma':<10}{greeks['gamma']:>15.4f}{'per $1 change in spot':>25}")
print(f"{'Theta':<10}{greeks['theta']:>15.4f}{'per day':>25}")
print(f"{'Vega':<10}{greeks['vega']:>15.4f}{'per 1% vol change':>25}")
print(f"{'Rho':<10}{greeks['rho']:>15.4f}{'per 1% rate change':>25}")
print("="*50)

# Plot Greeks
print("\nGenerating Greeks plots for put option...")
plot_greeks(S, K, T, r, sigma)
print("Plots saved as 'put_greeks_analysis.png'")