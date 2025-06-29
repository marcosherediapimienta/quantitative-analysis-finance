import numpy as np
import yfinance as yf
from datetime import datetime
import scipy.stats as stats
from scipy.optimize import brentq
import matplotlib.pyplot as plt


def binomial_american_option_price(S, K, T, r, sigma, N=1000, option_type='call'):
    """
    Price an American option using the Cox-Ross-Rubinstein binomial tree.
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
    
    # Precompute terminal payoffs
    ST = S * (u ** np.arange(N+1)) * (d ** (N - np.arange(N+1)))
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    
    # Backward induction with early exercise check
    for i in range(N-1, -1, -1):
        ST = S * (u ** np.arange(i+1)) * (d ** (i - np.arange(i+1)))
        continuation_value = discount * (p * payoff[1:i+2] + (1-p) * payoff[0:i+1])
        
        if option_type == 'call':
            exercise_value = np.maximum(ST - K, 0)
        else:
            exercise_value = np.maximum(K - ST, 0)
        
        payoff = np.maximum(continuation_value, exercise_value)
    
    return payoff[0]


def binomial_greeks_american_option(S, K, T, r, sigma, N=1000, option_type='call', h=None):
    """
    Estimate Delta, Gamma, Vega, Theta, Rho for an American option using the binomial model.
    Uses finite differences (bumping) on the binomial price.
    """
    if h is None:
        h = max(0.01 * S, 0.01)  # 1% del spot, pero nunca menor que 0.01
    # Delta
    price_up = binomial_american_option_price(S + h, K, T, r, sigma, N, option_type)
    price_down = binomial_american_option_price(S - h, K, T, r, sigma, N, option_type)
    price = binomial_american_option_price(S, K, T, r, sigma, N, option_type)
    delta = (price_up - price_down) / (2 * h)
    gamma = (price_up - 2 * price + price_down) / (h ** 2)
    # Vega
    vega_bump = 0.01
    price_vega_up = binomial_american_option_price(S, K, T, r, sigma + vega_bump, N, option_type)
    price_vega_down = binomial_american_option_price(S, K, T, r, sigma - vega_bump, N, option_type)
    vega = (price_vega_up - price_vega_down) / (2 * vega_bump)
    # Theta (backward difference)
    theta_bump = 0.01
    price_theta = binomial_american_option_price(S, K, T - theta_bump, r, sigma, N, option_type)
    theta = (price_theta - price) / theta_bump
    # Rho
    rho_bump = 0.01
    price_rho_up = binomial_american_option_price(S, K, T, r + rho_bump, sigma, N, option_type)
    price_rho_down = binomial_american_option_price(S, K, T, r - rho_bump, sigma, N, option_type)
    rho = (price_rho_up - price_rho_down) / (2 * rho_bump)
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

def plot_binomial_tree_summary(S, u, d, N, prefix='binomial_tree_summary'):
    """
    For large N, save the normalized (probability) distribution of asset prices at maturity and the evolution of min/max/mean price at each step as PNG files.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    prices_by_step = []
    for i in range(N+1):
        prices = S * (u ** np.arange(i+1)) * (d ** (i - np.arange(i+1)))
        prices_by_step.append(prices)
    mins = [np.min(p) for p in prices_by_step]
    maxs = [np.max(p) for p in prices_by_step]
    means = [np.mean(p) for p in prices_by_step]
    plt.figure(figsize=(10,5))
    plt.plot(mins, label='Min price')
    plt.plot(maxs, label='Max price')
    plt.plot(means, label='Mean price')
    plt.xlabel('Step')
    plt.ylabel('Asset Price')
    plt.title('Evolution of Asset Price in Binomial Tree')
    plt.legend()
    plt.grid(True)
    fname1 = f'visualizations/{prefix}_evolution.png'
    plt.savefig(fname1, bbox_inches='tight')
    print(f"Evolution plot saved to {fname1}")
    plt.close()
    # Normalized histogram (probability)
    plt.figure(figsize=(10,5))
    plt.hist(prices_by_step[-1], bins=50, color='skyblue', edgecolor='k', density=True)
    plt.xlabel('Asset Price at Maturity')
    plt.ylabel('Probability')
    plt.title(f'Asset Price Probability Distribution at Maturity (N={N})')
    plt.grid(True)
    fname2 = f'visualizations/{prefix}_maturity_hist.png'
    plt.savefig(fname2, bbox_inches='tight')
    print(f"Maturity probability distribution plot saved to {fname2}")
    plt.close()

def get_historical_volatility(ticker, window=252):
    """
    Calcula la volatilidad histórica anualizada usando precios de cierre diarios de Yahoo Finance.
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
    print("\nBinomial Model American Option Pricing (Cox-Ross-Rubinstein)")
    option_type = input("Option type ('call' or 'put') [call]: ").strip().lower() or 'call'
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
                    except ValueError:
                        print("Invalid input. Please enter a number.")
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
                    except ValueError:
                        print("Invalid input. Please enter a number.")
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
    if market_price is not None:
        # Implied volatility
        if option_type == 'call':
            iv = implied_volatility_option(market_price, S, K, T, r, option_type='call')
        else:
            vi_put = valor_intrinseco_put_desc(S, K, T, r)
            if market_price < vi_put:
                print(f"[WARNING] The market price of the put (${market_price:.2f}) is less than the discounted intrinsic value (${vi_put:.2f}). It is not possible to find a consistent implied volatility. Using historical volatility as fallback.")
                iv = None
            else:
                iv = implied_volatility_option(market_price, S, K, T, r, option_type='put')
        if iv is None:
            try:
                iv = get_historical_volatility(ticker, window=252)
                print(f"No implied volatility found, using 1y historical volatility as fallback: {iv:.2%}")
            except Exception as e:
                print(f'Could not fetch historical volatility: {e}. Using 20% as fallback.')
                iv = 0.2
        sigma = iv
    else:
        print("No market price available, using historical volatility.")
        sigma = get_historical_volatility(ticker)
        print(f"Historical volatility (used): {sigma*100:.2f}%")
    N = get_input("Enter number of steps (e.g., 100)", 100, int, lambda x: x > 0)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    price = binomial_american_option_price(S, K, T, r, sigma, N, option_type)
    greeks = binomial_greeks_american_option(S, K, T, r, sigma, N, option_type)
    print("\n" + "="*50)
    print(f"American {option_type.capitalize()} Option Pricing (Binomial Model)")
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
    print("\nGreeks (Binomial estimates):")
    for greek, value in greeks.items():
        print(f"  {greek.capitalize():<6}: {value:.4f}")
    print("="*50)
    plot_tree = input("Plot binomial tree summary? (y/n) [n]: ").strip().lower() or 'n'
    if plot_tree == 'y':
        plot_binomial_tree_summary(S, u, d, N)