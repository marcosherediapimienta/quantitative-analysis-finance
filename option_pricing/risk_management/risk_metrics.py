import numpy as np
import yfinance as yf
import os
import importlib.util
import matplotlib.pyplot as plt

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
bs_call_mod = import_from_path('bs_call', os.path.join(base, 'black_scholes_model/european_options/call_implied_volatility.py'))
bs_put_mod = import_from_path('bs_put', os.path.join(base, 'black_scholes_model/european_options/put_implied_volatility.py'))
binomial_mod = import_from_path('binomial', os.path.join(base, 'binomial_model/european_options/binomial.py'))
mc_call_mod = import_from_path('mc_call', os.path.join(base, 'monte-carlo/european_options/call.py'))
mc_put_mod = import_from_path('mc_put', os.path.join(base, 'monte-carlo/european_options/put.py'))
fd_call_mod = import_from_path('fd_call', os.path.join(base, 'finite_difference_method/european_options/call.py'))
fd_put_mod = import_from_path('fd_put', os.path.join(base, 'finite_difference_method/european_options/put.py'))

# --- PORTFOLIO DEFINITION (you can modify it manually) ---
# Note: T is the time to maturity expressed in years (using 365 calendar days).
portfolio = [
    {'ticker': '^SPX','type': 'call','K': 7400,'T': 0.0712,'contracts': 15},
    {'ticker': '^SPX','type': 'put','K': 2600,'T': 0.0712,'contracts': 10},   
]

# --- GENERAL PRICING FUNCTION ---
def get_option_price(model, option_type, S, K, T, r, sigma):
    try:
        if model == 'black_scholes':
            if option_type == 'call':
                return bs_call_mod.black_scholes_call_price(S, K, T, r, sigma)
            else:
                return bs_put_mod.black_scholes_put_price(S, K, T, r, sigma)
        elif model == 'binomial':
            return binomial_mod.binomial_european_option_price(S, K, T, r, sigma, N=100, option_type=option_type)
        elif model == 'monte_carlo':
            if option_type == 'call':
                return mc_call_mod.monte_carlo_european_call(S, K, T, r, sigma, n_simulations=10000)
            else:
                return mc_put_mod.monte_carlo_european_put(S, K, T, r, sigma, n_simulations=10000)
        elif model == 'finite_difference':
            if option_type == 'call':
                price = fd_call_mod.finite_difference_european_call(S, K, T, r, sigma)
            else:
                price = fd_put_mod.finite_difference_european_put(S, K, T, r, sigma)
            if not np.isfinite(price) or abs(price) > 1e6:
                return np.nan
            return price
        else:
            raise ValueError('Model not supported')
    except Exception:
        return np.nan

# --- STEP 1: Download prices and historical volatility ---
for opt in portfolio:
    prices = yf.download(opt['ticker'], period='1y', interval='1d')['Close']
    S0 = prices.iloc[-1]
    if hasattr(S0, 'item'):
        S0 = float(S0.item())
    else:
        S0 = float(S0)
    returns = np.log(prices / prices.shift(1)).dropna()
    sigma_val = returns.std() * np.sqrt(252)
    if hasattr(sigma_val, 'item'):
        sigma = float(sigma_val.item())
    else:
        sigma = float(sigma_val)
    opt['S0'] = S0
    opt['sigma'] = sigma

# --- STEP 2: Calculate implied volatility if market price is available ---
for opt in portfolio:
    try:
        user_input = input(f"Enter the market price of the option {opt['ticker']} {opt['type']} (K={opt['K']}, T={opt['T']}) to calculate implied volatility (leave blank if not available): ")
        if user_input.strip() != '':
            market_price = float(user_input)
            if opt['type'] == 'call':
                implied_vol = bs_call_mod.implied_volatility_newton(market_price, opt['S0'], opt['K'], opt['T'], 0.0421)
            else:
                implied_vol = bs_put_mod.implied_volatility_newton(market_price, opt['S0'], opt['K'], opt['T'], 0.0421)
            if np.isnan(implied_vol) or implied_vol <= 0 or implied_vol > 3:
                opt['implied_vol'] = None
                print(f"  Implied volatility not available or invalid for {opt['ticker']} {opt['type']}.")
            else:
                opt['implied_vol'] = implied_vol
                print(f"  Implied volatility calculated: {implied_vol:.2%}")
        else:
            opt['implied_vol'] = None
    except Exception:
        opt['implied_vol'] = None

# --- STEP 3: Show portfolio information ---
print("\n--- PORTFOLIO INFORMATION ---")
total_market_value = 0
for opt in portfolio:
    print(f"{opt['ticker']} | Type: {opt['type']} | Strike: {opt['K']} | T: {opt['T']} | Contracts: {opt['contracts']}")
    print(f"  Spot price: {opt['S0']:.2f} | Annualized volatility: {opt['sigma']:.2%}")
    if opt.get('implied_vol') is not None:
        print(f"  Implied volatility: {opt['implied_vol']:.2%}")
        used_vol = opt['implied_vol']
    else:
        print(f"  Implied volatility not available, using historical: {opt['sigma']:.2%}")
        used_vol = opt['sigma']
    try:
        price = get_option_price('black_scholes', opt['type'], opt['S0'], opt['K'], opt['T'], 0.0421, used_vol)
    except:
        price = np.nan
    print(f"  Theoretical price (Black-Scholes): {price:.2f}")
    total_market_value += opt['contracts'] * price
    print()
print(f"Total theoretical portfolio value (Black-Scholes): {total_market_value:.2f}\n")

# --- STEP 4: Simulate future price scenarios (GBM) ---
# VAR/ES HORIZON (in years): 1 trading day = 1/252, 1 week = 5/252, etc.
HORIZON_VAR = 15/252 
N_SIMULATIONS = 1000
r = 0.0421  # risk-free rate
simulated_prices = {}
for opt in portfolio:
    S0 = opt['S0']
    sigma = opt['sigma']
    # Simulate the price at 1 day, not until maturity
    Z = np.random.standard_normal(N_SIMULATIONS)
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * HORIZON_VAR + sigma * np.sqrt(HORIZON_VAR) * Z)
    simulated_prices[opt['ticker']] = ST

# --- STEP 5: Calculate portfolio value in each scenario and model ---
models = ['black_scholes', 'binomial', 'monte_carlo', 'finite_difference']
CONFIDENCE_LEVEL = 0.99
results = {model: [] for model in models}
for i in range(N_SIMULATIONS):
    for model in models:
        port_value = 0
        for opt in portfolio:
            S = simulated_prices[opt['ticker']][i]
            K = opt['K']
            # Remaining time to maturity in the future scenario
            T_future = max(opt['T'] - HORIZON_VAR, 0)
            sigma = opt['sigma']
            contracts = opt['contracts']
            price = get_option_price(model, opt['type'], S, K, T_future, r, sigma)
            port_value += contracts * price
        results[model].append(port_value)

# --- STEP 6: Calculate VaR and ES for each model ---
# Calculate the current portfolio value for each model (V0) using the real maturity
V0_dict = {}
for model in models:
    V0 = 0
    for opt in portfolio:
        used_vol = opt.get('implied_vol', None) or opt['sigma']
        price = get_option_price(model, opt['type'], opt['S0'], opt['K'], opt['T'], r, used_vol)
        V0 += opt['contracts'] * price
    V0_dict[model] = V0

for model in models:
    pnl = np.array(results[model]) - V0_dict[model]
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) == 0:
        print(f"Model: {model}")
        print("  No valid results for this model (possible numerical error or unsupported parameters).\n")
        continue
    var = np.percentile(pnl, (1-CONFIDENCE_LEVEL)*100)
    es = pnl[pnl <= var].mean()
    print(f"Model: {model}")
    print(f"  Value at Risk (VaR) at {CONFIDENCE_LEVEL*100:.1f}%: {var:.2f}")
    print(f"  Expected Shortfall (ES) at {CONFIDENCE_LEVEL*100:.1f}%: {es:.2f}\n")

# --- STEP 7: Visualization of results ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()
plot_idx = 0
for model in models:
    pnl = np.array(results[model]) - V0_dict[model]
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) == 0:
        print(f"Cannot plot model {model} (no valid data).")
        continue
    var = np.percentile(pnl, (1-CONFIDENCE_LEVEL)*100)
    es = pnl[pnl <= var].mean()
    axs[plot_idx].hist(pnl, bins=40, color='skyblue', edgecolor='k', alpha=0.7)
    axs[plot_idx].axvline(var, color='red', linestyle='--', label=f'VaR ({var:.2f})')
    axs[plot_idx].axvline(es, color='orange', linestyle='--', label=f'ES ({es:.2f})')
    axs[plot_idx].set_title(f"Model: {model}")
    axs[plot_idx].set_xlabel('Simulated P&L')
    axs[plot_idx].set_ylabel('Frequency')
    axs[plot_idx].legend()
    plot_idx += 1
plt.tight_layout()
plt.savefig('risk_metrics_results.png')
print('Plot saved as risk_metrics_results.png')

# --- EXTRA VISUALIZATION: Distribution of simulated prices for each underlying ---
fig2, axs2 = plt.subplots(1, len(simulated_prices), figsize=(7*len(simulated_prices), 5))
if len(simulated_prices) == 1:
    axs2 = [axs2]
for idx, (ticker, prices_sim) in enumerate(simulated_prices.items()):
    axs2[idx].hist(prices_sim, bins=50, color='lightgreen', edgecolor='k', alpha=0.7)
    axs2[idx].set_title(f"Distribution of simulated prices: {ticker}")
    axs2[idx].set_xlabel('Simulated price')
    axs2[idx].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('simulated_prices_distribution.png')
print('Simulated prices distribution plot saved as simulated_prices_distribution.png')

# --- EXTRA VISUALIZATION: Log-price distribution ---
fig3, axs3 = plt.subplots(1, len(simulated_prices), figsize=(7*len(simulated_prices), 5))
if len(simulated_prices) == 1:
    axs3 = [axs3]
for idx, (ticker, prices_sim) in enumerate(simulated_prices.items()):
    log_prices = np.log(prices_sim)
    axs3[idx].hist(log_prices, bins=50, color='orange', edgecolor='k', alpha=0.7)
    axs3[idx].set_title(f"Distribution of simulated log-prices: {ticker}")
    axs3[idx].set_xlabel('log(Simulated price)')
    axs3[idx].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('simulated_logprices_distribution.png')
print('Simulated log-prices distribution plot saved as simulated_logprices_distribution.png')

# --- CALCULATION OF TOTAL PORTFOLIO DELTA (Black-Scholes) ---
def black_scholes_delta(option_type, S, K, T, r, sigma):
    from scipy.stats import norm
    if T == 0 or sigma == 0:
        # Option expired or zero volatility
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

delta_total = 0
for opt in portfolio:
    used_vol = opt.get('implied_vol', None) or opt['sigma']
    delta = black_scholes_delta(opt['type'], opt['S0'], opt['K'], opt['T'], r, used_vol)
    delta_total += opt['contracts'] * delta
print(f"\nTotal portfolio delta (Black-Scholes): {delta_total:.4f}")

# --- SIMULATION OF DELTA-HEDGED P&L ---
pnl_hedged = {model: [] for model in models}
S0 = portfolio[0]['S0']  # Assume a single underlying for the hedge
for i in range(N_SIMULATIONS):
    for model in models:
        port_value = 0
        for opt in portfolio:
            S = simulated_prices[opt['ticker']][i]
            K = opt['K']
            T_future = max(opt['T'] - HORIZON_VAR, 0)
            sigma = opt['sigma']
            contracts = opt['contracts']
            price = get_option_price(model, opt['type'], S, K, T_future, r, sigma)
            port_value += contracts * price
        # P&L of the delta-hedged position: subtract the spot change times the initial delta
        spot_pnl = (simulated_prices[portfolio[0]['ticker']][i] - S0) * (-delta_total)
        pnl_hedged[model].append(port_value + spot_pnl)

# --- CALCULATION AND VISUALIZATION OF DELTA-HEDGED VaR/ES ---
print("\n--- VaR and ES for the delta-hedged portfolio (neutralized to initial spot) ---")
fig4, axs4 = plt.subplots(2, 2, figsize=(14, 10))
axs4 = axs4.flatten()
plot_idx = 0
for model in models:
    pnl = np.array(pnl_hedged[model]) - V0_dict[model]
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) == 0:
        print(f"Cannot plot model {model} (no valid data for delta-hedge).")
        continue
    var = np.percentile(pnl, (1-CONFIDENCE_LEVEL)*100)
    es = pnl[pnl <= var].mean()
    print(f"Model: {model}")
    print(f"  Value at Risk (VaR) delta-hedged at {CONFIDENCE_LEVEL*100:.1f}%: {var:.2f}")
    print(f"  Expected Shortfall (ES) delta-hedged at {CONFIDENCE_LEVEL*100:.1f}%: {es:.2f}\n")
    axs4[plot_idx].hist(pnl, bins=40, color='lightcoral', edgecolor='k', alpha=0.7)
    axs4[plot_idx].axvline(var, color='red', linestyle='--', label=f'VaR ({var:.2f})')
    axs4[plot_idx].axvline(es, color='orange', linestyle='--', label=f'ES ({es:.2f})')
    axs4[plot_idx].set_title(f"Model: {model} (delta-hedge)")
    axs4[plot_idx].set_xlabel('Simulated P&L (delta-hedge)')
    axs4[plot_idx].set_ylabel('Frequency')
    axs4[plot_idx].legend()
    plot_idx += 1
plt.tight_layout()
plt.savefig('risk_metrics_results_delta_hedge.png')
print('Delta-hedge plot saved as risk_metrics_results_delta_hedge.png')
