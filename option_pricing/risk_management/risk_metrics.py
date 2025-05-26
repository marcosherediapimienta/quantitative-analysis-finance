import numpy as np
import pandas as pd
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
bs_call_mod = import_from_path('bs_call', os.path.join(base, 'black_scholes_model/european_options/Call_BS.py'))
bs_put_mod = import_from_path('bs_put', os.path.join(base, 'black_scholes_model/european_options/Put_BS.py'))
binomial_eur_mod = import_from_path('binomial_eur', os.path.join(base, 'binomial_model/european_options/european_binomial.py'))
binomial_ame_mod = import_from_path('binomial_ame', os.path.join(base, 'binomial_model/american_options/american_binomial.py'))
mc_eur_mod = import_from_path('mc_eur', os.path.join(base, 'monte_carlo/european_options/european_MC.py'))
mc_ame_mod = import_from_path('mc_ame', os.path.join(base, 'monte_carlo/american_options/american_MC.py'))
fd_call_mod = import_from_path('fd_call', os.path.join(base, 'finite_difference_method/european_options/call.py'))
fd_put_mod = import_from_path('fd_put', os.path.join(base, 'finite_difference_method/european_options/put.py'))

# --- UTILITY FUNCTIONS ---
def safe_download(ticker, period='1y', interval='1d'):
    try:
        data = yf.download(ticker, period=period, interval=interval)['Close']
        if (isinstance(data, pd.Series) and data.isnull().all()) or \
           (isinstance(data, pd.DataFrame) and data.isnull().all().all()) or \
           data.empty or len(data) == 0:
            print(f"Error: Could not download data for {ticker}.")
            return None
        return data
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None

def safe_float(val, default=np.nan):
    try:
        if hasattr(val, 'item'):
            return float(val.item())
        return float(val)
    except Exception:
        return default

def get_portfolio_spot(portfolio):
    return portfolio[0]['S0']

# --- PORTFOLIO DEFINITION ---
portfolio = [
    {'ticker': '^SPX','type': 'call','K': 5800,'T': 0.0876,'contracts': 10, 'style': 'european'},
    {'ticker': '^SPX','type': 'put','K': 5800,'T': 0.0876,'contracts': 10, 'style': 'european'},   
]

# --- GENERAL PRICING FUNCTION ---
def get_option_price(model, option_type, S, K, T, r, sigma, style='european'):
    try:
        if model == 'black_scholes':
            return bs_call_mod.black_scholes_call_price(S, K, T, r, sigma) if option_type == 'call' else bs_put_mod.black_scholes_put_price(S, K, T, r, sigma)
        elif model == 'binomial':
            if style == 'american':
                return binomial_ame_mod.binomial_american_option_price(S, K, T, r, sigma, N=1000, option_type=option_type)
            else:
                return binomial_eur_mod.binomial_european_option_price(S, K, T, r, sigma, N=1000, option_type=option_type)
        elif model == 'monte_carlo':
            if style == 'american':
                return mc_ame_mod.american_option_longstaff_schwartz(S, K, T, r, sigma, n_sim=10000, n_steps=50, option_type=option_type)
            else:
                return mc_eur_mod.monte_carlo_european_option(S, K, T, r, sigma, n_sim=10000, option_type=option_type)
        elif model == 'finite_difference':
            price = fd_call_mod.finite_difference_european_call(S, K, T, r, sigma) if option_type == 'call' else fd_put_mod.finite_difference_european_put(S, K, T, r, sigma)
            return price if np.isfinite(price) and abs(price) <= 1e6 else np.nan
        else:
            raise ValueError('Model not supported')
    except Exception as e:
        print(f"Error in get_option_price: {e}")
        return np.nan

# --- DOWNLOAD PRICES AND HISTORICAL VOLATILITY ---
for opt in portfolio:
    prices = safe_download(opt['ticker'])
    if prices is None:
        opt['S0'] = np.nan
        opt['sigma'] = np.nan
        continue
    S0 = safe_float(prices.iloc[-1])
    returns = np.log(prices / prices.shift(1)).dropna()
    sigma = safe_float(returns.std() * np.sqrt(252))
    opt['S0'] = S0
    opt['sigma'] = sigma

# --- IMPLIED VOLATILITY IMPUTATION ---
for opt in portfolio:
    try:
        # Buscar el strike más cercano si no hay coincidencia exacta
        ticker_obj = yf.Ticker(opt['ticker'])
        expirations = ticker_obj.options
        if not expirations:
            opt['implied_vol'] = None
            continue
        expiration = expirations[0]  # Por defecto, la primera expiración
        opt_chain = ticker_obj.option_chain(expiration)
        strikes = opt_chain.calls['strike'].values
        closest_idx = (np.abs(strikes - opt['K'])).argmin()
        K_yahoo = float(strikes[closest_idx])
        if opt['type'] == 'call':
            row = opt_chain.calls[opt_chain.calls['strike'] == K_yahoo]
        else:
            row = opt_chain.puts[opt_chain.puts['strike'] == K_yahoo]
        if not row.empty:
            market_price = float(row['lastPrice'].values[0])
            print(f"Market price for {opt['ticker']} {opt['type']} (K={K_yahoo}): {market_price}")
        else:
            user_input = input(f"Enter the market price of the option {opt['ticker']} {opt['type']} (K={opt['K']}, T={opt['T']}) to calculate implied volatility (leave blank if not available): ")
            if user_input.strip() != '':
                market_price = float(user_input)
            else:
                opt['implied_vol'] = None
                continue
        implied_vol = bs_call_mod.implied_volatility_newton(market_price, opt['S0'], opt['K'], opt['T'], 0.0421) if opt['type'] == 'call' else bs_put_mod.implied_volatility_newton(market_price, opt['S0'], opt['K'], opt['T'], 0.0421)
        if np.isnan(implied_vol) or implied_vol <= 0 or implied_vol > 3:
            opt['implied_vol'] = None
            print(f"Warning: Implied volatility not available or invalid for {opt['ticker']} {opt['type']}.")
        else:
            opt['implied_vol'] = implied_vol
            print(f"Implied volatility calculated: {implied_vol:.2%}")
    except Exception as e:
        opt['implied_vol'] = None
        print(f"Warning: Error calculating implied volatility: {e}")

# --- PORTFOLIO INFO ---
def print_portfolio_info(portfolio):
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
            price = get_option_price('black_scholes', opt['type'], opt['S0'], opt['K'], opt['T'], 0.0421, used_vol, style=opt.get('style', 'european'))
        except Exception as e:
            price = np.nan
            print(f"Warning: Error calculating theoretical price: {e}")
        print(f"  Theoretical price (Black-Scholes): {price:.2f}")
        total_market_value += opt['contracts'] * price
    print(f"Total theoretical portfolio value (Black-Scholes): {total_market_value:.2f}\n")

print_portfolio_info(portfolio)

# --- AGGREGATE GREEKS (DELTA, GAMMA, VEGA, THETA, RHO) ---
def black_scholes_greeks(option_type, S, K, T, r, sigma):
    from scipy.stats import norm
    if T == 0 or sigma == 0:
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

def print_portfolio_greeks(portfolio, r):
    total = {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
    for opt in portfolio:
        used_vol = opt.get('implied_vol', None) or opt['sigma']
        greeks = black_scholes_greeks(opt['type'], opt['S0'], opt['K'], opt['T'], r, used_vol)
        for g in total:
            total[g] += opt['contracts'] * greeks[g]
    print("\n--- PORTFOLIO GREEKS (Black-Scholes) ---")
    for g in total:
        print(f"Total {g.capitalize()}: {total[g]:.4f}")

def print_portfolio_greeks_binomial(portfolio, r):
    total = {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
    for opt in portfolio:
        used_vol = opt.get('implied_vol', None) or opt['sigma']
        style = opt.get('style', 'european')
        if style == 'american':
            greeks = binomial_ame_mod.binomial_greeks_american_option(opt['S0'], opt['K'], opt['T'], r, used_vol, N=100, option_type=opt['type'])
        else:
            greeks = binomial_eur_mod.binomial_greeks_european_option(opt['S0'], opt['K'], opt['T'], r, used_vol, N=100, option_type=opt['type'])
        for g in total:
            total[g] += opt['contracts'] * greeks[g]
    print("\n--- PORTFOLIO GREEKS (Binomial) ---")
    for g in total:
        print(f"Total {g.capitalize()}: {total[g]:.4f}")

r = 0.0421  # risk-free rate
print_portfolio_greeks(portfolio, r)
print_portfolio_greeks_binomial(portfolio, r)

HORIZON_VAR = 10/252 
N_SIMULATIONS = 1000

def simulate_prices(portfolio, horizon, n_sim, r):
    simulated = {}
    for opt in portfolio:
        S0 = opt['S0']
        sigma = opt['sigma']
        if np.isnan(S0) or np.isnan(sigma):
            simulated[opt['ticker']] = np.full(n_sim, np.nan)
            continue
        Z = np.random.standard_normal(n_sim)
        ST = S0 * np.exp((r - 0.5 * sigma ** 2) * horizon + sigma * np.sqrt(horizon) * Z)
        simulated[opt['ticker']] = ST
    return simulated

simulated_prices = simulate_prices(portfolio, HORIZON_VAR, N_SIMULATIONS, r)

# --- TOTAL DELTA ---
def black_scholes_delta(option_type, S, K, T, r, sigma):
    from scipy.stats import norm
    if T == 0 or sigma == 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def calc_total_delta(portfolio, r):
    return sum(
        opt['contracts'] * black_scholes_delta(
            opt['type'], opt['S0'], opt['K'], opt['T'], r, opt.get('implied_vol', None) or opt['sigma']
        )
        for opt in portfolio
    )

delta_total = calc_total_delta(portfolio, r)
print(f"\nTotal portfolio delta (Black-Scholes): {delta_total:.4f}")

# --- DELTA-HEDGED P&L SIMULATION ---
models = ['black_scholes', 'binomial', 'monte_carlo', 'finite_difference']
CONFIDENCE_LEVEL = 0.99

def calc_pnl_hedged(portfolio, models, simulated_prices, N_SIMULATIONS, HORIZON_VAR, r, delta_total):
    pnl_hedged = {model: [] for model in models}
    S0 = get_portfolio_spot(portfolio)
    for i in range(N_SIMULATIONS):
        for model in models:
            port_value = 0
            for opt in portfolio:
                S = simulated_prices[opt['ticker']][i]
                K = opt['K']
                T_future = max(opt['T'] - HORIZON_VAR, 0)
                sigma = opt['sigma']
                contracts = opt['contracts']
                style = opt.get('style', 'european')
                price = get_option_price(model, opt['type'], S, K, T_future, r, sigma, style=style)
                port_value += contracts * price
            spot_pnl = (simulated_prices[portfolio[0]['ticker']][i] - S0) * (-delta_total)
            pnl_hedged[model].append(port_value + spot_pnl)
    return pnl_hedged

pnl_hedged = calc_pnl_hedged(portfolio, models, simulated_prices, N_SIMULATIONS, HORIZON_VAR, r, delta_total)

V0_dict = {model: sum(opt['contracts'] * get_option_price(model, opt['type'], opt['S0'], opt['K'], opt['T'], r, opt.get('implied_vol', None) or opt['sigma'], style=opt.get('style', 'european')) for opt in portfolio) for model in models}

results = {model: [] for model in models}
for i in range(N_SIMULATIONS):
    for model in models:
        port_value = 0
        for opt in portfolio:
            S = simulated_prices[opt['ticker']][i]
            K = opt['K']
            T_future = max(opt['T'] - HORIZON_VAR, 0)
            sigma = opt['sigma']
            contracts = opt['contracts']
            style = opt.get('style', 'european')
            price = get_option_price(model, opt['type'], S, K, T_future, r, sigma, style=style)
            port_value += contracts * price
        results[model].append(port_value)

# --- VAR/ES PLOT ---
def plot_var_es(pnl_dict, V0_dict, CONFIDENCE_LEVEL, filename, title_prefix, color):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    plot_idx = 0
    for model in models:
        pnl = np.array(pnl_dict[model]) - V0_dict[model]
        pnl = pnl[np.isfinite(pnl)]
        if len(pnl) == 0:
            print(f"Warning: Cannot plot model {model} (no valid data for {title_prefix.lower()}).")
            continue
        var = np.percentile(pnl, (1-CONFIDENCE_LEVEL)*100)
        es = pnl[pnl <= var].mean()
        print(f"Model: {model}")
        print(f"  Value at Risk (VaR) {title_prefix} at {CONFIDENCE_LEVEL*100:.1f}%: {var:.2f}")
        print(f"  Expected Shortfall (ES) {title_prefix} at {CONFIDENCE_LEVEL*100:.1f}%: {es:.2f}\n")
        axs[plot_idx].hist(pnl, bins=40, color=color, edgecolor='k', alpha=0.7)
        axs[plot_idx].axvline(var, color='red', linestyle='--', label=f'VaR ({var:.2f})')
        axs[plot_idx].axvline(es, color='orange', linestyle='--', label=f'ES ({es:.2f})')
        axs[plot_idx].set_title(f"Model: {model} ({title_prefix})")
        axs[plot_idx].set_xlabel('Simulated P&L')
        axs[plot_idx].set_ylabel('Frequency')
        axs[plot_idx].legend()
        plot_idx += 1
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f'{title_prefix} plot saved as {filename}')

plot_var_es(results, V0_dict, CONFIDENCE_LEVEL, 'risk_metrics_results.png', 'plain', 'skyblue')
plot_var_es(pnl_hedged, V0_dict, CONFIDENCE_LEVEL, 'risk_metrics_results_delta_hedge.png', 'delta-hedge', 'lightcoral')

# --- SIMULATED DISTRIBUTIONS PLOT ---
def plot_simulated_distributions(simulated_prices, prefix):
    fig, axs = plt.subplots(1, len(simulated_prices), figsize=(7*len(simulated_prices), 5))
    if len(simulated_prices) == 1:
        axs = [axs]
    for idx, (ticker, prices_sim) in enumerate(simulated_prices.items()):
        axs[idx].hist(prices_sim, bins=50, color='lightgreen', edgecolor='k', alpha=0.7)
        axs[idx].set_title(f"Distribution of simulated prices: {ticker}")
        axs[idx].set_xlabel('Simulated price')
        axs[idx].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{prefix}_simulated_prices_distribution.png')
    plt.close(fig)
    print(f'Simulated prices distribution plot saved as {prefix}_simulated_prices_distribution.png')

    fig, axs = plt.subplots(1, len(simulated_prices), figsize=(7*len(simulated_prices), 5))
    if len(simulated_prices) == 1:
        axs = [axs]
    for idx, (ticker, prices_sim) in enumerate(simulated_prices.items()):
        log_prices = np.log(prices_sim)
        axs[idx].hist(log_prices, bins=50, color='orange', edgecolor='k', alpha=0.7)
        axs[idx].set_title(f"Distribution of simulated log-prices: {ticker}")
        axs[idx].set_xlabel('log(Simulated price)')
        axs[idx].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{prefix}_simulated_logprices_distribution.png')
    plt.close(fig)
    print(f'Simulated log-prices distribution plot saved as {prefix}_simulated_logprices_distribution.png')

plot_simulated_distributions(simulated_prices, 'risk_metrics')
m
# --- STRESS TESTING / SCENARIOS ---
def stress_test_scenarios(portfolio, models, V0_dict, shocks, r):
    print("\n--- STRESS TESTING / SCENARIOS ---")
    def calc_total_delta(portfolio, r):
        return sum(
            opt['contracts'] * black_scholes_delta(
                opt['type'], opt['S0'], opt['K'], opt['T'], r, opt.get('implied_vol', None) or opt['sigma']
            )
            for opt in portfolio
        )
    delta_total = calc_total_delta(portfolio, r)
    S0 = portfolio[0]['S0']
    for shock in shocks:
        print(f"\nShock: {shock['desc']}")
        for model in models:
            stressed_value = 0
            for opt in portfolio:
                S = opt['S0'] * shock.get('spot_mult', 1.0)
                sigma_stress = (
                    (opt['implied_vol'] * shock['vol_mult']) if ('vol_mult' in shock and opt.get('implied_vol') is not None)
                    else (opt['sigma'] * shock['vol_mult']) if 'vol_mult' in shock else opt.get('implied_vol', None) or opt['sigma']
                )
                T = opt['T']
                r_stress = r + shock.get('r_shift', 0.0)
                style = opt.get('style', 'european')
                price = get_option_price(model, opt['type'], S, opt['K'], T, r_stress, sigma_stress, style=style)
                stressed_value += opt['contracts'] * price
            pnl = stressed_value - V0_dict[model]
            S_stress = S0 * shock.get('spot_mult', 1.0)
            spot_pnl = (S_stress - S0) * (-delta_total)
            pnl_hedged = pnl + spot_pnl
            print(f"  Model: {model} | Stressed P&L: {pnl:.2f} | Stressed P&L (delta-hedge): {pnl_hedged:.2f}")

shocks = [
    {'desc': 'Spot -10%', 'spot_mult': 0.9},
    {'desc': 'Spot -20%', 'spot_mult': 0.8},
    {'desc': 'Vol +50%', 'vol_mult': 1.5},
    {'desc': 'Spot -10% and Vol +50%', 'spot_mult': 0.9, 'vol_mult': 1.5},
    {'desc': 'r +100bps', 'r_shift': 0.01},
    {'desc': 'r -100bps', 'r_shift': -0.01},
]
stress_test_scenarios(portfolio, models, V0_dict, shocks, r)
