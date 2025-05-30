import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
np.random.seed(42)
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib.pyplot as plt
import copy
import pandas as pd

from option_pricing.monte_carlo.american_options.american_MC import american_option_longstaff_schwartz, american_mc_greeks
from option_pricing.monte_carlo.european_options.european_MC import monte_carlo_european_option, mc_greeks
from option_pricing.binomial_model.american_options.american_binomial import get_historical_volatility

# Estructura de una opción en la cartera:
# {
#   'type': 'call' o 'put',
#   'style': 'european' o 'american',
#   'S': spot,
#   'K': strike,
#   'T': tiempo a vencimiento (años),
#   'r': tasa libre de riesgo,
#   'qty': cantidad (positiva: long, negativa: short),
#   'market_price': precio de mercado (para IV)
# }

def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

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
    def objective(sigma_):
        if option_type == 'call':
            return black_scholes_call_price(S, K, T, r, sigma_) - market_price
        else:
            return black_scholes_put_price(S, K, T, r, sigma_) - market_price
    try:
        return brentq(objective, 1e-6, 5.0, xtol=tol, maxiter=max_iter)
    except Exception:
        return None

def price_option_mc(opt, n_sim=10000, n_steps=50):
    iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
    if iv is None:
        ticker = opt.get('ticker', None)
        if ticker is not None:
            iv = get_historical_volatility(ticker)
        else:
            iv = 0.2
    if opt['style'] == 'american':
        price = american_option_longstaff_schwartz(opt['S'], opt['K'], opt['T'], opt['r'], iv, n_sim, n_steps, opt['type'])
    else:
        price = monte_carlo_european_option(opt['S'], opt['K'], opt['T'], opt['r'], iv, n_sim, opt['type'])
    return price

def option_greeks_mc(opt, n_sim=10000, n_steps=50):
    np.random.seed(42)  # Set seed for reproducibility
    iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
    if iv is None:
        ticker = opt.get('ticker', None)
        if ticker is not None:
            iv = get_historical_volatility(ticker)
        else:
            iv = 0.2
    if opt['style'] == 'american':
        greeks = american_mc_greeks(opt['S'], opt['K'], opt['T'], opt['r'], iv, n_sim, n_steps, opt['type'])
    else:
        greeks = mc_greeks(opt['S'], opt['K'], opt['T'], opt['r'], iv, n_sim, opt['type'])
    return greeks

def portfolio_greeks_mc(portfolio, n_sim=10000, n_steps=50):
    total = {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
    for opt in portfolio:
        greeks = option_greeks_mc(opt, n_sim, n_steps)
        for g in total:
            total[g] += greeks[g] * opt['qty']
    return total

def simulate_portfolio_mc_pricing(portfolio, n_sims=1000, n_steps=50, horizon=None, vol_shock_sigma=0.1, rho=-0.5):
    np.random.seed(42)  # Set seed for reproducibility
    base_val = sum(price_option_mc(opt, n_sim=1000, n_steps=n_steps) * opt['qty'] for opt in portfolio)
    pnl = []
    shocks_dict = {}
    for opt in portfolio:
        key = opt.get('ticker', opt['S'])
        shocks_dict.setdefault(key, [])
    for i in range(n_sims):
        shocked_portfolio = []
        Zs = {}
        sigma_shocks = []
        for opt in portfolio:
            iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
            if iv is None:
                ticker = opt.get('ticker', None)
                if ticker is not None:
                    iv = get_historical_volatility(ticker)
                else:
                    iv = 0.2
            T_sim = horizon if horizon is not None else opt['T']
            # Correlated shocks
            Z1, Z2 = np.random.normal(size=2)
            Z_spot = Z1
            Z_vol = rho * Z1 + np.sqrt(1 - rho**2) * Z2
            Zs[key] = Z_spot
            shocks_dict[key].append(Z_spot)
            # Simular shock de volatilidad lognormal correlacionado
            vol_shock = np.random.lognormal(mean=0, sigma=vol_shock_sigma) * np.exp(Z_vol * vol_shock_sigma)
            sigma_shocks.append(iv * vol_shock)
            S_T = opt['S'] * np.exp((opt['r'] - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z_spot)
            shocked_opt = opt.copy()
            shocked_opt['S'] = S_T
            shocked_portfolio.append(shocked_opt)
        shocked_val = sum(price_option_mc(opt, n_sim=500, n_steps=n_steps) * opt['qty'] for opt, sigma in zip(shocked_portfolio, sigma_shocks))
        pnl.append(shocked_val - base_val)
    return {'pnl': np.array(pnl), 'shocks': shocks_dict}

def var_es(pnl, alpha=0.01):
    pnl_sorted = np.sort(pnl)
    var = -np.percentile(pnl_sorted, alpha*100)
    es = -pnl_sorted[pnl_sorted <= -var].mean()
    return var, es

def gamma_bs(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def run_sensitivity_analysis_mc(portfolio, N, n_sim_sens, vis_dir, horizon):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import copy
    # Estrategias de hedge
    from .mc_portfolio_analysis import portfolio_greeks_mc, price_option_mc
    n_sim_greeks = 100000
    hedge_strategies = [
        ('Original', portfolio),
        ('Delta Hedge', portfolio + [{
            'type': 'call', 'style': 'european', 'S': portfolio[0]['S'], 'K': portfolio[0]['K'], 'T': portfolio[0]['T'], 'r': portfolio[0]['r'], 'qty': -portfolio_greeks_mc(portfolio, n_sim=n_sim_greeks, n_steps=N)['delta'], 'market_price': portfolio[0]['market_price']
        }]),
        # Gamma y Vega hedge requieren lógica previa, pero para automatización rápida usamos las carteras ya generadas en la llamada principal
    ]
    # Puedes agregar Delta-Gamma y Vega Hedge si tienes la lógica en la app
    # 1. Sensibilidad Spot (±10%)
    spot_base = portfolio[0]['S']
    spot_range = np.linspace(spot_base*0.9, spot_base*1.1, 21)
    spot_idx_low = 0
    spot_idx_base = 10
    spot_idx_high = 20
    plt.figure(figsize=(12,7))
    spot_rows = []
    for name, port in hedge_strategies:
        values = []
        for S in spot_range:
            port_mod = [opt.copy() for opt in port]
            for opt in port_mod:
                opt['S'] = S
            values.append(sum(price_option_mc(opt, n_sim=n_sim_sens, n_steps=N) * opt['qty'] for opt in port_mod))
        plt.plot(spot_range, values, label=name)
        base = values[spot_idx_base]
        low = values[spot_idx_low]
        high = values[spot_idx_high]
        dlow = low - base
        dhigh = high - base
        spot_rows.append({
            'Strategy': name,
            'Base': base,
            '-10%': low,
            '+10%': high,
            'Δ-10%': dlow,
            'Δ+10%': dhigh
        })
    df_spot = pd.DataFrame(spot_rows)
    df_spot.to_csv(os.path.join(vis_dir, 'sensitivity_spot_mc.csv'), index=False)
    plt.xlabel('Spot')
    plt.ylabel('Portfolio Value')
    plt.title('Sensitivity to Spot - All Strategies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'sensitivity_spot_mc.png'))
    plt.close()
    # 2. Sensibilidad tipo de interés (±1%)
    r_base = portfolio[0]['r']
    r_range = np.linspace(r_base-0.01, r_base+0.01, 21)
    r_idx_low = 0
    r_idx_base = 10
    r_idx_high = 20
    plt.figure(figsize=(12,7))
    r_rows = []
    for name, port in hedge_strategies:
        values = []
        for r in r_range:
            port_mod = [opt.copy() for opt in port]
            for opt in port_mod:
                opt['r'] = r
            values.append(sum(price_option_mc(opt, n_sim=n_sim_sens, n_steps=N) * opt['qty'] for opt in port_mod))
        plt.plot(r_range, values, label=name)
        base = values[r_idx_base]
        low = values[r_idx_low]
        high = values[r_idx_high]
        dlow = low - base
        dhigh = high - base
        r_rows.append({
            'Strategy': name,
            'Base': base,
            '-1%': low,
            '+1%': high,
            'Δ-1%': dlow,
            'Δ+1%': dhigh
        })
    df_r = pd.DataFrame(r_rows)
    df_r.to_csv(os.path.join(vis_dir, 'sensitivity_r_mc.csv'), index=False)
    plt.xlabel('Risk-free rate (r)')
    plt.ylabel('Portfolio Value')
    plt.title('Sensitivity to r - All Strategies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'sensitivity_r_mc.png'))
    plt.close()
    # 3. Sensibilidad volatilidad (±20%)
    plt.figure(figsize=(12,7))
    vol_base = 1.0
    vol_range = np.linspace(0.8, 1.2, 21)
    vol_idx_low = 0
    vol_idx_base = 10
    vol_idx_high = 20
    vol_rows = []
    for name, port in hedge_strategies:
        values = []
        for vol_mult in vol_range:
            port_mod = [opt.copy() for opt in port]
            for opt in port_mod:
                iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                if iv is None:
                    iv = 0.2
                opt['market_price'] = black_scholes_call_price(opt['S'], opt['K'], opt['T'], opt['r'], iv*vol_mult) if opt['type']=='call' else black_scholes_put_price(opt['S'], opt['K'], opt['T'], opt['r'], iv*vol_mult)
            values.append(sum(price_option_mc(opt, n_sim=n_sim_sens, n_steps=N) * opt['qty'] for opt in port_mod))
        plt.plot(vol_range, values, label=name)
        base = values[vol_idx_base]
        low = values[vol_idx_low]
        high = values[vol_idx_high]
        dlow = low - base
        dhigh = high - base
        vol_rows.append({
            'Strategy': name,
            'Base': base,
            '-20%': low,
            '+20%': high,
            'Δ-20%': dlow,
            'Δ+20%': dhigh
        })
    df_vol = pd.DataFrame(vol_rows)
    df_vol.to_csv(os.path.join(vis_dir, 'sensitivity_vol_mc.csv'), index=False)
    plt.xlabel('Volatility multiplier')
    plt.ylabel('Portfolio Value')
    plt.title('Sensitivity to Volatility - All Strategies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'sensitivity_vol_mc.png'))
    plt.close()

if __name__ == "__main__":
    np.random.seed(42)  # Set seed for reproducibility
    portfolio = [
        {'type': 'call', 'style': 'european', 'S': 5912.17, 'K': 5915, 'T': 0.0849, 'r': 0.0421, 'qty': -10, 'market_price': 111.93},
        {'type': 'put',  'style': 'american', 'S': 5912.17, 'K': 5910, 'T': 0.0849, 'r': 0.0421, 'qty': -5,  'market_price': 106.89},
        {'type': 'call', 'style': 'european', 'S': 5912.17, 'K': 5920, 'T': 0.0849, 'r': 0.0421, 'qty': 10,   'market_price': 103.66},
        {'type': 'put', 'style': 'european', 'S': 5912.17, 'K': 5900, 'T': 0.0849, 'r': 0.0421, 'qty': 10,   'market_price': 102.92},

    ]
    horizonte_dias = 10 / 252
    N_steps = 20
    n_sim_main = 50000      # Para P&L y VaR/ES
    n_sim_greeks = 100000   # Para griegas
    n_sim_sens = 20000      # Para sensibilidades

    print("\n" + "="*60)
    print("OPTION PORTFOLIO SUMMARY USING MONTE CARLO (MC MODELS)")
    print("="*60)
    sim_mc = simulate_portfolio_mc_pricing(portfolio, n_sims=n_sim_main, n_steps=N_steps, horizon=horizonte_dias)
    pnl_mc = sim_mc['pnl']
    shocks_mc = sim_mc['shocks']
    var_mc, es_mc = var_es(pnl_mc, alpha=0.01)
    value_mc = sum(price_option_mc(opt, n_sim=n_sim_greeks, n_steps=N_steps) * opt['qty'] for opt in portfolio)
    greeks_total_mc = portfolio_greeks_mc(portfolio, n_sim=n_sim_greeks, n_steps=N_steps)
    for i, opt in enumerate(portfolio, 1):
        price_mc = price_option_mc(opt, n_sim=n_sim_greeks, n_steps=N_steps)
        greeks_mc = option_greeks_mc(opt, n_sim=n_sim_greeks, n_steps=N_steps)
        iv_mc = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
        if iv_mc is None:
            ticker = opt.get('ticker', None)
            if ticker is not None:
                iv_mc = get_historical_volatility(ticker)
            else:
                iv_mc = 0.2
        print(f"Option #{i} (MC):")
        print(f"  Type:     {opt['type'].capitalize()}   |  Style: {opt['style'].capitalize()}   |  Quantity: {opt['qty']}")
        print(f"  Spot:     {opt['S']:.2f}   |  Strike: {opt['K']:.2f}   |  Maturity (years): {opt['T']:.4f}")
        print(f"  Market price: {opt['market_price']:.2f}   |  Model price (MC): {price_mc:.2f}")
        print(f"  Volatility used: {iv_mc*100:.2f}%")
        print("  Greeks MC:")
        for g, v in greeks_mc.items():
            if g == 'gamma':
                print(f"    {g.capitalize():<6}: {v:.8f}")
            else:
                print(f"    {g.capitalize():<6}: {v:.4f}")
        print("-"*60)
    print("\nAggregated portfolio Greeks (MC):")
    for g, v in greeks_total_mc.items():
        print(f"  {g.capitalize():<6}: {v:.4f}")
    print(f"\nCurrent portfolio value (MC): {value_mc:.2f}")
    print(f"\nHistorical VaR (MC simulation, PRICING MC, horizon {horizonte_dias*252:.0f} days, 99%): {var_mc:.2f}")
    print(f"Historical ES (MC simulation, PRICING MC, 99%): {es_mc:.2f}")
    print("="*60)
    plt.figure(figsize=(14,8))
    plt.hist(pnl_mc, bins=50, color='lightgreen', edgecolor='k', alpha=0.7, density=True)
    plt.axvline(-var_mc, color='red', linestyle='--', label=f'VaR MC ({-var_mc:.2f})')
    plt.axvline(-es_mc, color='orange', linestyle='--', label=f'ES MC ({-es_mc:.2f})')
    plt.title('Simulated P&L distribution of the portfolio (PRICING MC)')
    plt.xlabel('P&L')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    VIS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
    os.makedirs(VIS_DIR, exist_ok=True)
    plt.savefig(os.path.join(VIS_DIR, 'histogram_pnl_portfolio_mc.png'))
    plt.close()

    # ===========================
    # DELTA HEDGING (MC)
    # ===========================
    print("\n" + "="*60)
    print("DELTA HEDGING ANALYSIS (MONTE CARLO)")
    print("="*60)
    delta_hedge_fraction_mc = 0.7  # Default to 70% coverage
    subyacentes_mc = {}
    for opt in portfolio:
        key = opt.get('ticker', opt['S'])
        greeks_mc = option_greeks_mc(opt, n_sim=n_sim_greeks, n_steps=N_steps)
        subyacentes_mc.setdefault(key, {'S0': opt['S'], 'delta': 0})
        subyacentes_mc[key]['delta'] += greeks_mc['delta'] * opt['qty'] * delta_hedge_fraction_mc
    pnl_mc_hedged = []
    for i in range(len(pnl_mc)):
        hedge_pnl = 0
        for key, v in subyacentes_mc.items():
            S0 = v['S0']
            delta = v['delta']
            Z = shocks_mc[key][i]
            for opt in portfolio:
                if opt.get('ticker', opt['S']) == key:
                    iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                    if iv is None:
                        iv = 0.2
                    r = opt['r']
                    T_sim = horizonte_dias if horizonte_dias is not None else opt['T']
                    break
            S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
            hedge_pnl += -delta * (S_T - S0)
        pnl_mc_hedged.append(pnl_mc[i] + hedge_pnl)
    pnl_mc_hedged = np.array(pnl_mc_hedged)
    var_mc_hedged, es_mc_hedged = var_es(pnl_mc_hedged, alpha=0.01)
    print(f"Delta hedge per underlying (MC):")
    for key, v in subyacentes_mc.items():
        print(f"  Underlying {key}: delta = {v['delta']:.4f}")
    print(f"\nVaR after delta hedge (MC, 99%): {var_mc_hedged:.2f}")
    print(f"ES after delta hedge (MC, 99%): {es_mc_hedged:.2f}")
    print(f"VaR reduction: {var_mc - var_mc_hedged:.2f}")
    print(f"ES reduction: {es_mc - es_mc_hedged:.2f}")
    print("="*60)

    # ===========================
    # GAMMA + DELTA HEDGING (MONTE CARLO)
    # ===========================
    print("\n" + "="*60)
    print("GAMMA + DELTA HEDGING ANALYSIS (MONTE CARLO)")
    print("="*60)
    greeks_total_mc = portfolio_greeks_mc(portfolio, n_sim=n_sim_greeks, n_steps=N_steps)
    gamma_cartera_mc = greeks_total_mc['gamma']
    delta_cartera_mc = greeks_total_mc['delta']
    hedge_opt_mc = {
        'type': 'call',
        'style': 'european',
        'S': portfolio[0]['S'],
        'K': portfolio[0]['K'],
        'T': portfolio[0]['T'],
        'r': portfolio[0]['r'],
        'qty': 0,
        'market_price': portfolio[0]['market_price'],
    }
    greeks_hedge_mc = option_greeks_mc(hedge_opt_mc, n_sim=n_sim_greeks, n_steps=N_steps)
    gamma_hedge_mc = greeks_hedge_mc['gamma']
    gamma_hedge_fraction = 0.7
    qty_gamma_hedge_mc = -gamma_cartera_mc * gamma_hedge_fraction / gamma_hedge_mc if gamma_hedge_mc != 0 else 0
    hedge_opt_mc['qty'] = qty_gamma_hedge_mc
    portfolio_gamma_hedged_mc = portfolio + [hedge_opt_mc]
    greeks_total_gamma_mc = portfolio_greeks_mc(portfolio_gamma_hedged_mc, n_sim=n_sim_greeks, n_steps=N_steps)
    delta_gamma_hedged_mc = greeks_total_gamma_mc['delta']
    pnl_gamma_delta_hedged_mc = []
    for i in range(len(pnl_mc)):
        shocked_portfolio = []
        for opt in portfolio_gamma_hedged_mc:
            key = opt.get('ticker', opt['S'])
            S0 = opt['S']
            iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
            if iv is None:
                iv = 0.2
            r = opt['r']
            T_sim = horizonte_dias if horizonte_dias is not None else opt['T']
            Z = shocks_mc[key][i] if key in shocks_mc else np.random.normal(0, 1)
            S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
            shocked_opt = opt.copy()
            shocked_opt['S'] = S_T
            shocked_portfolio.append(shocked_opt)
        val = sum(price_option_mc(opt, n_sim=500, n_steps=N_steps) * opt['qty'] for opt in shocked_portfolio)
        hedge_pnl = 0
        S0 = portfolio[0]['S']
        delta = delta_gamma_hedged_mc
        Z = shocks_mc[portfolio[0].get('ticker', portfolio[0]['S'])][i]
        S_T = S0 * np.exp((portfolio[0]['r'] - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
        hedge_pnl += -delta * (S_T - S0)
        pnl_gamma_delta_hedged_mc.append(val - value_mc + hedge_pnl)
    pnl_gamma_delta_hedged_mc = np.array(pnl_gamma_delta_hedged_mc)
    var_gamma_delta_hedged_mc, es_gamma_delta_hedged_mc = var_es(pnl_gamma_delta_hedged_mc, alpha=0.01)
    print(f"Gamma hedge: qty = {qty_gamma_hedge_mc:.4f} of ATM call (S={hedge_opt_mc['S']}, K={hedge_opt_mc['K']}) covering {gamma_hedge_fraction*100:.0f}% of gamma")
    print(f"Delta after gamma hedge: {delta_gamma_hedged_mc:.4f}")
    print(f"\nVaR after gamma+delta hedge (MC, 99%): {var_gamma_delta_hedged_mc:.2f}")
    print(f"ES after gamma+delta hedge (MC, 99%): {es_gamma_delta_hedged_mc:.2f}")
    print(f"VaR reduction vs original: {var_mc - var_gamma_delta_hedged_mc:.2f}")
    print(f"ES reduction vs original: {es_mc - es_gamma_delta_hedged_mc:.2f}")
    print("="*60)

    # ===========================
    # VEGA HEDGING (MC)
    # ===========================
    print("\n" + "="*60)
    print("VEGA HEDGING ANALYSIS (MC)")
    print("="*60)
    # 1. Calcula vega neta de la cartera
    vega_total_mc = portfolio_greeks_mc(portfolio, n_sim=n_sim_greeks, n_steps=N_steps)['vega']
    # 2. Define opción de hedge (call europea ATM, mismo S/K/T/r)
    hedge_opt_vega_mc = {
        'type': 'call',
        'style': 'european',
        'S': portfolio[0]['S'],
        'K': portfolio[0]['K'],
        'T': portfolio[0]['T'],
        'r': portfolio[0]['r'],
        'qty': 0,
        'market_price': portfolio[0]['market_price'],
    }
    greeks_hedge_vega_mc = option_greeks_mc(hedge_opt_vega_mc, n_sim=n_sim_greeks, n_steps=N_steps)
    vega_hedge_mc = greeks_hedge_vega_mc['vega']
    vega_hedge_fraction = 0.7  # 70% de la vega neta
    qty_vega_hedge_mc = -vega_total_mc * vega_hedge_fraction / vega_hedge_mc if vega_hedge_mc != 0 else 0
    hedge_opt_vega_mc['qty'] = qty_vega_hedge_mc
    # 3. Crea nueva cartera con vega hedge
    portfolio_vega_hedged_mc = portfolio + [hedge_opt_vega_mc]
    # 4. Simula P&L con vega hedge usando los mismos shocks
    pnl_vega_hedged_mc = []
    for i in range(len(pnl_mc)):
        shocked_portfolio = []
        for opt in portfolio_vega_hedged_mc:
            key = opt.get('ticker', opt['S'])
            S0 = opt['S']
            iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
            if iv is None:
                ticker = opt.get('ticker', None)
                if ticker is not None:
                    iv = get_historical_volatility(ticker)
                else:
                    iv = 0.2
            r = opt['r']
            T_sim = horizonte_dias if horizonte_dias is not None else opt['T']
            Z = shocks_mc[key][i] if key in shocks_mc else np.random.normal(0, 1)
            S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
            shocked_opt = opt.copy()
            shocked_opt['S'] = S_T
            shocked_portfolio.append(shocked_opt)
        val = sum(price_option_mc(opt, n_sim=500, n_steps=N_steps) * opt['qty'] for opt in shocked_portfolio)
        pnl_vega_hedged_mc.append(val - value_mc)
    pnl_vega_hedged_mc = np.array(pnl_vega_hedged_mc)
    var_vega_hedged_mc, es_vega_hedged_mc = var_es(pnl_vega_hedged_mc, alpha=0.01)
    print(f"Vega hedge: qty = {qty_vega_hedge_mc:.4f} of ATM call (S={hedge_opt_vega_mc['S']}, K={hedge_opt_vega_mc['K']}) covering {vega_hedge_fraction*100:.0f}% of vega")
    print(f"\nVaR after vega hedge (MC, 99%): {var_vega_hedged_mc:.2f}")
    print(f"ES after vega hedge (MC, 99%): {es_vega_hedged_mc:.2f}")
    print(f"VaR reduction vs original: {var_mc - var_vega_hedged_mc:.2f}")
    print(f"ES reduction vs original: {es_mc - es_vega_hedged_mc:.2f}")
    print("="*60)

    plt.figure(figsize=(14,8))
    plt.hist(pnl_mc, bins=50, color='skyblue', edgecolor='k', alpha=0.5, density=True, label='Original')
    plt.hist(pnl_mc_hedged, bins=50, color='orange', edgecolor='k', alpha=0.5, density=True, label='Delta Hedge')
    plt.hist(pnl_gamma_delta_hedged_mc, bins=50, color='green', edgecolor='k', alpha=0.5, density=True, label='Gamma+Delta Hedge')
    plt.hist(pnl_vega_hedged_mc, bins=50, color='purple', edgecolor='k', alpha=0.5, density=True, label='Vega Hedge')
    var_o, es_o = var_mc, es_mc
    var_d, es_d = var_mc_hedged, es_mc_hedged
    var_g, es_g = var_gamma_delta_hedged_mc, es_gamma_delta_hedged_mc
    var_v, es_v = var_vega_hedged_mc, es_vega_hedged_mc
    plt.axvline(-var_o, color='blue', linestyle='--', label=f'VaR Original ({-var_o:.0f})')
    plt.axvline(-es_o, color='blue', linestyle=':', label=f'ES Original ({-es_o:.0f})')
    plt.axvline(-var_d, color='orange', linestyle='--', label=f'VaR Delta ({-var_d:.0f})')
    plt.axvline(-es_d, color='orange', linestyle=':', label=f'ES Delta ({-es_d:.0f})')
    plt.axvline(-var_g, color='green', linestyle='--', label=f'VaR Gamma+Delta ({-var_g:.0f})')
    plt.axvline(-es_g, color='green', linestyle=':', label=f'ES Gamma+Delta ({-es_g:.0f})')
    plt.axvline(-var_v, color='purple', linestyle='--', label=f'VaR Vega ({-var_v:.0f})')
    plt.axvline(-es_v, color='purple', linestyle=':', label=f'ES Vega ({-es_v:.0f})')
    plt.title('P&L Distribution Comparison (MONTE CARLO)')
    plt.xlabel('P&L')
    plt.ylabel('Density')
    plt.legend(loc='upper left', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'histogram_compare_mc.png'))
    plt.close()

    # ===========================
    # SENSITIVITY ANALYSIS (MC) - VALOR ABSOLUTO
    # ===========================
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS (MC) - PORTFOLIO VALUE")
    print("="*60)

    # Estrategias de hedge
    hedge_strategies = [
        ('Original', portfolio),
        ('Delta Hedge', portfolio + [{
            'type': 'call', 'style': 'european', 'S': portfolio[0]['S'], 'K': portfolio[0]['K'], 'T': portfolio[0]['T'], 'r': portfolio[0]['r'], 'qty': -portfolio_greeks_mc(portfolio, n_sim=n_sim_greeks, n_steps=N_steps)['delta'], 'market_price': portfolio[0]['market_price']
        }]),
        ('Delta-Gamma Hedge', portfolio_gamma_hedged_mc),
        ('Vega Hedge', portfolio_vega_hedged_mc),
    ]

    # 1. Sensibilidad Spot (±10%)
    spot_base = portfolio[0]['S']
    spot_range = np.linspace(spot_base*0.9, spot_base*1.1, 21)
    spot_idx_low = 0
    spot_idx_base = 10
    spot_idx_high = 20
    plt.figure(figsize=(12,7))
    print("\n=== Spot Sensitivity (±10%) ===")
    print(f"{'Strategy':<20}{'Base':>12}{'-10%':>12}{'+10%':>12}{'Δ-10%':>12}{'Δ+10%':>12}")
    spot_rows = []
    for name, port in hedge_strategies:
        values = []
        for S in spot_range:
            port_mod = [opt.copy() for opt in port]
            for opt in port_mod:
                opt['S'] = S
            values.append(sum(price_option_mc(opt, n_sim=n_sim_sens, n_steps=N_steps) * opt['qty'] for opt in port_mod))
        plt.plot(spot_range, values, label=name)
        plt.scatter([spot_range[spot_idx_low], spot_range[spot_idx_base], spot_range[spot_idx_high]],
                    [values[spot_idx_low], values[spot_idx_base], values[spot_idx_high]],
                    marker='o', s=80)
        for idx, label in zip([spot_idx_low, spot_idx_base, spot_idx_high], ['-10%', 'Base', '+10%']):
            plt.annotate(f"{label}\n{values[idx]:.2f}", (spot_range[idx], values[idx]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        base = values[spot_idx_base]
        low = values[spot_idx_low]
        high = values[spot_idx_high]
        dlow = low - base
        dhigh = high - base
        print(f"{name:<20}{base:12.4f}{low:12.4f}{high:12.4f}{dlow:12.4f}{dhigh:12.4f}")
        spot_rows.append({
            'Strategy': name,
            'Base': base,
            '-10%': low,
            '+10%': high,
            'Δ-10%': dlow,
            'Δ+10%': dhigh
        })
    df_spot = pd.DataFrame(spot_rows)
    df_spot.to_csv(os.path.join(VIS_DIR, 'sensitivity_spot_mc.csv'), index=False)
    plt.xlabel('Spot')
    plt.ylabel('Portfolio Value')
    plt.title('Sensitivity to Spot - All Strategies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'sensitivity_spot_mc.png'))
    plt.close()

    # 2. Sensibilidad tipo de interés (±1%)
    r_base = portfolio[0]['r']
    r_range = np.linspace(r_base-0.01, r_base+0.01, 21)
    r_idx_low = 0
    r_idx_base = 10
    r_idx_high = 20
    plt.figure(figsize=(12,7))
    print("\n=== r Sensitivity (±1%) ===")
    print(f"{'Strategy':<20}{'Base':>12}{'-1%':>12}{'+1%':>12}{'Δ-1%':>12}{'Δ+1%':>12}")
    r_rows = []
    for name, port in hedge_strategies:
        values = []
        for r in r_range:
            port_mod = [opt.copy() for opt in port]
            for opt in port_mod:
                opt['r'] = r
            values.append(sum(price_option_mc(opt, n_sim=n_sim_sens, n_steps=N_steps) * opt['qty'] for opt in port_mod))
        plt.plot(r_range, values, label=name)
        plt.scatter([r_range[r_idx_low], r_range[r_idx_base], r_range[r_idx_high]],
                    [values[r_idx_low], values[r_idx_base], values[r_idx_high]],
                    marker='o', s=80)
        for idx, label in zip([r_idx_low, r_idx_base, r_idx_high], ['-1%', 'Base', '+1%']):
            plt.annotate(f"{label}\n{values[idx]:.2f}", (r_range[idx], values[idx]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        base = values[r_idx_base]
        low = values[r_idx_low]
        high = values[r_idx_high]
        dlow = low - base
        dhigh = high - base
        print(f"{name:<20}{base:12.4f}{low:12.4f}{high:12.4f}{dlow:12.4f}{dhigh:12.4f}")
        r_rows.append({
            'Strategy': name,
            'Base': base,
            '-1%': low,
            '+1%': high,
            'Δ-1%': dlow,
            'Δ+1%': dhigh
        })
    df_r = pd.DataFrame(r_rows)
    df_r.to_csv(os.path.join(VIS_DIR, 'sensitivity_r_mc.csv'), index=False)
    plt.xlabel('Risk-free rate (r)')
    plt.ylabel('Portfolio Value')
    plt.title('Sensitivity to r - All Strategies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'sensitivity_r_mc.png'))
    plt.close()

    # 3. Sensibilidad volatilidad (±20%)
    plt.figure(figsize=(12,7))
    vol_base = 1.0
    vol_range = np.linspace(0.8, 1.2, 21)  # multiplicador de la volatilidad implícita
    vol_idx_low = 0
    vol_idx_base = 10
    vol_idx_high = 20
    print("\n=== Volatility Sensitivity (±20%) ===")
    print(f"{'Strategy':<20}{'Base':>12}{'-20%':>12}{'+20%':>12}{'Δ-20%':>12}{'Δ+20%':>12}")
    vol_rows = []
    for name, port in hedge_strategies:
        values = []
        for vol_mult in vol_range:
            port_mod = [opt.copy() for opt in port]
            for opt in port_mod:
                iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                if iv is None:
                    iv = 0.2
                opt['market_price'] = black_scholes_call_price(opt['S'], opt['K'], opt['T'], opt['r'], iv*vol_mult) if opt['type']=='call' else black_scholes_put_price(opt['S'], opt['K'], opt['T'], opt['r'], iv*vol_mult)
            values.append(sum(price_option_mc(opt, n_sim=n_sim_sens, n_steps=N_steps) * opt['qty'] for opt in port_mod))
        plt.plot(vol_range, values, label=name)
        plt.scatter([vol_range[vol_idx_low], vol_range[vol_idx_base], vol_range[vol_idx_high]],
                    [values[vol_idx_low], values[vol_idx_base], values[vol_idx_high]],
                    marker='o', s=80)
        for idx, label in zip([vol_idx_low, vol_idx_base, vol_idx_high], ['-20%', 'Base', '+20%']):
            plt.annotate(f"{label}\n{values[idx]:.2f}", (vol_range[idx], values[idx]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        base = values[vol_idx_base]
        low = values[vol_idx_low]
        high = values[vol_idx_high]
        dlow = low - base
        dhigh = high - base
        print(f"{name:<20}{base:12.4f}{low:12.4f}{high:12.4f}{dlow:12.4f}{dhigh:12.4f}")
        vol_rows.append({
            'Strategy': name,
            'Base': base,
            '-20%': low,
            '+20%': high,
            'Δ-20%': dlow,
            'Δ+20%': dhigh
        })
    df_vol = pd.DataFrame(vol_rows)
    df_vol.to_csv(os.path.join(VIS_DIR, 'sensitivity_vol_mc.csv'), index=False)
    plt.xlabel('Volatility multiplier')
    plt.ylabel('Portfolio Value')
    plt.title('Sensitivity to Volatility - All Strategies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'sensitivity_vol_mc.png'))
    plt.close()

    print('Sensitivity analysis completed. PNG files generated.') 