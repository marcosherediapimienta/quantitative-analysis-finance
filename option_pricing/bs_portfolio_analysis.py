import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import copy

# =============================
# Black-Scholes Functions
# =============================
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

def bs_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega_ = S * np.sqrt(T) * norm.pdf(d1)
    return {'delta': delta, 'gamma': gamma, 'vega': vega_, 'theta': theta, 'rho': rho}

def price_option(opt):
    iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
    if iv is None:
        iv = 0.2
    if opt['type'] == 'call':
        price = black_scholes_call_price(opt['S'], opt['K'], opt['T'], opt['r'], iv)
    else:
        price = black_scholes_put_price(opt['S'], opt['K'], opt['T'], opt['r'], iv)
    return price, iv

def option_greeks(opt):
    iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
    if iv is None:
        iv = 0.2
    return bs_greeks(opt['S'], opt['K'], opt['T'], opt['r'], iv, opt['type'])

def portfolio_value(portfolio):
    value = 0
    for opt in portfolio:
        price, _ = price_option(opt)
        value += price * opt['qty']
    return value

def portfolio_greeks(portfolio):
    total = {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
    for opt in portfolio:
        greeks = option_greeks(opt)
        for g in total:
            total[g] += greeks[g] * opt['qty']
    return total

def simulate_portfolio(portfolio, n_sims=10000, horizon=None, vol_shock_sigma=0.1, rho=-0.5):
    base_val = portfolio_value(portfolio)
    pnl = []
    shocks_dict = {}
    params = []
    for opt in portfolio:
        _, iv = price_option(opt)
        params.append({
            'S': opt['S'],
            'K': opt['K'],
            'T': opt['T'],
            'r': opt['r'],
            'qty': opt['qty'],
            'type': opt['type'],
            'market_price': opt['market_price'],
            'iv': iv,
            'key': opt.get('ticker', opt['S'])
        })
    for p in params:
        shocks_dict.setdefault(p['key'], [])
    for i in range(n_sims):
        shocked_portfolio = []
        Zs = {}
        sigma_shocks = []
        for idx, p in enumerate(params):
            # Correlated shocks
            Z1, Z2 = np.random.normal(size=2)
            Z_spot = Z1
            Z_vol = rho * Z1 + np.sqrt(1 - rho**2) * Z2
            Zs[p['key']] = Z_spot
            shocks_dict[p['key']].append(Z_spot)
            T_sim = horizon if horizon is not None else p['T']
            # Simular shock de volatilidad lognormal correlacionado
            vol_shock = np.random.lognormal(mean=0, sigma=vol_shock_sigma) * np.exp(Z_vol * vol_shock_sigma)
            sigma_shocks.append(p['iv'] * vol_shock)
            S_T = p['S'] * np.exp((p['r'] - 0.5 * p['iv'] ** 2) * T_sim + p['iv'] * np.sqrt(T_sim) * Z_spot)
            shocked_opt = p.copy()
            shocked_opt['S'] = S_T
            shocked_portfolio.append(shocked_opt)
        shocked_val = portfolio_value(shocked_portfolio, sigma_overrides=sigma_shocks)
        pnl.append(shocked_val - base_val)
    return {'pnl': np.array(pnl), 'shocks': shocks_dict}

def var_es(pnl, alpha=0.01):
    pnl_sorted = np.sort(pnl)
    var = -np.percentile(pnl_sorted, alpha*100)
    es = -pnl_sorted[pnl_sorted <= -var].mean()
    return var, es

if __name__ == "__main__":
    # --- Definición de la cartera ---
    portfolio = [
        {'type': 'call', 'S': 5802.82, 'K': 5800, 'T': 0.0849, 'r': 0.0421, 'qty': -10, 'market_price': 152.80},
        {'type': 'put',  'S': 5802.82, 'K': 5800, 'T': 0.0849, 'r': 0.0421, 'qty': -5,  'market_price': 147.20},
        {'type': 'call', 'S': 5802.82, 'K': 5900, 'T': 0.0849, 'r': 0.0421, 'qty': 5,   'market_price': 80.00},
    ]
    horizonte_dias = 10 / 252  # días de trading

    print("\n" + "="*60)
    print("OPTION PORTFOLIO SUMMARY USING BLACK-SCHOLES")
    print("="*60)
    sim_bs = simulate_portfolio(portfolio, n_sims=10000, horizon=horizonte_dias)
    pnl_bs = sim_bs['pnl']
    shocks_bs = sim_bs['shocks']
    var_bs, es_bs = var_es(pnl_bs, alpha=0.01)
    value_bs = portfolio_value(portfolio)
    greeks_total_bs = portfolio_greeks(portfolio)
    for i, opt in enumerate(portfolio, 1):
        price, iv = price_option(opt)
        greeks = option_greeks(opt)
        print(f"Option #{i}:")
        print(f"  Type:     {opt['type'].capitalize()}   |  Quantity: {opt['qty']}")
        print(f"  Spot:     {opt['S']:.2f}   |  Strike: {opt['K']:.2f}   |  Maturity (years): {opt['T']:.4f}")
        print(f"  Market price: {opt['market_price']:.2f}   |  Model price: {price:.2f}")
        print(f"  Implied volatility: {iv*100:.2f}%")
        print("  Greeks:")
        for g, v in greeks.items():
            if g == 'gamma':
                print(f"    {g.capitalize():<6}: {v:.8f}")
            else:
                print(f"    {g.capitalize():<6}: {v:.4f}")
        print("-"*60)
    print("\nAggregated portfolio Greeks:")
    for g, v in greeks_total_bs.items():
        print(f"  {g.capitalize():<6}: {v:.4f}")
    print(f"\nCurrent portfolio value: {value_bs:.2f}")
    print(f"\nHistorical VaR (MC simulation, BS, horizon {horizonte_dias*252:.0f} days, 99%): {var_bs:.2f}")
    print(f"Historical ES (MC simulation, BS, 99%): {es_bs:.2f}")
    print("="*60)
    plt.figure(figsize=(14,8))
    plt.hist(pnl_bs, bins=50, color='skyblue', edgecolor='k', alpha=0.7, density=True)
    plt.axvline(-var_bs, color='red', linestyle='--', label=f'VaR ({-var_bs:.2f})')
    plt.axvline(-es_bs, color='orange', linestyle='--', label=f'ES ({-es_bs:.2f})')
    plt.title('Simulated P&L distribution of the portfolio (Black-Scholes)')
    plt.xlabel('P&L')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('histogram_pnl_portfolio_bs.png')
    plt.close()

    # ===========================
    # DELTA HEDGING (BS)
    # ===========================
    print("\n" + "="*60)
    print("DELTA HEDGING ANALYSIS (BLACK-SCHOLES)")
    print("="*60)
    subyacentes = {}
    for opt in portfolio:
        key = opt.get('ticker', opt['S'])
        greeks = option_greeks(opt)
        subyacentes.setdefault(key, {'S0': opt['S'], 'delta': 0})
        subyacentes[key]['delta'] += greeks['delta'] * opt['qty']
    pnl_bs_hedged = []
    for i in range(len(pnl_bs)):
        hedge_pnl = 0
        for key, v in subyacentes.items():
            S0 = v['S0']
            delta = v['delta']
            Z = shocks_bs[key][i]
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
        pnl_bs_hedged.append(pnl_bs[i] + hedge_pnl)
    pnl_bs_hedged = np.array(pnl_bs_hedged)
    var_bs_hedged, es_bs_hedged = var_es(pnl_bs_hedged, alpha=0.01)
    print(f"Delta hedge per underlying:")
    for key, v in subyacentes.items():
        print(f"  Underlying {key}: delta = {v['delta']:.4f}")
    print(f"\nVaR after delta hedge (BS, 99%): {var_bs_hedged:.2f}")
    print(f"ES after delta hedge (BS, 99%): {es_bs_hedged:.2f}")
    print(f"VaR reduction: {var_bs - var_bs_hedged:.2f}")
    print(f"ES reduction: {es_bs - es_bs_hedged:.2f}")
    print("="*60)

    # ===========================
    # GAMMA + DELTA HEDGING (BS)
    # ===========================
    print("\n" + "="*60)
    print("GAMMA + DELTA HEDGING ANALYSIS (BLACK-SCHOLES)")
    print("="*60)
    greeks_total = portfolio_greeks(portfolio)
    gamma_cartera = greeks_total['gamma']
    delta_cartera = greeks_total['delta']
    hedge_opt = {
        'type': 'call',
        'S': portfolio[0]['S'],
        'K': portfolio[0]['K'],
        'T': portfolio[0]['T'],
        'r': portfolio[0]['r'],
        'qty': 0,
        'market_price': portfolio[0]['market_price'],
    }
    greeks_hedge = option_greeks(hedge_opt)
    gamma_hedge = greeks_hedge['gamma']
    gamma_hedge_fraction = 0.7
    qty_gamma_hedge = -gamma_cartera * gamma_hedge_fraction / gamma_hedge if gamma_hedge != 0 else 0
    hedge_opt['qty'] = qty_gamma_hedge
    portfolio_gamma_hedged = portfolio + [hedge_opt]
    greeks_total_gamma = portfolio_greeks(portfolio_gamma_hedged)
    delta_gamma_hedged = greeks_total_gamma['delta']
    pnl_gamma_delta_hedged = []
    for i in range(len(pnl_bs)):
        shocked_portfolio = []
        for opt in portfolio_gamma_hedged:
            key = opt.get('ticker', opt['S'])
            S0 = opt['S']
            iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
            if iv is None:
                iv = 0.2
            r = opt['r']
            T_sim = horizonte_dias if horizonte_dias is not None else opt['T']
            Z = shocks_bs[key][i] if key in shocks_bs else np.random.normal(0, 1)
            S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
            shocked_opt = opt.copy()
            shocked_opt['S'] = S_T
            shocked_portfolio.append(shocked_opt)
        val = portfolio_value(shocked_portfolio)
        hedge_pnl = 0
        S0 = portfolio[0]['S']
        delta = delta_gamma_hedged
        Z = shocks_bs[portfolio[0].get('ticker', portfolio[0]['S'])][i]
        S_T = S0 * np.exp((portfolio[0]['r'] - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
        hedge_pnl += -delta * (S_T - S0)
        pnl_gamma_delta_hedged.append(val - value_bs + hedge_pnl)
    pnl_gamma_delta_hedged = np.array(pnl_gamma_delta_hedged)
    var_gamma_delta_hedged, es_gamma_delta_hedged = var_es(pnl_gamma_delta_hedged, alpha=0.01)
    print(f"Gamma hedge: qty = {qty_gamma_hedge:.4f} of ATM call (S={hedge_opt['S']}, K={hedge_opt['K']}) covering {gamma_hedge_fraction*100:.0f}% of gamma")
    print(f"Delta after gamma hedge: {delta_gamma_hedged:.4f}")
    print(f"\nVaR after gamma+delta hedge (BS, 99%): {var_gamma_delta_hedged:.2f}")
    print(f"ES after gamma+delta hedge (BS, 99%): {es_gamma_delta_hedged:.2f}")
    print(f"VaR reduction vs original: {var_bs - var_gamma_delta_hedged:.2f}")
    print(f"ES reduction vs original: {es_bs - es_gamma_delta_hedged:.2f}")
    print("="*60)

    # ===========================
    # VEGA HEDGING (BS)
    # ===========================
    print("\n" + "="*60)
    print("VEGA HEDGING ANALYSIS (BLACK-SCHOLES)")
    print("="*60)
    vega_total = portfolio_greeks(portfolio)['vega']
    hedge_opt_vega = {
        'type': 'call',
        'S': portfolio[0]['S'],
        'K': portfolio[0]['K'],
        'T': portfolio[0]['T'],
        'r': portfolio[0]['r'],
        'qty': 0,
        'market_price': portfolio[0]['market_price'],
    }
    greeks_hedge_vega = option_greeks(hedge_opt_vega)
    vega_hedge = greeks_hedge_vega['vega']
    vega_hedge_fraction = 0.7
    qty_vega_hedge = -vega_total * vega_hedge_fraction / vega_hedge if vega_hedge != 0 else 0
    hedge_opt_vega['qty'] = qty_vega_hedge
    portfolio_vega_hedged = portfolio + [hedge_opt_vega]
    pnl_vega_hedged = []
    for i in range(len(pnl_bs)):
        shocked_portfolio = []
        for opt in portfolio_vega_hedged:
            key = opt.get('ticker', opt['S'])
            S0 = opt['S']
            iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
            if iv is None:
                iv = 0.2
            r = opt['r']
            T_sim = horizonte_dias if horizonte_dias is not None else opt['T']
            Z = shocks_bs[key][i] if key in shocks_bs else np.random.normal(0, 1)
            S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
            shocked_opt = opt.copy()
            shocked_opt['S'] = S_T
            shocked_portfolio.append(shocked_opt)
        val = portfolio_value(shocked_portfolio)
        pnl_vega_hedged.append(val - value_bs)
    pnl_vega_hedged = np.array(pnl_vega_hedged)
    var_vega_hedged, es_vega_hedged = var_es(pnl_vega_hedged, alpha=0.01)
    print(f"Vega hedge: qty = {qty_vega_hedge:.4f} of ATM call (S={hedge_opt_vega['S']}, K={hedge_opt_vega['K']}) covering {vega_hedge_fraction*100:.0f}% of vega")
    print(f"\nVaR after vega hedge (BS, 99%): {var_vega_hedged:.2f}")
    print(f"ES after vega hedge (BS, 99%): {es_vega_hedged:.2f}")
    print(f"VaR reduction vs original: {var_bs - var_vega_hedged:.2f}")
    print(f"ES reduction vs original: {es_bs - es_vega_hedged:.2f}")
    print("="*60)

    # Añadir al histograma comparativo
    plt.figure(figsize=(14,8))
    plt.hist(pnl_bs, bins=50, color='skyblue', edgecolor='k', alpha=0.5, density=True, label='Original')
    plt.hist(pnl_bs_hedged, bins=50, color='orange', edgecolor='k', alpha=0.5, density=True, label='Delta Hedge')
    plt.hist(pnl_gamma_delta_hedged, bins=50, color='green', edgecolor='k', alpha=0.5, density=True, label='Gamma+Delta Hedge')
    plt.hist(pnl_vega_hedged, bins=50, color='purple', edgecolor='k', alpha=0.5, density=True, label='Vega Hedge')
    var_o, es_o = var_bs, es_bs
    var_d, es_d = var_bs_hedged, es_bs_hedged
    var_g, es_g = var_gamma_delta_hedged, es_gamma_delta_hedged
    var_v, es_v = var_vega_hedged, es_vega_hedged
    plt.axvline(-var_o, color='blue', linestyle='--', label=f'VaR Original ({-var_o:.0f})')
    plt.axvline(-es_o, color='blue', linestyle=':', label=f'ES Original ({-es_o:.0f})')
    plt.axvline(-var_d, color='orange', linestyle='--', label=f'VaR Delta ({-var_d:.0f})')
    plt.axvline(-es_d, color='orange', linestyle=':', label=f'ES Delta ({-es_d:.0f})')
    plt.axvline(-var_g, color='green', linestyle='--', label=f'VaR Gamma+Delta ({-var_g:.0f})')
    plt.axvline(-es_g, color='green', linestyle=':', label=f'ES Gamma+Delta ({-es_g:.0f})')
    plt.axvline(-var_v, color='purple', linestyle='--', label=f'VaR Vega ({-var_v:.0f})')
    plt.axvline(-es_v, color='purple', linestyle=':', label=f'ES Vega ({-es_v:.0f})')
    plt.title('P&L Distribution Comparison (Black-Scholes)')
    plt.xlabel('P&L')
    plt.ylabel('Density')
    plt.legend(loc='upper left', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('histogram_compare_bs.png')
    plt.close()

    # ===========================
    # SENSITIVITY ANALYSIS (BS) - VALOR ABSOLUTO
    # ===========================
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS (BLACK-SCHOLES) - PORTFOLIO VALUE")
    print("="*60)

    hedge_strategies = [
        ('Original', portfolio),
        ('Delta Hedge', portfolio + [{
            'type': 'call', 'S': portfolio[0]['S'], 'K': portfolio[0]['K'], 'T': portfolio[0]['T'], 'r': portfolio[0]['r'], 'qty': -portfolio_greeks(portfolio)['delta'], 'market_price': portfolio[0]['market_price']
        }]),
        ('Delta-Gamma Hedge', portfolio_gamma_hedged),
        ('Vega Hedge', portfolio_vega_hedged),
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
    for name, port in hedge_strategies:
        values = []
        for S in spot_range:
            port_mod = copy.deepcopy(port)
            for opt in port_mod:
                opt['S'] = S
            values.append(portfolio_value(port_mod))
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
    plt.xlabel('Spot')
    plt.ylabel('Portfolio Value')
    plt.title('Sensitivity to Spot - All Strategies (BS)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sensitivity_spot_all_bs.png')
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
    for name, port in hedge_strategies:
        values = []
        for r in r_range:
            port_mod = copy.deepcopy(port)
            for opt in port_mod:
                opt['r'] = r
            values.append(portfolio_value(port_mod))
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
    plt.xlabel('Risk-free rate (r)')
    plt.ylabel('Portfolio Value')
    plt.title('Sensitivity to r - All Strategies (BS)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sensitivity_r_all_bs.png')
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
    for name, port in hedge_strategies:
        values = []
        for vol_mult in vol_range:
            port_mod = copy.deepcopy(port)
            for opt in port_mod:
                iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
                if iv is None:
                    iv = 0.2
                new_iv = iv * vol_mult
                if opt['type'] == 'call':
                    opt['market_price'] = black_scholes_call_price(opt['S'], opt['K'], opt['T'], opt['r'], new_iv)
                else:
                    opt['market_price'] = black_scholes_put_price(opt['S'], opt['K'], opt['T'], opt['r'], new_iv)
            values.append(portfolio_value(port_mod))
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
    plt.xlabel('Volatility multiplier')
    plt.ylabel('Portfolio Value')
    plt.title('Sensitivity to Volatility - All Strategies (BS)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sensitivity_vol_all_bs.png')
    plt.close()

    print('Sensitivity analysis completed. PNG files generated.') 