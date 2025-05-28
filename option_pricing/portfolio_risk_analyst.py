import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib.pyplot as plt

from binomial_model.american_options.american_binomial import binomial_american_option_price, binomial_greeks_american_option, get_historical_volatility
from binomial_model.european_options.european_binomial import binomial_european_option_price, binomial_greeks_european_option
from monte_carlo.american_options.american_MC import american_option_longstaff_schwartz, american_mc_greeks
from monte_carlo.european_options.european_MC import monte_carlo_european_option, mc_greeks

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

def price_option(opt, N=100):
    iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
    if iv is None:
        iv = 0.2
    if opt['style'] == 'american':
        price = binomial_american_option_price(opt['S'], opt['K'], opt['T'], opt['r'], iv, N, opt['type'])
    else:
        price = binomial_european_option_price(opt['S'], opt['K'], opt['T'], opt['r'], iv, N, opt['type'])
    return price, iv

def portfolio_value(portfolio, N=100):
    value = 0
    for opt in portfolio:
        price, _ = price_option(opt, N)
        value += price * opt['qty']
    return value

def simulate_portfolio(portfolio, n_sims=10000, N=100, horizon=None):
    # Simula escenarios de precios del subyacente usando GBM y la volatilidad implícita de cada opción
    # horizon: horizonte de simulación en años (si None, usa el vencimiento de cada opción)
    base_val = portfolio_value(portfolio, N)
    pnl = []
    shocks_dict = {}
    # Precalcula las IVs y parámetros para cada opción
    params = []
    for opt in portfolio:
        _, iv = price_option(opt, N)
        params.append({
            'S': opt['S'],
            'K': opt['K'],
            'T': opt['T'],
            'r': opt['r'],
            'qty': opt['qty'],
            'style': opt['style'],
            'type': opt['type'],
            'market_price': opt['market_price'],
            'iv': iv,
            'key': opt.get('ticker', opt['S'])
        })
    # Inicializa shocks_dict
    for p in params:
        shocks_dict.setdefault(p['key'], [])
    for i in range(n_sims):
        shocked_portfolio = []
        Zs = {}
        for p in params:
            Z = np.random.normal(0, 1)
            Zs[p['key']] = Z
            shocks_dict[p['key']].append(Z)
            T_sim = horizon if horizon is not None else p['T']
            S_T = p['S'] * np.exp((p['r'] - 0.5 * p['iv'] ** 2) * T_sim + p['iv'] * np.sqrt(T_sim) * Z)
            shocked_opt = p.copy()
            shocked_opt['S'] = S_T
            shocked_portfolio.append(shocked_opt)
        shocked_val = portfolio_value(shocked_portfolio, N)
        pnl.append(shocked_val - base_val)
    return {'pnl': np.array(pnl), 'shocks': shocks_dict}

def var_es(pnl, alpha=0.01):
    pnl_sorted = np.sort(pnl)
    var = -np.percentile(pnl_sorted, alpha*100)
    es = -pnl_sorted[pnl_sorted <= -var].mean()
    return var, es

def option_greeks(opt, N=100):
    iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
    if iv is None:
        iv = 0.2
    if opt['style'] == 'american':
        greeks = binomial_greeks_american_option(opt['S'], opt['K'], opt['T'], opt['r'], iv, N, opt['type'])
    else:
        greeks = binomial_greeks_european_option(opt['S'], opt['K'], opt['T'], opt['r'], iv, N, opt['type'])
    return greeks

def portfolio_greeks(portfolio, N=100):
    total = {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
    for opt in portfolio:
        greeks = option_greeks(opt, N)
        for g in total:
            total[g] += greeks[g] * opt['qty']
    return total

def gamma_bs(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def price_option_mc(opt, n_sim=10000, n_steps=50):
    # Usa Monte Carlo para precio de opción con volatilidad implícita o histórica
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
    # Usa Monte Carlo para griegas con volatilidad implícita o histórica
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

def simulate_portfolio_mc_pricing(portfolio, n_sims=1000, n_steps=50, horizon=None):
    # Simula escenarios de precios del subyacente usando GBM y valora opciones con MC
    base_val = sum(price_option_mc(opt, n_sim=1000, n_steps=n_steps) * opt['qty'] for opt in portfolio)
    pnl = []
    shocks_dict = {}
    for opt in portfolio:
        key = opt.get('ticker', opt['S'])
        shocks_dict.setdefault(key, [])
    for i in range(n_sims):
        shocked_portfolio = []
        Zs = {}
        for opt in portfolio:
            iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
            if iv is None:
                ticker = opt.get('ticker', None)
                if ticker is not None:
                    iv = get_historical_volatility(ticker)
                else:
                    iv = 0.2
            T_sim = horizon if horizon is not None else opt['T']
            Z = np.random.normal(0, 1)
            key = opt.get('ticker', opt['S'])
            Zs[key] = Z
            shocks_dict[key].append(Z)
            S_T = opt['S'] * np.exp((opt['r'] - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
            shocked_opt = opt.copy()
            shocked_opt['S'] = S_T
            shocked_portfolio.append(shocked_opt)
        shocked_val = sum(price_option_mc(opt, n_sim=500, n_steps=n_steps) * opt['qty'] for opt in shocked_portfolio)
        pnl.append(shocked_val - base_val)
    return {'pnl': np.array(pnl), 'shocks': shocks_dict}

# ===========================
# BINOMIAL ANALYSIS
# ===========================
if __name__ == "__main__":
    # --- Definición de la cartera ---
    portfolio = [
        {'type': 'call', 'style': 'european', 'S': 5802.82, 'K': 5800, 'T': 0.0849, 'r': 0.0421, 'qty': -10, 'market_price': 152.80},
        {'type': 'put',  'style': 'european', 'S': 5802.82, 'K': 5800, 'T': 0.0849, 'r': 0.0421, 'qty': -5,  'market_price': 147.20},
        {'type': 'call', 'style': 'european', 'S': 5802.82, 'K': 5900, 'T': 0.0849, 'r': 0.0421, 'qty': 5,   'market_price': 80.00},
    ]
    horizonte_dias = 10 / 252  # días de trading
    N_steps = 100

    print("\n" + "="*60)
    print("OPTION PORTFOLIO SUMMARY USING BINOMIAL")
    print("="*60)
    sim_binomial = simulate_portfolio(portfolio, n_sims=10000, N=N_steps, horizon=horizonte_dias)
    pnl_binomial = sim_binomial['pnl']
    shocks_binomial = sim_binomial['shocks']
    var_binomial, es_binomial = var_es(pnl_binomial, alpha=0.01)
    value_binomial = portfolio_value(portfolio, N=N_steps)
    greeks_total_binomial = portfolio_greeks(portfolio, N=N_steps)
    for i, opt in enumerate(portfolio, 1):
        price, iv = price_option(opt, N=N_steps)
        greeks = option_greeks(opt, N=N_steps)
        gamma_bs_val = gamma_bs(opt['S'], opt['K'], opt['T'], opt['r'], iv)
        print(f"Option #{i}:")
        print(f"  Type:     {opt['type'].capitalize()}   |  Style: {opt['style'].capitalize()}   |  Quantity: {opt['qty']}")
        print(f"  Spot:     {opt['S']:.2f}   |  Strike: {opt['K']:.2f}   |  Maturity (years): {opt['T']:.4f}")
        print(f"  Market price: {opt['market_price']:.2f}   |  Model price: {price:.2f}")
        print(f"  Implied volatility: {iv*100:.2f}%")
        print("  Greeks:")
        for g, v in greeks.items():
            if g == 'gamma':
                print(f"    {g.capitalize():<6}: {v:.8f}")
            else:
                print(f"    {g.capitalize():<6}: {v:.4f}")
        print(f"    Gamma Black-Scholes: {gamma_bs_val:.8f}")
        print("-"*60)
    print("\nAggregated portfolio Greeks:")
    for g, v in greeks_total_binomial.items():
        print(f"  {g.capitalize():<6}: {v:.4f}")
    print(f"\nCurrent portfolio value: {value_binomial:.2f}")
    print(f"\nHistorical VaR (MC simulation, BINOMIAL, horizon {horizonte_dias*252:.0f} days, 99%): {var_binomial:.2f}")
    print(f"Historical ES (MC simulation, BINOMIAL, 99%): {es_binomial:.2f}")
    print("="*60)
    plt.figure(figsize=(14,8))
    plt.hist(pnl_binomial, bins=50, color='skyblue', edgecolor='k', alpha=0.7, density=True)
    plt.axvline(-var_binomial, color='red', linestyle='--', label=f'VaR ({-var_binomial:.2f})')
    plt.axvline(-es_binomial, color='orange', linestyle='--', label=f'ES ({-es_binomial:.2f})')
    plt.title('Simulated P&L distribution of the portfolio (BINOMIAL)')
    plt.xlabel('P&L')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('histogram_pnl_portfolio_binomial.png')
    plt.close()

    # ===========================
    # DELTA HEDGING (BINOMIAL)
    # ===========================
    print("\n" + "="*60)
    print("DELTA HEDGING ANALYSIS (BINOMIAL)")
    print("="*60)
    # 1. Calcular delta total por subyacente
    subyacentes = {}
    for opt in portfolio:
        key = opt.get('ticker', opt['S'])
        greeks = option_greeks(opt, N=N_steps)
        subyacentes.setdefault(key, {'S0': opt['S'], 'delta': 0})
        subyacentes[key]['delta'] += greeks['delta'] * opt['qty']
    # 2. Simular P&L hedgeado usando los mismos shocks
    pnl_binomial_hedged = []
    for i in range(len(pnl_binomial)):
        hedge_pnl = 0
        for key, v in subyacentes.items():
            S0 = v['S0']
            delta = v['delta']
            Z = shocks_binomial[key][i]
            # Busca la iv y r de una opción con ese subyacente (puedes tomar la primera que encuentres)
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
        pnl_binomial_hedged.append(pnl_binomial[i] + hedge_pnl)
    pnl_binomial_hedged = np.array(pnl_binomial_hedged)
    var_binomial_hedged, es_binomial_hedged = var_es(pnl_binomial_hedged, alpha=0.01)
    print(f"Delta hedge per underlying:")
    for key, v in subyacentes.items():
        print(f"  Underlying {key}: delta = {v['delta']:.4f}")
    print(f"\nVaR after delta hedge (BINOMIAL, 99%): {var_binomial_hedged:.2f}")
    print(f"ES after delta hedge (BINOMIAL, 99%): {es_binomial_hedged:.2f}")
    print(f"VaR reduction: {var_binomial - var_binomial_hedged:.2f}")
    print(f"ES reduction: {es_binomial - es_binomial_hedged:.2f}")
    print("="*60)

    # ===========================
    # GAMMA + DELTA HEDGING (BINOMIAL)
    # ===========================
    print("\n" + "="*60)
    print("GAMMA + DELTA HEDGING ANALYSIS (BINOMIAL)")
    print("="*60)
    # 1. Calcula gamma neta de la cartera
    greeks_total = portfolio_greeks(portfolio, N=N_steps)
    gamma_cartera = greeks_total['gamma']
    delta_cartera = greeks_total['delta']
    # 2. Define opción de hedge (call europea ATM, mismo S/K/T/r)
    hedge_opt = {
        'type': 'call',
        'style': 'european',
        'S': portfolio[0]['S'],
        'K': portfolio[0]['K'],
        'T': portfolio[0]['T'],
        'r': portfolio[0]['r'],
        'qty': 0,  # se calcula
        'market_price': portfolio[0]['market_price'],
    }
    greeks_hedge = option_greeks(hedge_opt, N=N_steps)
    gamma_hedge = greeks_hedge['gamma']
    gamma_hedge_fraction = 0.5  # Porcentaje de gamma a cubrir (50% típico en la práctica)
    # 3. Calcula cantidad necesaria para neutralizar gamma
    qty_gamma_hedge = -gamma_cartera * gamma_hedge_fraction / gamma_hedge if gamma_hedge != 0 else 0
    hedge_opt['qty'] = qty_gamma_hedge
    # 4. Crea nueva cartera con gamma hedge
    portfolio_gamma_hedged = portfolio + [hedge_opt]
    # 5. Calcula delta neta de la cartera gamma hedgeada
    greeks_total_gamma = portfolio_greeks(portfolio_gamma_hedged, N=N_steps)
    delta_gamma_hedged = greeks_total_gamma['delta']
    # 6. Simula P&L con gamma+delta hedge usando los mismos shocks
    pnl_gamma_delta_hedged = []
    for i in range(len(pnl_binomial)):
        # P&L de la cartera original + gamma hedge
        shocked_portfolio = []
        for opt in portfolio_gamma_hedged:
            key = opt.get('ticker', opt['S'])
            S0 = opt['S']
            iv = implied_volatility_option(opt['market_price'], opt['S'], opt['K'], opt['T'], opt['r'], opt['type'])
            if iv is None:
                iv = 0.2
            r = opt['r']
            T_sim = horizonte_dias if horizonte_dias is not None else opt['T']
            Z = shocks_binomial[key][i] if key in shocks_binomial else np.random.normal(0, 1)
            S_T = S0 * np.exp((r - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
            shocked_opt = opt.copy()
            shocked_opt['S'] = S_T
            shocked_portfolio.append(shocked_opt)
        val = portfolio_value(shocked_portfolio, N=N_steps)
        hedge_pnl = 0
        # Delta hedge sobre la delta neta de la cartera gamma hedgeada
        S0 = portfolio[0]['S']
        delta = delta_gamma_hedged
        Z = shocks_binomial[portfolio[0].get('ticker', portfolio[0]['S'])][i]
        S_T = S0 * np.exp((portfolio[0]['r'] - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
        hedge_pnl += -delta * (S_T - S0)
        pnl_gamma_delta_hedged.append(val - value_binomial + hedge_pnl)
    pnl_gamma_delta_hedged = np.array(pnl_gamma_delta_hedged)
    var_gamma_delta_hedged, es_gamma_delta_hedged = var_es(pnl_gamma_delta_hedged, alpha=0.01)
    print(f"Gamma hedge: qty = {qty_gamma_hedge:.4f} of ATM call (S={hedge_opt['S']}, K={hedge_opt['K']}) covering {gamma_hedge_fraction*100:.0f}% of gamma")
    print(f"Delta after gamma hedge: {delta_gamma_hedged:.4f}")
    print(f"\nVaR after gamma+delta hedge (BINOMIAL, 99%): {var_gamma_delta_hedged:.2f}")
    print(f"ES after gamma+delta hedge (BINOMIAL, 99%): {es_gamma_delta_hedged:.2f}")
    print(f"VaR reduction vs original: {var_binomial - var_gamma_delta_hedged:.2f}")
    print(f"ES reduction vs original: {es_binomial - es_gamma_delta_hedged:.2f}")
    print("="*60)

    # Visualización de la distribución de P&L antes y después de los hedges (BINOMIAL)
    plt.figure(figsize=(14,8))
    plt.hist(pnl_binomial, bins=50, color='skyblue', edgecolor='k', alpha=0.5, density=True, label='Original')
    plt.hist(pnl_binomial_hedged, bins=50, color='orange', edgecolor='k', alpha=0.5, density=True, label='Delta Hedge')
    plt.hist(pnl_gamma_delta_hedged, bins=50, color='green', edgecolor='k', alpha=0.5, density=True, label='Gamma+Delta Hedge')
    # VaR y ES
    var_o, es_o = var_binomial, es_binomial
    var_d, es_d = var_binomial_hedged, es_binomial_hedged
    var_g, es_g = var_gamma_delta_hedged, es_gamma_delta_hedged
    plt.axvline(-var_o, color='blue', linestyle='--', label=f'VaR Original ({-var_o:.0f})')
    plt.axvline(-es_o, color='blue', linestyle=':', label=f'ES Original ({-es_o:.0f})')
    plt.axvline(-var_d, color='orange', linestyle='--', label=f'VaR Delta ({-var_d:.0f})')
    plt.axvline(-es_d, color='orange', linestyle=':', label=f'ES Delta ({-es_d:.0f})')
    plt.axvline(-var_g, color='green', linestyle='--', label=f'VaR Gamma+Delta ({-var_g:.0f})')
    plt.axvline(-es_g, color='green', linestyle=':', label=f'ES Gamma+Delta ({-es_g:.0f})')
    plt.title('P&L Distribution Comparison (BINOMIAL)')
    plt.xlabel('P&L')
    plt.ylabel('Density')
    plt.legend(loc='upper left', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('histogram_compare_binomial.png')
    plt.close()

    # ===========================
    # MONTE CARLO ANALYSIS (using your MC pricing models)
    # ===========================
    print("\n" + "="*60)
    print("OPTION PORTFOLIO SUMMARY USING MONTE CARLO (MC MODELS)")
    print("="*60)
    sim_mc = simulate_portfolio_mc_pricing(portfolio, n_sims=1000, n_steps=50, horizon=horizonte_dias)
    pnl_mc = sim_mc['pnl']
    shocks_mc = sim_mc['shocks']
    var_mc, es_mc = var_es(pnl_mc, alpha=0.01)
    value_mc = sum(price_option_mc(opt, n_sim=10000, n_steps=50) * opt['qty'] for opt in portfolio)
    greeks_total_mc = portfolio_greeks_mc(portfolio, n_sim=10000, n_steps=50)
    for i, opt in enumerate(portfolio, 1):
        price_mc = price_option_mc(opt, n_sim=10000, n_steps=50)
        greeks_mc = option_greeks_mc(opt, n_sim=10000, n_steps=50)
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
    plt.savefig('histogram_pnl_portfolio_mc.png')
    plt.close()

    # ===========================
    # DELTA HEDGING (MC)
    # ===========================
    print("\n" + "="*60)
    print("DELTA HEDGING ANALYSIS (MONTE CARLO)")
    print("="*60)
    # 1. Calcular delta total por subyacente (MC)
    subyacentes_mc = {}
    for opt in portfolio:
        key = opt.get('ticker', opt['S'])
        greeks_mc = option_greeks_mc(opt, n_sim=10000, n_steps=50)
        subyacentes_mc.setdefault(key, {'S0': opt['S'], 'delta': 0})
        subyacentes_mc[key]['delta'] += greeks_mc['delta'] * opt['qty']
    # 2. Simular P&L hedgeado (MC) usando los mismos shocks
    pnl_mc_hedged = []
    for i in range(len(pnl_mc)):
        hedge_pnl = 0
        for key, v in subyacentes_mc.items():
            S0 = v['S0']
            delta = v['delta']
            Z = shocks_mc[key][i]
            # Busca la iv y r de una opción con ese subyacente (puedes tomar la primera que encuentres)
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
    # 1. Calcula gamma neta de la cartera (MC)
    greeks_total_mc = portfolio_greeks_mc(portfolio, n_sim=10000, n_steps=50)
    gamma_cartera_mc = greeks_total_mc['gamma']
    delta_cartera_mc = greeks_total_mc['delta']
    # 2. Define opción de hedge (call europea ATM, mismo S/K/T/r)
    hedge_opt_mc = {
        'type': 'call',
        'style': 'european',
        'S': portfolio[0]['S'],
        'K': portfolio[0]['K'],
        'T': portfolio[0]['T'],
        'r': portfolio[0]['r'],
        'qty': 0,  # se calcula
        'market_price': portfolio[0]['market_price'],
    }
    greeks_hedge_mc = option_greeks_mc(hedge_opt_mc, n_sim=10000, n_steps=50)
    gamma_hedge_mc = greeks_hedge_mc['gamma']
    gamma_hedge_fraction = 0.5  # Porcentaje de gamma a cubrir (50% típico en la práctica)
    # 3. Calcula cantidad necesaria para neutralizar gamma
    qty_gamma_hedge_mc = -gamma_cartera_mc * gamma_hedge_fraction / gamma_hedge_mc if gamma_hedge_mc != 0 else 0
    hedge_opt_mc['qty'] = qty_gamma_hedge_mc
    # 4. Crea nueva cartera con gamma hedge
    portfolio_gamma_hedged_mc = portfolio + [hedge_opt_mc]
    # 5. Calcula delta neta de la cartera gamma hedgeada (MC)
    greeks_total_gamma_mc = portfolio_greeks_mc(portfolio_gamma_hedged_mc, n_sim=10000, n_steps=50)
    delta_gamma_hedged_mc = greeks_total_gamma_mc['delta']
    # 6. Simula P&L con gamma+delta hedge usando los mismos shocks
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
        val = sum(price_option_mc(opt, n_sim=500, n_steps=50) * opt['qty'] for opt in shocked_portfolio)
        hedge_pnl = 0
        # Delta hedge sobre la delta neta de la cartera gamma hedgeada (MC)
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

    # Visualización de la distribución de P&L antes y después de los hedges (MC)
    plt.figure(figsize=(14,8))
    plt.hist(pnl_mc, bins=50, color='skyblue', edgecolor='k', alpha=0.5, density=True, label='Original')
    plt.hist(pnl_mc_hedged, bins=50, color='orange', edgecolor='k', alpha=0.5, density=True, label='Delta Hedge')
    plt.hist(pnl_gamma_delta_hedged_mc, bins=50, color='green', edgecolor='k', alpha=0.5, density=True, label='Gamma+Delta Hedge')
    # VaR y ES
    var_o, es_o = var_mc, es_mc
    var_d, es_d = var_mc_hedged, es_mc_hedged
    var_g, es_g = var_gamma_delta_hedged_mc, es_gamma_delta_hedged_mc
    plt.axvline(-var_o, color='blue', linestyle='--', label=f'VaR Original ({-var_o:.0f})')
    plt.axvline(-es_o, color='blue', linestyle=':', label=f'ES Original ({-es_o:.0f})')
    plt.axvline(-var_d, color='orange', linestyle='--', label=f'VaR Delta ({-var_d:.0f})')
    plt.axvline(-es_d, color='orange', linestyle=':', label=f'ES Delta ({-es_d:.0f})')
    plt.axvline(-var_g, color='green', linestyle='--', label=f'VaR Gamma+Delta ({-var_g:.0f})')
    plt.axvline(-es_g, color='green', linestyle=':', label=f'ES Gamma+Delta ({-es_g:.0f})')
    plt.title('P&L Distribution Comparison (MONTE CARLO)')
    plt.xlabel('P&L')
    plt.ylabel('Density')
    plt.legend(loc='upper left', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('histogram_compare_mc.png')
    plt.close()