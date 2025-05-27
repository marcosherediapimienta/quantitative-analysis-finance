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
            'iv': iv
        })
    for i in range(n_sims):
        shocked_portfolio = []
        for p in params:
            Z = np.random.normal(0, 1)
            T_sim = horizon if horizon is not None else p['T']
            S_T = p['S'] * np.exp((p['r'] - 0.5 * p['iv'] ** 2) * T_sim + p['iv'] * np.sqrt(T_sim) * Z)
            shocked_opt = p.copy()
            shocked_opt['S'] = S_T
            shocked_portfolio.append(shocked_opt)
        shocked_val = portfolio_value(shocked_portfolio, N)
        pnl.append(shocked_val - base_val)
    return np.array(pnl)

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
    for _ in range(n_sims):
        shocked_portfolio = []
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
            S_T = opt['S'] * np.exp((opt['r'] - 0.5 * iv ** 2) * T_sim + iv * np.sqrt(T_sim) * Z)
            shocked_opt = opt.copy()
            shocked_opt['S'] = S_T
            shocked_portfolio.append(shocked_opt)
        shocked_val = sum(price_option_mc(opt, n_sim=500, n_steps=n_steps) * opt['qty'] for opt in shocked_portfolio)
        pnl.append(shocked_val - base_val)
    return np.array(pnl)

# ===========================
# Ejemplo de uso principal
# ===========================
if __name__ == "__main__":
    # --- Definición de la cartera de ejemplo ---
    portfolio = [
        {'type': 'call', 'style': 'european', 'S': 5802.82, 'K': 5800, 'T': 0.0849, 'r': 0.0421, 'qty': 10, 'market_price': 152.80},
        {'type': 'put',  'style': 'european', 'S': 5802.82, 'K': 5810, 'T': 0.0849, 'r': 0.0421, 'qty': -5, 'market_price': 91.76}
    ]
    horizonte_dias = 10 / 252  # 15 días de trading
    N_steps = 1000

    # ===========================
    # ANÁLISIS BINOMIAL
    # ===========================
    print("\n" + "="*60)
    print("RESUMEN DE LA CARTERA DE OPCIONES USANDO BINOMIAL")
    print("="*60)
    pnl_binomial = simulate_portfolio(portfolio, n_sims=10000, N=N_steps, horizon=horizonte_dias)
    var_binomial, es_binomial = var_es(pnl_binomial, alpha=0.01)
    value_binomial = portfolio_value(portfolio, N=N_steps)
    greeks_total_binomial = portfolio_greeks(portfolio, N=N_steps)
    for i, opt in enumerate(portfolio, 1):
        price, iv = price_option(opt, N=N_steps)
        greeks = option_greeks(opt, N=N_steps)
        gamma_bs_val = gamma_bs(opt['S'], opt['K'], opt['T'], opt['r'], iv)
        print(f"Opción #{i}:")
        print(f"  Tipo:     {opt['type'].capitalize()}   |  Estilo: {opt['style'].capitalize()}   |  Cantidad: {opt['qty']}")
        print(f"  Spot:     {opt['S']:.2f}   |  Strike: {opt['K']:.2f}   |  Vencimiento (años): {opt['T']:.4f}")
        print(f"  Precio mercado: {opt['market_price']:.2f}   |  Precio modelo: {price:.2f}")
        print(f"  Volatilidad implícita: {iv*100:.2f}%")
        print("  Griegas:")
        for g, v in greeks.items():
            if g == 'gamma':
                print(f"    {g.capitalize():<6}: {v:.8f}")
            else:
                print(f"    {g.capitalize():<6}: {v:.4f}")
        print(f"    Gamma Black-Scholes: {gamma_bs_val:.8f}")
        print("-"*60)
    print("\nGriegas agregadas de la cartera:")
    for g, v in greeks_total_binomial.items():
        print(f"  {g.capitalize():<6}: {v:.4f}")
    print(f"\nValor actual de la cartera: {value_binomial:.2f}")
    print(f"\nVaR histórico (simulación MC, BINOMIAL, horizonte {horizonte_dias*252:.0f} días, 99%): {var_binomial:.2f}")
    print(f"ES histórico (simulación MC, BINOMIAL, 99%): {es_binomial:.2f}")
    print("="*60)
    plt.figure(figsize=(14,8))
    plt.hist(pnl_binomial, bins=50, color='skyblue', edgecolor='k', alpha=0.7, density=True)
    plt.axvline(-var_binomial, color='red', linestyle='--', label=f'VaR ({-var_binomial:.2f})')
    plt.axvline(-es_binomial, color='orange', linestyle='--', label=f'ES ({-es_binomial:.2f})')
    plt.title('Distribución de P&L simulados de la cartera (BINOMIAL)')
    plt.xlabel('P&L')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('histograma_pnl_cartera_binomial.png')
    plt.close()

    # ===========================
    # ANÁLISIS MONTE CARLO (usando tus modelos MC de pricing)
    # ===========================
    print("\n" + "="*60)
    print("RESUMEN DE LA CARTERA DE OPCIONES USANDO MONTE CARLO (MODELOS MC)")
    print("="*60)
    pnl_mc = simulate_portfolio_mc_pricing(portfolio, n_sims=1000, n_steps=50, horizon=horizonte_dias)
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
        print(f"Opción #{i} (MC):")
        print(f"  Tipo:     {opt['type'].capitalize()}   |  Estilo: {opt['style'].capitalize()}   |  Cantidad: {opt['qty']}")
        print(f"  Spot:     {opt['S']:.2f}   |  Strike: {opt['K']:.2f}   |  Vencimiento (años): {opt['T']:.4f}")
        print(f"  Precio mercado: {opt['market_price']:.2f}   |  Precio modelo (MC): {price_mc:.2f}")
        print(f"  Volatilidad usada: {iv_mc*100:.2f}%")
        print("  Griegas MC:")
        for g, v in greeks_mc.items():
            if g == 'gamma':
                print(f"    {g.capitalize():<6}: {v:.8f}")
            else:
                print(f"    {g.capitalize():<6}: {v:.4f}")
        print("-"*60)
    print("\nGriegas agregadas de la cartera (MC):")
    for g, v in greeks_total_mc.items():
        print(f"  {g.capitalize():<6}: {v:.4f}")
    print(f"\nValor actual de la cartera (MC): {value_mc:.2f}")
    print(f"\nVaR histórico (simulación MC, PRICING MC, horizonte {horizonte_dias*252:.0f} días, 99%): {var_mc:.2f}")
    print(f"ES histórico (simulación MC, PRICING MC, 99%): {es_mc:.2f}")
    print("="*60)
    plt.figure(figsize=(14,8))
    plt.hist(pnl_mc, bins=50, color='lightgreen', edgecolor='k', alpha=0.7, density=True)
    plt.axvline(-var_mc, color='red', linestyle='--', label=f'VaR MC ({-var_mc:.2f})')
    plt.axvline(-es_mc, color='orange', linestyle='--', label=f'ES MC ({-es_mc:.2f})')
    plt.title('Distribución de P&L simulados de la cartera (PRICING MC)')
    plt.xlabel('P&L')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('histograma_pnl_cartera_mc.png')
    plt.close()