import numpy as np
import yfinance as yf
import os
import importlib.util
import matplotlib.pyplot as plt

# --- UTILIDAD: Importar módulos de pricing desde rutas absolutas ---
def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# --- RUTAS A MODELOS DE PRICING ---
base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'quantitative-analysis-finance/option_pricing'))
bs_call_mod = import_from_path('bs_call', os.path.join(base, 'black_scholes_model/european_options/call_implied_volatility.py'))
bs_put_mod = import_from_path('bs_put', os.path.join(base, 'black_scholes_model/european_options/put_implied_volatility.py'))
binomial_mod = import_from_path('binomial', os.path.join(base, 'binomial_model/european_options/binomial.py'))
mc_call_mod = import_from_path('mc_call', os.path.join(base, 'monte-carlo/european_options/call.py'))
mc_put_mod = import_from_path('mc_put', os.path.join(base, 'monte-carlo/european_options/put.py'))
fd_call_mod = import_from_path('fd_call', os.path.join(base, 'finite_difference_method/european_options/call.py'))
fd_put_mod = import_from_path('fd_put', os.path.join(base, 'finite_difference_method/european_options/put.py'))

# --- DEFINICIÓN DE LA CARTERA (puedes modificarla manualmente) ---
portfolio = [
    {'ticker': '^SPX','type': 'call','K': 5955,'T': 0.0712,'contracts': 15},  
]

# --- FUNCIÓN GENERAL DE PRICING ---
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
            raise ValueError('Modelo no soportado')
    except Exception:
        return np.nan

# --- PASO 1: Descargar precios y volatilidad histórica ---
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

# --- PASO 2: Calcular volatilidad implícita si hay precio de mercado ---
for opt in portfolio:
    try:
        user_input = input(f"Introduce el precio de mercado de la opción {opt['ticker']} {opt['type']} (K={opt['K']}, T={opt['T']}) para calcular la volatilidad implícita (deja vacío si no tienes): ")
        if user_input.strip() != '':
            market_price = float(user_input)
            if opt['type'] == 'call':
                implied_vol = bs_call_mod.implied_volatility_newton(market_price, opt['S0'], opt['K'], opt['T'], 0.0421)
            else:
                implied_vol = bs_put_mod.implied_volatility_newton(market_price, opt['S0'], opt['K'], opt['T'], 0.0421)
            if np.isnan(implied_vol) or implied_vol <= 0 or implied_vol > 3:
                opt['implied_vol'] = None
                print(f"  Volatilidad implícita no disponible o inválida para {opt['ticker']} {opt['type']}.")
            else:
                opt['implied_vol'] = implied_vol
                print(f"  Volatilidad implícita calculada: {implied_vol:.2%}")
        else:
            opt['implied_vol'] = None
    except Exception:
        opt['implied_vol'] = None

# --- PASO 3: Mostrar información de la cartera ---
print("\n--- INFORMACIÓN DE LA CARTERA ---")
total_market_value = 0
for opt in portfolio:
    print(f"{opt['ticker']} | Tipo: {opt['type']} | Strike: {opt['K']} | T: {opt['T']} | Contratos: {opt['contracts']}")
    print(f"  Precio spot: {opt['S0']:.2f} | Volatilidad anualizada: {opt['sigma']:.2%}")
    if opt.get('implied_vol') is not None:
        print(f"  Volatilidad implícita: {opt['implied_vol']:.2%}")
        used_vol = opt['implied_vol']
    else:
        print(f"  Volatilidad implícita no disponible, usando histórica: {opt['sigma']:.2%}")
        used_vol = opt['sigma']
    try:
        price = get_option_price('black_scholes', opt['type'], opt['S0'], opt['K'], opt['T'], 0.0421, used_vol)
    except:
        price = np.nan
    print(f"  Precio teórico (Black-Scholes): {price:.2f}")
    total_market_value += opt['contracts'] * price
    print()
print(f"Valor teórico total de la cartera (Black-Scholes): {total_market_value:.2f}\n")

# --- PASO 4: Simulación de escenarios de precios futuros (GBM) ---
N_SIMULATIONS = 1000
r = 0.0421  # tasa libre de riesgo
simulated_prices = {}
for opt in portfolio:
    S0 = opt['S0']
    sigma = opt['sigma']
    T = opt['T']
    Z = np.random.standard_normal(N_SIMULATIONS)
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    simulated_prices[opt['ticker']] = ST

# --- PASO 5: Calcular valor del portafolio en cada escenario y modelo ---
models = ['black_scholes', 'binomial', 'monte_carlo', 'finite_difference']
CONFIDENCE_LEVEL = 0.99
results = {model: [] for model in models}
for i in range(N_SIMULATIONS):
    for model in models:
        port_value = 0
        for opt in portfolio:
            S = simulated_prices[opt['ticker']][i]
            K = opt['K']
            T = opt['T']
            sigma = opt['sigma']
            contracts = opt['contracts']
            price = get_option_price(model, opt['type'], S, K, T, r, sigma)
            port_value += contracts * price
        results[model].append(port_value)

# --- PASO 6: Calcular VaR y ES para cada modelo ---
for model in models:
    pnl = np.array(results[model]) - np.mean(results[model])
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) == 0:
        print(f"Modelo: {model}")
        print("  No hay resultados válidos para este modelo (posible error numérico o parámetros no soportados).\n")
        continue
    var = np.percentile(pnl, (1-CONFIDENCE_LEVEL)*100)
    es = pnl[pnl <= var].mean()
    print(f"Modelo: {model}")
    print(f"  Value at Risk (VaR) al {CONFIDENCE_LEVEL*100:.1f}%: {var:.2f}")
    print(f"  Expected Shortfall (ES) al {CONFIDENCE_LEVEL*100:.1f}%: {es:.2f}\n")

# --- PASO 7: Visualización de resultados ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()
plot_idx = 0
for model in models:
    pnl = np.array(results[model]) - np.mean(results[model])
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) == 0:
        print(f"No se puede graficar el modelo {model} (sin datos válidos).")
        continue
    var = np.percentile(pnl, (1-CONFIDENCE_LEVEL)*100)
    es = pnl[pnl <= var].mean()
    axs[plot_idx].hist(pnl, bins=40, color='skyblue', edgecolor='k', alpha=0.7)
    axs[plot_idx].axvline(var, color='red', linestyle='--', label=f'VaR ({var:.2f})')
    axs[plot_idx].axvline(es, color='orange', linestyle='--', label=f'ES ({es:.2f})')
    axs[plot_idx].set_title(f"Modelo: {model}")
    axs[plot_idx].set_xlabel('P&L simulado')
    axs[plot_idx].set_ylabel('Frecuencia')
    axs[plot_idx].legend()
    plot_idx += 1
plt.tight_layout()
plt.savefig('risk_metrics_results.png')
plt.show()
print('Gráfico guardado como risk_metrics_results.png')
