"""
pricing_from_yahoo_option.py

Ejemplo: Toma una opción real de Yahoo Finance y la procesa con el backend (pricing, griegas, visualización).
"""
import yfinance as yf
import numpy as np
from option_pricing.models.option import Option
from option_pricing.services.black_scholes_service import BlackScholesService
from option_pricing.services.binomial_service import BinomialService
from option_pricing.services.monte_carlo_service import MonteCarloService
from option_pricing.utils.plot_utils import (
    plot_binomial_price_histogram,
    plot_bs_price_histogram,
    plot_mc_price_histogram,
    plot_binomial_sensitivity,
    plot_bs_sensitivity,
    plot_mc_sensitivity
)
from option_pricing.data.data_options import get_expirations, get_option_chain, get_us_risk_free_rate
from option_pricing.utils.math_utils import implied_volatility

# --- Configuración interactiva ---
ticker = input("Ticker (ej: AAPL): ").upper().strip()
expirations = get_expirations(ticker)
if not expirations:
    raise ValueError(f"No hay expiraciones para {ticker}")
print('¿Quieres analizar una opción CALL o PUT?')
opt_type = input('Escribe CALL o PUT: ').strip().lower()
print('Expirations:')
for idx, exp in enumerate(expirations):
    print(f"  {idx+1}. {exp}")
exp_idx = int(input("Selecciona número de expiración: ")) - 1
expiration = expirations[exp_idx]
calls, puts = get_option_chain(ticker, expiration)
if opt_type == 'put':
    row_idx = int(input('Selecciona número de opción put (fila): '))
    row = puts.iloc[row_idx]
    tipo = 'put'
else:
    row_idx = int(input('Selecciona número de opción call (fila): '))
    row = calls.iloc[row_idx]
    tipo = 'call'
strike = float(row['strike'])
last_price = float(row['lastPrice'])
spot_data = yf.download(ticker, period='1d')
if spot_data.empty or 'Close' not in spot_data.columns:
    raise ValueError(f"No se pudo obtener el precio spot para {ticker}")
spot = float(spot_data['Close'].iloc[-1])
default_r = get_us_risk_free_rate() or 0.05
rate = float(input(f"Tasa libre de riesgo (default {default_r:.4f}): ") or default_r)
maturity = (np.datetime64(expiration) - np.datetime64('today')) / np.timedelta64(1, 'D') / 365.0
vol = implied_volatility(Option(tipo, 'european', spot, strike, maturity, 0.2, rate), last_price)

option = Option(tipo, 'european', spot, strike, maturity, vol, rate)
print(f"Option params: tipo={tipo}, spot={spot}, strike={strike}, maturity={maturity:.3f}, vol={vol:.3f}, rate={rate:.3f}")

# --- Pricing y visualizaciones ---
plot_binomial_price_histogram(option, N=1000, n_sim=500, save_path=f'option_pricing/visualizations/{ticker}_binomial_hist.png')
plot_binomial_sensitivity(option, 'spot', N=1000, save_path=f'option_pricing/visualizations/{ticker}_binomial_sens_spot.png')
plot_bs_price_histogram(option, n_sim=500, save_path=f'option_pricing/visualizations/{ticker}_bs_hist.png')
plot_bs_sensitivity(option, 'spot', save_path=f'option_pricing/visualizations/{ticker}_bs_sens_spot.png')
plot_mc_price_histogram(option, n_sim=500, seed=42, save_path=f'option_pricing/visualizations/{ticker}_mc_hist.png')
plot_mc_sensitivity(option, 'spot', n_sim=500, seed=42, save_path=f'option_pricing/visualizations/{ticker}_mc_sens_spot.png')

print('Precio Binomial:', BinomialService.price(option, N=1000))
print('Precio Black-Scholes:', BlackScholesService.price(option))
print('Precio Monte Carlo:', MonteCarloService.price(option, n_sim=10000, seed=42))
print('Precio mercado (lastPrice):', last_price)
