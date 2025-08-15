import numpy as np
import matplotlib.pyplot as plt
from option_pricing.models.option import Option
from option_pricing.services.binomial_service import BinomialService
from option_pricing.services.black_scholes_service import BlackScholesService
from option_pricing.services.monte_carlo_service import MonteCarloService
import os

# --- HISTOGRAMAS DE PRECIOS ---
def plot_binomial_price_histogram(option: Option, N=1000, n_sim=1000, save_path=None):
    prices = [BinomialService.price(option, N) for _ in range(n_sim)]
    plt.figure(figsize=(8,5))
    plt.hist(prices, bins=30, alpha=0.7)
    plt.xlabel('Precio opción')
    plt.ylabel('Frecuencia')
    plt.title('Histograma precios - Binomial')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f'Histograma guardado en {save_path}')
    else:
        plt.show()
    plt.close()

def plot_bs_price_histogram(option: Option, n_sim=1000, save_path=None):
    prices = [BlackScholesService.price(option) for _ in range(n_sim)]
    plt.figure(figsize=(8,5))
    plt.hist(prices, bins=30, alpha=0.7)
    plt.xlabel('Precio opción')
    plt.ylabel('Frecuencia')
    plt.title('Histograma precios - Black-Scholes')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f'Histograma guardado en {save_path}')
    else:
        plt.show()
    plt.close()

def plot_mc_price_histogram(option: Option, n_sim=1000, seed=42, save_path=None):
    prices = [MonteCarloService.price(option, n_sim=1, seed=seed) for _ in range(n_sim)]
    plt.figure(figsize=(8,5))
    plt.hist(prices, bins=30, alpha=0.7)
    plt.xlabel('Precio opción')
    plt.ylabel('Frecuencia')
    plt.title('Histograma precios - Monte Carlo')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f'Histograma guardado en {save_path}')
    else:
        plt.show()
    plt.close()

# --- HISTOGRAMAS DE PNL DE PORTFOLIO ---
def plot_binomial_pnl_portfolio_histogram(options, N=1000, n_sim=1000, save_path=None):
    pnls = [sum(BinomialService.price(opt, N) for opt in options) for _ in range(n_sim)]
    plt.figure(figsize=(8,5))
    plt.hist(pnls, bins=30, alpha=0.7)
    plt.xlabel('Valor portfolio')
    plt.ylabel('Frecuencia')
    plt.title('Histograma PnL Portfolio - Binomial')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f'Histograma PnL guardado en {save_path}')
    else:
        plt.show()
    plt.close()

def plot_bs_pnl_portfolio_histogram(options, n_sim=1000, save_path=None):
    pnls = [sum(BlackScholesService.price(opt) for opt in options) for _ in range(n_sim)]
    plt.figure(figsize=(8,5))
    plt.hist(pnls, bins=30, alpha=0.7)
    plt.xlabel('Valor portfolio')
    plt.ylabel('Frecuencia')
    plt.title('Histograma PnL Portfolio - Black-Scholes')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f'Histograma PnL guardado en {save_path}')
    else:
        plt.show()
    plt.close()

def plot_mc_pnl_portfolio_histogram(options, n_sim=1000, seed=42, save_path=None):
    pnls = [sum(MonteCarloService.price(opt, n_sim=1, seed=seed) for opt in options) for _ in range(n_sim)]
    plt.figure(figsize=(8,5))
    plt.hist(pnls, bins=30, alpha=0.7)
    plt.xlabel('Valor portfolio')
    plt.ylabel('Frecuencia')
    plt.title('Histograma PnL Portfolio - Monte Carlo')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f'Histograma PnL guardado en {save_path}')
    else:
        plt.show()
    plt.close()

# --- SENSIBILIDAD ---
def plot_binomial_sensitivity(option: Option, param: str, N=1000, save_path=None):
    steps = 20
    if param == 'spot':
        values = np.linspace(option.spot * 0.5, option.spot * 1.5, steps)
        prices = [BinomialService.price(Option(option.type, option.style, v, option.strike, option.maturity, option.volatility, option.rate), N) for v in values]
    elif param == 'r':
        values = np.linspace(0.0, option.rate * 2, steps)
        prices = [BinomialService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity, option.volatility, v), N) for v in values]
    elif param == 'vol':
        values = np.linspace(0.01, option.volatility * 2, steps)
        prices = [BinomialService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity, v, option.rate), N) for v in values]
    else:
        raise ValueError('Parámetro no soportado')
    plt.figure(figsize=(8,5))
    plt.plot(values, prices, marker='o')
    plt.xlabel(param)
    plt.ylabel('Precio opción')
    plt.title(f'Sensibilidad {param} - Binomial')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f'Gráfico guardado en {save_path}')
    else:
        plt.show()
    plt.close()

def plot_bs_sensitivity(option: Option, param: str, save_path=None):
    steps = 20
    if param == 'spot':
        values = np.linspace(option.spot * 0.5, option.spot * 1.5, steps)
        prices = [BlackScholesService.price(Option(option.type, option.style, v, option.strike, option.maturity, option.volatility, option.rate)) for v in values]
    elif param == 'r':
        values = np.linspace(0.0, option.rate * 2, steps)
        prices = [BlackScholesService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity, option.volatility, v)) for v in values]
    elif param == 'vol':
        values = np.linspace(0.01, option.volatility * 2, steps)
        prices = [BlackScholesService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity, v, option.rate)) for v in values]
    else:
        raise ValueError('Parámetro no soportado')
    plt.figure(figsize=(8,5))
    plt.plot(values, prices, marker='o')
    plt.xlabel(param)
    plt.ylabel('Precio opción')
    plt.title(f'Sensibilidad {param} - Black-Scholes')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f'Gráfico guardado en {save_path}')
    else:
        plt.show()
    plt.close()

def plot_mc_sensitivity(option: Option, param: str, n_sim=10000, seed=42, save_path=None):
    steps = 20
    if param == 'spot':
        values = np.linspace(option.spot * 0.5, option.spot * 1.5, steps)
        prices = [MonteCarloService.price(Option(option.type, option.style, v, option.strike, option.maturity, option.volatility, option.rate), n_sim=n_sim, seed=seed) for v in values]
    elif param == 'r':
        values = np.linspace(0.0, option.rate * 2, steps)
        prices = [MonteCarloService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity, option.volatility, v), n_sim=n_sim, seed=seed) for v in values]
    elif param == 'vol':
        values = np.linspace(0.01, option.volatility * 2, steps)
        prices = [MonteCarloService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity, v, option.rate), n_sim=n_sim, seed=seed) for v in values]
    else:
        raise ValueError('Parámetro no soportado')
    plt.figure(figsize=(8,5))
    plt.plot(values, prices, marker='o')
    plt.xlabel(param)
    plt.ylabel('Precio opción')
    plt.title(f'Sensibilidad {param} - Monte Carlo')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f'Gráfico guardado en {save_path}')
    else:
        plt.show()
    plt.close()
