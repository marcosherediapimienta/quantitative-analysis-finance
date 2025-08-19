import numpy as np
from ..models.option import Option
from ..models.greeks import Greeks

class MonteCarloService:
    @staticmethod
    def price(option: Option, n_sim=10000, seed=None):
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(n_sim)
        ST = option.spot * np.exp((option.rate - 0.5 * option.volatility**2) * option.maturity + option.volatility * np.sqrt(option.maturity) * Z)
        if option.type == 'call':
            payoff = np.maximum(ST - option.strike, 0)
        else:
            payoff = np.maximum(option.strike - ST, 0)
        price = np.exp(-option.rate * option.maturity) * np.mean(payoff)
        return price

    @staticmethod
    def greeks(option: Option, n_sim=10000, h=None, seed=None):
        if h is None:
            h = max(0.01 * option.spot, 0.01)
        price_up = MonteCarloService.price(Option(option.type, option.style, option.spot + h, option.strike, option.maturity, option.volatility, option.rate), n_sim, seed)
        price_down = MonteCarloService.price(Option(option.type, option.style, option.spot - h, option.strike, option.maturity, option.volatility, option.rate), n_sim, seed)
        price = MonteCarloService.price(option, n_sim, seed)
        delta = (price_up - price_down) / (2 * h)
        gamma = (price_up - 2 * price + price_down) / (h ** 2)
        vega_bump = 0.01
        price_vega_up = MonteCarloService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity, option.volatility + vega_bump, option.rate), n_sim, seed)
        price_vega_down = MonteCarloService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity, option.volatility - vega_bump, option.rate), n_sim, seed)
        vega = (price_vega_up - price_vega_down) / (2 * vega_bump)
        theta_bump = 0.01
        price_theta = MonteCarloService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity - theta_bump, option.volatility, option.rate), n_sim, seed)
        theta = (price_theta - price) / theta_bump
        rho_bump = 0.01
        price_rho_up = MonteCarloService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity, option.volatility, option.rate + rho_bump), n_sim, seed)
        price_rho_down = MonteCarloService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity, option.volatility, option.rate - rho_bump), n_sim, seed)
        rho = (price_rho_up - price_rho_down) / (2 * rho_bump)
        return Greeks(delta, gamma, vega, theta, rho)
