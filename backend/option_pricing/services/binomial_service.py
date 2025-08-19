import numpy as np
from backend.option_pricing.models.option import Option
from backend.option_pricing.models.greeks import Greeks

class BinomialService:
    @staticmethod
    def price(option: Option, N=1000):
        dt = option.maturity / N
        u = np.exp(option.volatility * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(option.rate * dt) - d) / (u - d)
        discount = np.exp(-option.rate * dt)
        j = np.arange(N + 1)
        ST = option.spot * (u ** j) * (d ** (N - j))
        if option.type == 'call':
            payoff = np.maximum(ST - option.strike, 0)
        else:
            payoff = np.maximum(option.strike - ST, 0)
        for _ in range(N):
            payoff = discount * (p * payoff[1:] + (1 - p) * payoff[:-1])
        return payoff[0]

    @staticmethod
    def greeks(option: Option, N=1000, h=None):
        if h is None:
            h = max(0.01 * option.spot, 0.01)
        price_up = BinomialService.price(Option(option.type, option.style, option.spot + h, option.strike, option.maturity, option.volatility, option.rate), N)
        price_down = BinomialService.price(Option(option.type, option.style, option.spot - h, option.strike, option.maturity, option.volatility, option.rate), N)
        price = BinomialService.price(option, N)
        delta = (price_up - price_down) / (2 * h)
        gamma = (price_up - 2 * price + price_down) / (h ** 2)
        vega_bump = 0.01
        price_vega_up = BinomialService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity, option.volatility + vega_bump, option.rate), N)
        price_vega_down = BinomialService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity, option.volatility - vega_bump, option.rate), N)
        vega = (price_vega_up - price_vega_down) / (2 * vega_bump)
        theta_bump = 0.01
        price_theta = BinomialService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity - theta_bump, option.volatility, option.rate), N)
        theta = (price_theta - price) / theta_bump
        rho_bump = 0.01
        price_rho_up = BinomialService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity, option.volatility, option.rate + rho_bump), N)
        price_rho_down = BinomialService.price(Option(option.type, option.style, option.spot, option.strike, option.maturity, option.volatility, option.rate - rho_bump), N)
        rho = (price_rho_up - price_rho_down) / (2 * rho_bump)
        return Greeks(delta, gamma, vega, theta, rho)
