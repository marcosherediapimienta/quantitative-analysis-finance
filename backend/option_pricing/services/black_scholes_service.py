import numpy as np
from scipy.stats import norm
from option_pricing.models.option import Option
from option_pricing.models.greeks import Greeks

class BlackScholesService:
    @staticmethod
    def price(option: Option):
        S, K, T, r, sigma = option.spot, option.strike, option.maturity, option.rate, option.volatility
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option.type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def greeks(option: Option):
        S, K, T, r, sigma = option.spot, option.strike, option.maturity, option.rate, option.volatility
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        delta = norm.cdf(d1) if option.type == 'call' else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option.type == 'call' else -d2))
        rho = K * T * np.exp(-r * T) * (norm.cdf(d2) if option.type == 'call' else -norm.cdf(-d2))
        return Greeks(delta, gamma, vega, theta, rho)
