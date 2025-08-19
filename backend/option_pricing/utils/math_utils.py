import numpy as np
from scipy.stats import norm
from ..models.option import Option
from ..services.black_scholes_service import BlackScholesService

def calculate_discount_factor(rate, time):
    return 1 / (1 + rate * time)

def implied_volatility(option: Option, market_price: float, tol=1e-6, max_iter=100):
    # Newton-Raphson para volatilidad implícita
    sigma = 0.2  # Valor inicial
    for i in range(max_iter):
        option.volatility = sigma
        price = BlackScholesService.price(option)
        d1 = (np.log(option.spot / option.strike) + (option.rate + 0.5 * sigma ** 2) * option.maturity) / (sigma * np.sqrt(option.maturity))
        vega = option.spot * norm.pdf(d1) * np.sqrt(option.maturity)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega if vega > 1e-8 else 0.01
        sigma = max(sigma, 1e-4)
    raise RuntimeError('No converge la volatilidad implícita')
