from option_pricing.models.option import Option
from option_pricing.models.portfolio import Portfolio
from option_pricing.services.binomial_service import BinomialService
from option_pricing.services.black_scholes_service import BlackScholesService
from option_pricing.services.monte_carlo_service import MonteCarloService

class PortfolioAnalysis:
    @staticmethod
    def value(portfolio: Portfolio, model: str = 'black_scholes'):
        model_map = {
            'binomial': BinomialService.price,
            'black_scholes': BlackScholesService.price,
            'monte_carlo': MonteCarloService.price
        }
        price_func = model_map.get(model)
        if not price_func:
            raise ValueError(f"Modelo '{model}' no soportado")
        return sum(price_func(option) for option in portfolio.options)

    @staticmethod
    def greeks(portfolio: Portfolio, model: str = 'black_scholes'):
        model_map = {
            'binomial': BinomialService.greeks,
            'black_scholes': BlackScholesService.greeks,
            'monte_carlo': MonteCarloService.greeks
        }
        greeks_func = model_map.get(model)
        if not greeks_func:
            raise ValueError(f"Modelo '{model}' no soportado")
        return [greeks_func(option) for option in portfolio.options]
