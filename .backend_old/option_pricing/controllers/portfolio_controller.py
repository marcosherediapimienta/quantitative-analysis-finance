from backend.option_pricing.models.portfolio import Portfolio
from backend.option_pricing.services.binomial_service import BinomialService
from backend.option_pricing.services.black_scholes_service import BlackScholesService
from backend.option_pricing.services.monte_carlo_service import MonteCarloService


class PortfolioController:
    def get_portfolio_value(self, portfolio_data, model: str = 'black_scholes', n_sim=None, seed=None):
        portfolio = Portfolio(portfolio_data)
        if model == 'monte_carlo':
            return sum(MonteCarloService.price(option, n_sim=n_sim, seed=seed) for option in portfolio.options)
        model_map = {
            'binomial': BinomialService.price,
            'black_scholes': BlackScholesService.price
        }
        price_func = model_map.get(model)
        if not price_func:
            raise ValueError(f"Modelo '{model}' no soportado")
        return sum(price_func(option) for option in portfolio.options)

    def get_portfolio_greeks(self, portfolio_data, model: str = 'black_scholes', n_sim=None, seed=None):
        portfolio = Portfolio(portfolio_data)
        if model == 'monte_carlo':
            return [MonteCarloService.greeks(option, n_sim=n_sim, seed=seed) for option in portfolio.options]
        model_map = {
            'binomial': BinomialService.greeks,
            'black_scholes': BlackScholesService.greeks
        }
        greeks_func = model_map.get(model)
        if not greeks_func:
            raise ValueError(f"Modelo '{model}' no soportado")
        return [greeks_func(option) for option in portfolio.options]
