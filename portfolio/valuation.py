from typing import Dict, List, Optional

from data.yahoo_data import HistoricalVolatilityEstimator
from models import (
    AmericanLongstaffSchwartzPricer,
    EuropeanMonteCarloPricer,
    OptionContract,
    OptionType,
)
from tools.analytics import ImpliedVolatilitySolver, MonteCarloGreeksCalculator
from tools.config import DEFAULT_FALLBACK_VOLATILITY, DEFAULT_MC_SEED

def solve_option_implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
) -> Optional[float]:
    return ImpliedVolatilitySolver().solve(
        market_price=market_price,
        spot=S,
        strike=K,
        time_to_maturity=T,
        risk_free_rate=r,
        option_type=OptionType.from_string(option_type),
    )


def _resolve_volatility(option_data: Dict) -> float:
    sigma = solve_option_implied_volatility(
        option_data["market_price"],
        option_data["S"],
        option_data["K"],
        option_data["T"],
        option_data["r"],
        option_data["type"],
    )
    if sigma is not None:
        return sigma
    ticker = option_data.get("ticker")
    if ticker is None:
        return DEFAULT_FALLBACK_VOLATILITY
    return HistoricalVolatilityEstimator().estimate(ticker)


def price_option_monte_carlo(option_data: Dict, n_sim: int = 1000, n_steps: int = 50) -> float:
    sigma = _resolve_volatility(option_data)
    contract = OptionContract(
        spot=option_data["S"],
        strike=option_data["K"],
        time_to_maturity=option_data["T"],
        risk_free_rate=option_data["r"],
        volatility=sigma,
        option_type=OptionType.from_string(option_data["type"]),
    )
    if option_data["style"] == "american":
        return AmericanLongstaffSchwartzPricer().price(
            contract=contract,
            num_simulations=n_sim,
            num_steps=n_steps,
            seed=DEFAULT_MC_SEED,
            regression_type=option_data.get("regression_type", "quadratic"),
        )
    return EuropeanMonteCarloPricer().price(
        contract=contract,
        num_simulations=n_sim,
        seed=DEFAULT_MC_SEED,
    )


def option_greeks_monte_carlo(option_data: Dict, n_sim: int = 1000, n_steps: int = 50) -> Dict[str, float]:
    sigma = _resolve_volatility(option_data)
    opt_type = option_data["type"]
    style = option_data["style"]

    if style == "american":
        calculator = MonteCarloGreeksCalculator(
            pricing_fn=lambda **kwargs: AmericanLongstaffSchwartzPricer().price(
                contract=OptionContract(
                    spot=kwargs["S"],
                    strike=kwargs["K"],
                    time_to_maturity=kwargs["T"],
                    risk_free_rate=kwargs["r"],
                    volatility=kwargs["sigma"],
                    option_type=OptionType.from_string(kwargs["option_type"]),
                ),
                num_simulations=kwargs["n_sim"],
                num_steps=kwargs["n_steps"],
                seed=kwargs["seed"],
                regression_type=kwargs.get("regression_type", "quadratic"),
            )
        )
        return calculator.calculate(
            S=option_data["S"],
            K=option_data["K"],
            T=option_data["T"],
            r=option_data["r"],
            sigma=sigma,
            n_sim=n_sim,
            n_steps=n_steps,
            option_type=opt_type,
            seed=option_data.get("seed"),
            regression_type=option_data.get("regression_type", "quadratic"),
        )

    calculator = MonteCarloGreeksCalculator(
        pricing_fn=lambda **kwargs: EuropeanMonteCarloPricer().price(
            contract=OptionContract(
                spot=kwargs["S"],
                strike=kwargs["K"],
                time_to_maturity=kwargs["T"],
                risk_free_rate=kwargs["r"],
                volatility=kwargs["sigma"],
                option_type=OptionType.from_string(kwargs["option_type"]),
            ),
            num_simulations=kwargs["n_sim"],
            seed=kwargs["seed"],
        )
    )
    return calculator.calculate(
        S=option_data["S"],
        K=option_data["K"],
        T=option_data["T"],
        r=option_data["r"],
        sigma=sigma,
        n_sim=n_sim,
        option_type=opt_type,
        seed=option_data.get("seed"),
    )


def portfolio_greeks_monte_carlo(portfolio: List[Dict], n_sim: int = 1000, n_steps: int = 50) -> Dict[str, float]:
    total = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    for option_data in portfolio:
        greeks = option_greeks_monte_carlo(option_data, n_sim, n_steps)
        for greek in total:
            total[greek] += greeks[greek] * option_data["qty"]
    return total
