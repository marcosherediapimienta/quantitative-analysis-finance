from typing import Callable, Dict, Optional

from scipy.optimize import brentq

from models.bs import BlackScholesPricer
from tools.analytics.numerics import ImpliedVolatilityConfig
from tools.analytics.options import OptionType

class ImpliedVolatilitySolver:
    _bs_price: Dict[OptionType, Callable[[float, float, float, float, float], float]] = {
        OptionType.CALL: BlackScholesPricer.call_price,
        OptionType.PUT: BlackScholesPricer.put_price,
    }

    def __init__(self, config: Optional[ImpliedVolatilityConfig] = None) -> None:
        self.config = config or ImpliedVolatilityConfig()

    def solve(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        option_type: OptionType,
    ) -> Optional[float]:
        sigma = self.config.initial_sigma
        for _ in range(self.config.max_iterations):
            price = self._bs_price[option_type](spot, strike, time_to_maturity, risk_free_rate, sigma)
            vega = BlackScholesPricer.vega(spot, strike, time_to_maturity, risk_free_rate, sigma)
            diff = price - market_price

            if abs(vega) < self.config.min_abs_vega:
                break
            if abs(diff) < self.config.tolerance:
                return float(sigma)

            sigma = sigma - diff / vega
            if sigma <= self.config.sigma_min:
                sigma = self.config.sigma_min

        def objective(sigma_: float) -> float:
            return self._bs_price[option_type](spot, strike, time_to_maturity, risk_free_rate, sigma_) - market_price

        try:
            return float(
                brentq(
                    objective,
                    self.config.sigma_min,
                    self.config.sigma_max,
                    xtol=self.config.tolerance,
                    maxiter=self.config.max_iterations,
                )
            )
        except Exception:
            return None
