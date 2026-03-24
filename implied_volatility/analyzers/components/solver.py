from typing import Any, Dict

from models.black_scholes import analytical_greeks, price_european_black_scholes

from ...tools.config import (
    DEFAULT_BISECTION_MAX_ITERATIONS,
    DEFAULT_INITIAL_VOLATILITY,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_METHOD,
    DEFAULT_TOLERANCE,
    DEFAULT_USE_FALLBACK,
    MAX_VOLATILITY,
    MIN_VEGA,
    MIN_VOLATILITY,
)
from ...tools.types import IVContract, SolverMethod


class ImpliedVolatilitySolver:
    def __init__(
        self,
        method: SolverMethod = DEFAULT_METHOD,
        tolerance: float = DEFAULT_TOLERANCE,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        bisection_max_iterations: int = DEFAULT_BISECTION_MAX_ITERATIONS,
        initial_volatility: float = DEFAULT_INITIAL_VOLATILITY,
        min_volatility: float = MIN_VOLATILITY,
        max_volatility: float = MAX_VOLATILITY,
        min_vega: float = MIN_VEGA,
        use_fallback: bool = DEFAULT_USE_FALLBACK,
    ) -> None:
        if method not in ("newton", "bisection"):
            raise ValueError("method debe ser 'newton' o 'bisection'.")
        self.method = method
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.bisection_max_iterations = bisection_max_iterations
        self.initial_volatility = initial_volatility
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        self.min_vega = min_vega
        self.use_fallback = use_fallback

    def _result(
        self,
        implied_volatility: float,
        model_price: float,
        iterations: int,
        converged: bool,
        method: str,
    ) -> Dict[str, Any]:
        return {
            "implied_volatility": implied_volatility,
            "model_price": model_price,
            "iterations": iterations,
            "converged": converged,
            "method": method,
        }

    def _to_bs_contract(self, contract: IVContract, volatility: float) -> Dict[str, Any]:
        return {
            "spot": contract["spot"],
            "strike": contract["strike"],
            "time_to_maturity": contract["time_to_maturity"],
            "risk_free_rate": contract["risk_free_rate"],
            "volatility": max(volatility, 1e-8),
            "option_type": contract["option_type"],
        }

    def _price(self, contract: IVContract, volatility: float) -> float:
        return float(price_european_black_scholes(self._to_bs_contract(contract, volatility)))

    def _vega(self, contract: IVContract, volatility: float) -> float:
        greeks = analytical_greeks(self._to_bs_contract(contract, volatility))
        return float(greeks["vega"])

    def _solve_bisection(self, contract: IVContract, max_iterations: int, method_label: str) -> Dict[str, Any]:
        low = self.min_volatility
        high = self.max_volatility
        market_price = contract["market_price"]

        low_price = self._price(contract, low)
        high_price = self._price(contract, high)
        low_diff = low_price - market_price
        high_diff = high_price - market_price

        if low_diff * high_diff > 0:
            raise ValueError(
                "No se puede acotar la volatilidad implicita con los limites actuales. "
                "Ajusta min_volatility/max_volatility o revisa market_price."
            )

        mid = low
        mid_price = low_price

        for iteration in range(1, max_iterations + 1):
            mid = 0.5 * (low + high)
            mid_price = self._price(contract, mid)
            diff = mid_price - market_price

            if abs(diff) <= self.tolerance:
                return self._result(mid, mid_price, iteration, True, method_label)

            if low_diff * diff <= 0:
                high = mid
                high_diff = diff
            else:
                low = mid
                low_diff = diff

        return self._result(mid, mid_price, max_iterations, False, method_label)

    def _solve_newton(self, contract: IVContract) -> Dict[str, Any]:
        market_price = contract["market_price"]
        sigma = min(max(self.initial_volatility, self.min_volatility), self.max_volatility)

        for iteration in range(1, self.max_iterations + 1):
            model_price = self._price(contract, sigma)
            diff = model_price - market_price

            if abs(diff) <= self.tolerance:
                return self._result(sigma, model_price, iteration, True, "newton")

            vega = self._vega(contract, sigma)
            if abs(vega) <= self.min_vega:
                break

            sigma = sigma - diff / vega
            sigma = min(max(sigma, self.min_volatility), self.max_volatility)

        if self.use_fallback:
            return self._solve_bisection(
                contract=contract,
                max_iterations=self.bisection_max_iterations,
                method_label="bisection_fallback",
            )

        return self._result(
            sigma,
            self._price(contract, sigma),
            self.max_iterations,
            False,
            "newton",
        )

    def solve(self, contract: IVContract) -> Dict[str, Any]:
        solvers = {
            "newton": self._solve_newton,
            "bisection": lambda c: self._solve_bisection(
                contract=c,
                max_iterations=self.max_iterations,
                method_label="bisection",
            ),
        }
        return solvers[self.method](contract)
