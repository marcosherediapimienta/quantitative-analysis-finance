from typing import Callable, Dict, Optional

from ..tools.config import DEFAULT_EXERCISE_STYLE, DEFAULT_SCHEME
from ..tools.types import BumpConfig, Contract, ExerciseStyle, FactorScheme
from .components.american import price_american_binomial
from .components.european import price_european_binomial
from .components.greeks import finite_difference_greeks

_PRICERS: Dict[ExerciseStyle, Callable[..., float]] = {
    "european": price_european_binomial,
    "american": price_american_binomial,
}


class CRRAnalyzer:
    def __init__(
        self,
        exercise_style: ExerciseStyle = DEFAULT_EXERCISE_STYLE,
        scheme: FactorScheme = DEFAULT_SCHEME,
        bumps: Optional[BumpConfig] = None,
    ) -> None:
        self.exercise_style = exercise_style
        self.scheme = scheme
        self.bumps: BumpConfig = bumps or {}

    def price(self, contract: Contract) -> float:
        return _PRICERS[self.exercise_style](contract=contract, scheme=self.scheme)

    def greeks(self, contract: Contract) -> Dict[str, float]:
        metrics = finite_difference_greeks(pricer=self.price, contract=contract, bumps=self.bumps)
        metrics.pop("price", None)
        return metrics

    def analyze(self, contract: Contract) -> Dict[str, float]:
        return finite_difference_greeks(pricer=self.price, contract=contract, bumps=self.bumps)
