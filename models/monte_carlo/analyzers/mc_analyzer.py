from typing import Dict, Optional

from ..tools.config import (
    DEFAULT_EXERCISE_STYLE,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_NUM_STEPS,
    DEFAULT_REGRESSION_TYPE,
)
from ..tools.types import BumpConfig, Contract, ExerciseStyle, RegressionType
from .components.american import price_american_monte_carlo
from .components.european import price_european_monte_carlo
from .components.greeks import finite_difference_greeks


class MCAnalyzer:
    def __init__(
        self,
        exercise_style: ExerciseStyle = DEFAULT_EXERCISE_STYLE,
        num_simulations: int = DEFAULT_NUM_SIMULATIONS,
        num_steps: int = DEFAULT_NUM_STEPS,
        seed: Optional[int] = None,
        regression_type: RegressionType = DEFAULT_REGRESSION_TYPE,
        bumps: Optional[BumpConfig] = None,
    ) -> None:
        self.exercise_style = exercise_style
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.seed = seed
        self.regression_type = regression_type
        self.bumps: BumpConfig = bumps or {}

    def price(self, contract: Contract) -> float:
        if self.exercise_style == "european":
            return price_european_monte_carlo(
                contract=contract,
                num_simulations=self.num_simulations,
                seed=self.seed,
            )
        return price_american_monte_carlo(
            contract=contract,
            num_simulations=self.num_simulations,
            num_steps=self.num_steps,
            seed=self.seed,
            regression_type=self.regression_type,
        )

    def greeks(self, contract: Contract) -> Dict[str, float]:
        metrics = finite_difference_greeks(pricer=self.price, contract=contract, bumps=self.bumps)
        metrics.pop("price", None)
        return metrics

    def analyze(self, contract: Contract) -> Dict[str, float]:
        return finite_difference_greeks(pricer=self.price, contract=contract, bumps=self.bumps)
