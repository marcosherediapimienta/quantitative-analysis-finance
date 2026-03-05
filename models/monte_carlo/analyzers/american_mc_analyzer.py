from typing import Dict, Mapping, Optional
from ..tools.american import price_american_monte_carlo
from ..tools.config import DEFAULT_NUM_SIMULATIONS, DEFAULT_NUM_STEPS, DEFAULT_REGRESSION_TYPE
from ..tools.greeks import finite_difference_greeks
from ..tools.types import Contract

class AmericanMCAnalyzer:
    def __init__(
        self,
        num_simulations: int = DEFAULT_NUM_SIMULATIONS,
        num_steps: int = DEFAULT_NUM_STEPS,
        seed: Optional[int] = None,
        regression_type: str = DEFAULT_REGRESSION_TYPE,
        bumps: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.seed = seed
        self.regression_type = regression_type
        self.bumps = dict(bumps or {})

    def price(self, contract: Contract) -> float:
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
