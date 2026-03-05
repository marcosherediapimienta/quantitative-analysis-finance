from typing import Dict, Optional
from ..tools.american import price_american_monte_carlo
from ..tools.config import DEFAULT_NUM_SIMULATIONS, DEFAULT_NUM_STEPS, DEFAULT_REGRESSION_TYPE
from ..tools.types import Contract

class AmericanMCAnalyzer:
    def __init__(
        self,
        num_simulations: int = DEFAULT_NUM_SIMULATIONS,
        num_steps: int = DEFAULT_NUM_STEPS,
        seed: Optional[int] = None,
        regression_type: str = DEFAULT_REGRESSION_TYPE,
    ) -> None:
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.seed = seed
        self.regression_type = regression_type

    def price(self, contract: Contract) -> float:
        return price_american_monte_carlo(
            contract=contract,
            num_simulations=self.num_simulations,
            num_steps=self.num_steps,
            seed=self.seed,
            regression_type=self.regression_type,
        )

    def analyze(self, contract: Contract) -> Dict[str, float]:
        return {"price": self.price(contract)}
