from typing import Dict, Optional
from ..tools.config import DEFAULT_NUM_SIMULATIONS
from ..tools.european import price_european_monte_carlo
from ..tools.types import Contract

class EuropeanMCAnalyzer:
    def __init__(self, num_simulations: int = DEFAULT_NUM_SIMULATIONS, seed: Optional[int] = None) -> None:
        self.num_simulations = num_simulations
        self.seed = seed

    def price(self, contract: Contract) -> float:
        return price_european_monte_carlo(
            contract=contract,
            num_simulations=self.num_simulations,
            seed=self.seed,
        )

    def analyze(self, contract: Contract) -> Dict[str, float]:
        return {"price": self.price(contract)}
