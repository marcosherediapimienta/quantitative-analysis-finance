from typing import Dict, Mapping, Optional
from ..tools.config import DEFAULT_NUM_SIMULATIONS
from ..tools.european import price_european_monte_carlo
from ..tools.greeks import finite_difference_greeks
from ..tools.types import Contract

class EuropeanMCAnalyzer:
    def __init__(
        self,
        num_simulations: int = DEFAULT_NUM_SIMULATIONS,
        seed: Optional[int] = None,
        bumps: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.num_simulations = num_simulations
        self.seed = seed
        self.bumps = dict(bumps or {})

    def price(self, contract: Contract) -> float:
        return price_european_monte_carlo(
            contract=contract,
            num_simulations=self.num_simulations,
            seed=self.seed,
        )

    def greeks(self, contract: Contract) -> Dict[str, float]:
        metrics = finite_difference_greeks(pricer=self.price, contract=contract, bumps=self.bumps)
        metrics.pop("price", None)
        return metrics

    def analyze(self, contract: Contract) -> Dict[str, float]:
        return finite_difference_greeks(pricer=self.price, contract=contract, bumps=self.bumps)
