from typing import Dict, Mapping, Optional
from ..tools.american import price_american_binomial
from ..tools.config import DEFAULT_SCHEME
from ..tools.greeks import finite_difference_greeks
from ..tools.types import Contract

class AmericanCRRAnalyzer:
    def __init__(
        self,
        scheme: str = DEFAULT_SCHEME,
        bumps: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.scheme = scheme
        self.bumps = dict(bumps or {})

    def price(self, contract: Contract) -> float:
        return price_american_binomial(contract=contract, scheme=self.scheme)

    def greeks(self, contract: Contract) -> Dict[str, float]:
        metrics = finite_difference_greeks(pricer=self.price, contract=contract, bumps=self.bumps)
        metrics.pop("price", None)
        return metrics

    def analyze(self, contract: Contract) -> Dict[str, float]:
        return finite_difference_greeks(pricer=self.price, contract=contract, bumps=self.bumps)
