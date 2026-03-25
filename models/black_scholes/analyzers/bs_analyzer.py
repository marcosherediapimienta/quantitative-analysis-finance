from typing import Dict

from ..tools.types import Contract
from .components.european import price_european_black_scholes
from .components.greeks import analytical_greeks


class BSAnalyzer:
    def price(self, contract: Contract) -> float:
        return price_european_black_scholes(contract=contract)

    def greeks(self, contract: Contract) -> Dict[str, float]:
        return analytical_greeks(contract)

    def analyze(self, contract: Contract) -> Dict[str, float]:
        return {"price": self.price(contract), **self.greeks(contract)}
