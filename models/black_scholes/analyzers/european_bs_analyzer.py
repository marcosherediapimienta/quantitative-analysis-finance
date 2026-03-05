from typing import Dict
from ..tools.european import price_european_black_scholes
from ..tools.types import Contract

class EuropeanBSAnalyzer:
    def price(self, contract: Contract) -> float:
        return price_european_black_scholes(contract=contract)

    def analyze(self, contract: Contract) -> Dict[str, float]:
        return {"price": self.price(contract)}
