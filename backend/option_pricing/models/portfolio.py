from typing import List
from backend.option_pricing.models.option import Option

class Portfolio:
    def __init__(self, options: List[Option]):
        self.options = options

    def value(self, spot_prices: List[float]) -> float:
        # Calcula el valor total del portfolio dado una lista de precios spot
        return sum(opt.payoff(S) for opt, S in zip(self.options, spot_prices))

    def __repr__(self):
        return f"Portfolio(options={self.options})"
