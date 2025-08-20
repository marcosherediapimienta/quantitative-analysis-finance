from typing import Literal

class Option:
    def __init__(self, type: Literal['call', 'put'], style: Literal['european', 'american'], spot: float, strike: float, maturity: float, volatility: float, rate: float):
        self.type = type
        self.style = style
        self.spot = spot
        self.strike = strike
        self.maturity = maturity
        self.volatility = volatility
        self.rate = rate
        self._validate()

    def _validate(self):
        if self.type not in ['call', 'put']:
            raise ValueError('type must be "call" or "put"')
        if self.style not in ['european', 'american']:
            raise ValueError('style must be "european" or "american"')
        if self.spot <= 0 or self.strike <= 0 or self.maturity <= 0 or self.volatility < 0 or self.rate < 0:
            raise ValueError('Numeric parameters must be positive')

    def __repr__(self):
        return (f"Option(type={self.type}, style={self.style}, spot={self.spot}, strike={self.strike}, maturity={self.maturity}, volatility={self.volatility}, rate={self.rate})")

    def payoff(self, S: float) -> float:
        if self.type == 'call':
            return max(S - self.strike, 0)
        else:
            return max(self.strike - S, 0)
