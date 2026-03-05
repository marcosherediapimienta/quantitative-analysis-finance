from typing import Literal, TypedDict

OptionType = Literal["call", "put"]
FactorScheme = Literal["crr"]

class Contract(TypedDict):
    spot: float
    strike: float
    time_to_maturity: float
    risk_free_rate: float
    volatility: float
    steps: int
    option_type: OptionType
