from typing import Literal, TypedDict

OptionType = Literal["call", "put"]
SolverMethod = Literal["newton", "bisection"]


class IVContract(TypedDict):
    spot: float
    strike: float
    time_to_maturity: float
    risk_free_rate: float
    market_price: float
    option_type: OptionType
