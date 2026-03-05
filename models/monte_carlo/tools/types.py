from typing import Literal, TypedDict

OptionType = Literal["call", "put"]
RegressionType = Literal["linear", "quadratic", "cubic"]

class Contract(TypedDict):
    spot: float
    strike: float
    time_to_maturity: float
    risk_free_rate: float
    volatility: float
    option_type: OptionType
