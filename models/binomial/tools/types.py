from typing import Literal, TypedDict

OptionType = Literal["call", "put"]
FactorScheme = Literal["crr"]
ExerciseStyle = Literal["european", "american"]


class Contract(TypedDict):
    spot: float
    strike: float
    time_to_maturity: float
    risk_free_rate: float
    volatility: float
    steps: int
    option_type: OptionType


class BumpConfig(TypedDict, total=False):
    spot_bump_rel: float
    spot_bump_min: float
    vol_bump_abs: float
    time_bump_abs: float
    rate_bump_abs: float
