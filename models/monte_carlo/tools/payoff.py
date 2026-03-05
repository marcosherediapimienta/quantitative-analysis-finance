import numpy as np
from typing import Callable, Dict
from .types import OptionType

PayoffFn = Callable[[np.ndarray, float], np.ndarray]

PAYOFF_BY_TYPE: Dict[OptionType, PayoffFn] = {
    "call": lambda st, strike: np.maximum(st - strike, 0.0),
    "put": lambda st, strike: np.maximum(strike - st, 0.0),
}

def get_payoff(option_type: OptionType) -> PayoffFn:
    return PAYOFF_BY_TYPE[option_type]
