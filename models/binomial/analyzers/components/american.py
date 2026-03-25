import numpy as np

from ...tools.config import DEFAULT_SCHEME
from ...tools.types import Contract, FactorScheme
from .helper import (
    early_exercise_spot_prices,
    get_tree_factors,
    step_back,
    terminal_spot_prices,
)
from .payoff import get_payoff


def price_american_binomial(contract: Contract, scheme: FactorScheme = DEFAULT_SCHEME) -> float:
    up, down, p_up, discount = get_tree_factors(contract, scheme=scheme)
    payoff = get_payoff(contract["option_type"])
    values = payoff(terminal_spot_prices(contract, up=up, down=down), contract["strike"])

    for step in range(contract["steps"] - 1, -1, -1):
        continuation = step_back(values=values, p_up=p_up, discount=discount)
        exercise_spot = early_exercise_spot_prices(contract, up=up, down=down, step=step)
        exercise_value = payoff(exercise_spot, contract["strike"])
        values = np.maximum(continuation, exercise_value)

    return float(values[0])
