from ...tools.config import DEFAULT_SCHEME
from ...tools.types import Contract, FactorScheme
from .helper import get_tree_factors, step_back, terminal_spot_prices
from .payoff import get_payoff


def price_european_binomial(contract: Contract, scheme: FactorScheme = DEFAULT_SCHEME) -> float:
    up, down, p_up, discount = get_tree_factors(contract, scheme=scheme)
    payoff = get_payoff(contract["option_type"])
    values = payoff(terminal_spot_prices(contract, up=up, down=down), contract["strike"])

    for _ in range(contract["steps"]):
        values = step_back(values=values, p_up=p_up, discount=discount)

    return float(values[0])
