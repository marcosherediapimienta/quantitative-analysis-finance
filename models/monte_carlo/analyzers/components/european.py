import numpy as np
from typing import Optional

from ...tools.config import DEFAULT_NUM_SIMULATIONS
from ...tools.types import Contract
from .helper import simulate_terminal_prices
from .payoff import get_payoff


def price_european_monte_carlo(
    contract: Contract,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    seed: Optional[int] = None,
) -> float:
    terminal_prices = simulate_terminal_prices(contract=contract, num_simulations=num_simulations, seed=seed)
    payoff = get_payoff(contract["option_type"])(terminal_prices, contract["strike"])
    discount = np.exp(-contract["risk_free_rate"] * contract["time_to_maturity"])
    return float(discount * np.mean(payoff))
