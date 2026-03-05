import numpy as np
from typing import Optional
from .config import DEFAULT_NUM_SIMULATIONS, DEFAULT_NUM_STEPS, DEFAULT_REGRESSION_TYPE
from .helper import get_regression_degree, simulate_price_paths
from .payoff import get_payoff
from .types import Contract

def price_american_monte_carlo(
    contract: Contract,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    num_steps: int = DEFAULT_NUM_STEPS,
    seed: Optional[int] = None,
    regression_type: str = DEFAULT_REGRESSION_TYPE,
) -> float:
    paths = simulate_price_paths(
        contract=contract,
        num_simulations=num_simulations,
        num_steps=num_steps,
        seed=seed,
    )
    payoff_fn = get_payoff(contract["option_type"])
    payoff = payoff_fn(paths, contract["strike"])

    dt = contract["time_to_maturity"] / num_steps
    discount = np.exp(-contract["risk_free_rate"] * dt)
    option_value = payoff[:, -1]
    degree = get_regression_degree(regression_type)

    for step in range(num_steps - 1, 0, -1):
        continuation_all = option_value * discount
        option_value = continuation_all.copy()

        in_the_money = payoff[:, step] > 0.0
        if np.any(in_the_money):
            x = paths[in_the_money, step]
            y = continuation_all[in_the_money]
            coeffs = np.polyfit(x, y, deg=degree)
            continuation = np.polyval(coeffs, x)
            exercise = payoff[in_the_money, step]
            exercise_now = exercise > continuation
            exercise_idx = np.where(in_the_money)[0][exercise_now]
            option_value[exercise_idx] = exercise[exercise_now]

    continuation_at_t0 = float(np.mean(option_value) * discount)
    immediate_exercise_t0 = float(payoff_fn(np.array([contract["spot"]]), contract["strike"])[0])
    european_lower_bound = float(
        np.exp(-contract["risk_free_rate"] * contract["time_to_maturity"]) * np.mean(payoff[:, -1])
    )
    return max(immediate_exercise_t0, continuation_at_t0, european_lower_bound)
