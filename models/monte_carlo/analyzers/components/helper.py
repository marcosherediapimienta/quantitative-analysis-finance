import numpy as np
from typing import Dict, Optional

from ...tools.config import DEFAULT_REGRESSION_DEGREE
from ...tools.types import Contract, RegressionType

REGRESSION_DEGREE_BY_TYPE: Dict[RegressionType, int] = {
    "linear": 1,
    "quadratic": 2,
    "cubic": 3,
}


def get_regression_degree(regression_type: RegressionType) -> int:
    return REGRESSION_DEGREE_BY_TYPE.get(regression_type, DEFAULT_REGRESSION_DEGREE)


def simulate_terminal_prices(contract: Contract, num_simulations: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(num_simulations)
    drift = (contract["risk_free_rate"] - 0.5 * contract["volatility"] ** 2) * contract["time_to_maturity"]
    diffusion = contract["volatility"] * np.sqrt(contract["time_to_maturity"]) * shocks
    return contract["spot"] * np.exp(drift + diffusion)


def simulate_price_paths(
    contract: Contract,
    num_simulations: int,
    num_steps: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    dt = contract["time_to_maturity"] / num_steps
    drift = (contract["risk_free_rate"] - 0.5 * contract["volatility"] ** 2) * dt
    diffusion_scale = contract["volatility"] * np.sqrt(dt)

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((num_simulations, num_steps))

    paths = np.zeros((num_simulations, num_steps + 1))
    paths[:, 0] = contract["spot"]

    for step in range(1, num_steps + 1):
        paths[:, step] = paths[:, step - 1] * np.exp(drift + diffusion_scale * shocks[:, step - 1])

    return paths
