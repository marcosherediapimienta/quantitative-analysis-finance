import numpy as np
from typing import Callable, Dict, Tuple

from ...tools.config import DEFAULT_SCHEME, DEFAULT_STEPS
from ...tools.types import Contract, FactorScheme

TreeFactorFn = Callable[[float, float, float, int], Tuple[float, float, float, float]]


def _crr_factors(
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    steps: int,
) -> Tuple[float, float, float, float]:
    dt = time_to_maturity / steps
    up = float(np.exp(volatility * np.sqrt(dt)))
    down = 1.0 / up
    growth = float(np.exp(risk_free_rate * dt))
    p_up = (growth - down) / (up - down)
    discount = float(np.exp(-risk_free_rate * dt))
    return up, down, p_up, discount


FACTOR_SCHEMES: Dict[FactorScheme, TreeFactorFn] = {
    "crr": _crr_factors,
}


def normalize_contract(contract: Contract) -> Contract:
    normalized = dict(contract)
    normalized["steps"] = int(normalized.get("steps", DEFAULT_STEPS))
    return normalized  # type: ignore[return-value]


def get_tree_factors(contract: Contract, scheme: FactorScheme = DEFAULT_SCHEME) -> Tuple[float, float, float, float]:
    normalized = normalize_contract(contract)
    factor_fn = FACTOR_SCHEMES[scheme]
    return factor_fn(
        normalized["time_to_maturity"],
        normalized["risk_free_rate"],
        normalized["volatility"],
        normalized["steps"],
    )


def terminal_spot_prices(contract: Contract, up: float, down: float) -> np.ndarray:
    steps = contract["steps"]
    j = np.arange(steps + 1)
    return contract["spot"] * (up**j) * (down ** (steps - j))


def step_back(values: np.ndarray, p_up: float, discount: float) -> np.ndarray:
    return discount * (p_up * values[1:] + (1.0 - p_up) * values[:-1])


def early_exercise_spot_prices(contract: Contract, up: float, down: float, step: int) -> np.ndarray:
    j = np.arange(step + 1)
    return contract["spot"] * (up**j) * (down ** (step - j))
