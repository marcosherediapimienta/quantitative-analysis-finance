from typing import Optional

import numpy as np

from tools.analytics.options import OptionContract, OptionType

class AmericanLongstaffSchwartzPricer:
    _regression_degree = {
        "linear": 1,
        "quadratic": 2,
        "cubic": 3,
    }

    def price(
        self,
        contract: OptionContract,
        num_simulations: int = 10000,
        num_steps: int = 50,
        seed: Optional[int] = None,
        regression_type: str = "quadratic",
    ) -> float:
        dt = contract.time_to_maturity / num_steps
        discount_factor = np.exp(-contract.risk_free_rate * dt)

        s_paths = np.zeros((num_simulations, num_steps + 1))
        s_paths[:, 0] = contract.spot
        rng = np.random.default_rng(seed)
        for t in range(1, num_steps + 1):
            z = rng.standard_normal(num_simulations)
            s_paths[:, t] = s_paths[:, t - 1] * np.exp(
                (contract.risk_free_rate - 0.5 * contract.volatility**2) * dt
                + contract.volatility * np.sqrt(dt) * z
            )

        payoff = {
            OptionType.CALL: np.maximum(s_paths - contract.strike, 0.0),
            OptionType.PUT: np.maximum(contract.strike - s_paths, 0.0),
        }[contract.option_type]

        option_value = payoff[:, -1]
        degree = self._regression_degree.get(regression_type, 2)

        for t in range(num_steps - 1, 0, -1):
            itm = payoff[:, t] > 0
            if np.any(itm):
                x = s_paths[itm, t]
                y = option_value[itm] * discount_factor
                coeffs = np.polyfit(x, y, deg=degree)
                continuation = np.polyval(coeffs, x)
                exercise = payoff[itm, t]
                exercise_now = exercise > continuation
                idx = np.where(itm)[0][exercise_now]
                option_value[idx] = exercise[exercise_now]
            option_value = option_value * discount_factor
        return float(np.mean(option_value))
