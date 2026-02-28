from typing import Callable, Dict

import numpy as np

from tools.analytics.options import OptionContract, OptionType

class CoxRossRubinsteinAmericanModel:
    _payoff: Dict[OptionType, Callable[[np.ndarray, float], np.ndarray]] = {
        OptionType.CALL: lambda st, k: np.maximum(st - k, 0.0),
        OptionType.PUT: lambda st, k: np.maximum(k - st, 0.0),
    }

    @staticmethod
    def tree_factors(contract: OptionContract) -> Dict[str, float]:
        dt = contract.time_to_maturity / contract.steps
        u = np.exp(contract.volatility * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp(contract.risk_free_rate * dt) - d) / (u - d)
        discount_factor = np.exp(-contract.risk_free_rate * dt)
        return {"dt": dt, "u": u, "d": d, "p": p, "discount_factor": discount_factor}

    def price(self, contract: OptionContract) -> float:
        factors = self.tree_factors(contract)
        payoff = self._payoff[contract.option_type]

        st_terminal = contract.spot * (factors["u"] ** np.arange(contract.steps + 1)) * (
            factors["d"] ** (contract.steps - np.arange(contract.steps + 1))
        )
        option_value = payoff(st_terminal, contract.strike)

        for i in range(contract.steps - 1, -1, -1):
            st = contract.spot * (factors["u"] ** np.arange(i + 1)) * (factors["d"] ** (i - np.arange(i + 1)))
            continuation = factors["discount_factor"] * (
                factors["p"] * option_value[1 : i + 2] + (1.0 - factors["p"]) * option_value[0 : i + 1]
            )
            exercise_value = payoff(st, contract.strike)
            option_value = np.maximum(continuation, exercise_value)
        return float(option_value[0])
