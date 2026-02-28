from typing import Callable, Dict

import numpy as np

from tools.analytics.options import OptionContract, OptionType


class CoxRossRubinsteinBinomialModel:
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
        j = np.arange(contract.steps + 1)
        st_terminal = contract.spot * (factors["u"] ** j) * (factors["d"] ** (contract.steps - j))
        payoff = self._payoff[contract.option_type](st_terminal, contract.strike)

        for _ in range(contract.steps):
            payoff = factors["discount_factor"] * (
                factors["p"] * payoff[1:] + (1.0 - factors["p"]) * payoff[:-1]
            )
        return float(payoff[0])
