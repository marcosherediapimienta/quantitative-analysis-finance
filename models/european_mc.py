from typing import Optional

import numpy as np

from tools.analytics.options import OptionContract, OptionType


class EuropeanMonteCarloPricer:
    def price(
        self,
        contract: OptionContract,
        num_simulations: int = 10000,
        seed: Optional[int] = None,
    ) -> float:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(num_simulations)
        return self.price_with_shocks(contract=contract, z=z)

    @staticmethod
    def price_with_shocks(contract: OptionContract, z: np.ndarray) -> float:
        st = contract.spot * np.exp(
            (contract.risk_free_rate - 0.5 * contract.volatility**2) * contract.time_to_maturity
            + contract.volatility * np.sqrt(contract.time_to_maturity) * z
        )
        payoff = {
            OptionType.CALL: np.maximum(st - contract.strike, 0.0),
            OptionType.PUT: np.maximum(contract.strike - st, 0.0),
        }[contract.option_type]
        return float(np.exp(-contract.risk_free_rate * contract.time_to_maturity) * np.mean(payoff))
