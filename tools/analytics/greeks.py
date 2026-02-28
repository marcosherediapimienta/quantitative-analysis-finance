from typing import Dict, Optional

from models import CoxRossRubinsteinBinomialModel, FiniteDifferenceConfig, OptionContract

class BinomialGreeksCalculator:
    def __init__(
        self,
        pricer: Optional[CoxRossRubinsteinBinomialModel] = None,
        config: Optional[FiniteDifferenceConfig] = None,
    ) -> None:
        self.pricer = pricer or CoxRossRubinsteinBinomialModel()
        self.config = config or FiniteDifferenceConfig()

    def calculate(self, contract: OptionContract) -> Dict[str, float]:
        h = max(self.config.spot_bump_rel * contract.spot, self.config.spot_bump_min)
        base = self.pricer.price(contract)

        price_up = self.pricer.price(
            OptionContract(
                spot=contract.spot + h,
                strike=contract.strike,
                time_to_maturity=contract.time_to_maturity,
                risk_free_rate=contract.risk_free_rate,
                volatility=contract.volatility,
                steps=contract.steps,
                option_type=contract.option_type,
            )
        )
        price_down = self.pricer.price(
            OptionContract(
                spot=contract.spot - h,
                strike=contract.strike,
                time_to_maturity=contract.time_to_maturity,
                risk_free_rate=contract.risk_free_rate,
                volatility=contract.volatility,
                steps=contract.steps,
                option_type=contract.option_type,
            )
        )
        delta = (price_up - price_down) / (2.0 * h)
        gamma = (price_up - 2.0 * base + price_down) / (h**2)

        vol_bump = self.config.vol_bump_abs
        price_vega_up = self.pricer.price(
            OptionContract(
                spot=contract.spot,
                strike=contract.strike,
                time_to_maturity=contract.time_to_maturity,
                risk_free_rate=contract.risk_free_rate,
                volatility=contract.volatility + vol_bump,
                steps=contract.steps,
                option_type=contract.option_type,
            )
        )
        price_vega_down = self.pricer.price(
            OptionContract(
                spot=contract.spot,
                strike=contract.strike,
                time_to_maturity=contract.time_to_maturity,
                risk_free_rate=contract.risk_free_rate,
                volatility=max(1e-8, contract.volatility - vol_bump),
                steps=contract.steps,
                option_type=contract.option_type,
            )
        )
        vega = (price_vega_up - price_vega_down) / (2.0 * vol_bump)

        time_bump = min(self.config.time_bump_abs, max(1e-8, contract.time_to_maturity - 1e-8))
        price_theta = self.pricer.price(
            OptionContract(
                spot=contract.spot,
                strike=contract.strike,
                time_to_maturity=max(1e-8, contract.time_to_maturity - time_bump),
                risk_free_rate=contract.risk_free_rate,
                volatility=contract.volatility,
                steps=contract.steps,
                option_type=contract.option_type,
            )
        )
        theta = (price_theta - base) / time_bump

        rate_bump = self.config.rate_bump_abs
        price_rho_up = self.pricer.price(
            OptionContract(
                spot=contract.spot,
                strike=contract.strike,
                time_to_maturity=contract.time_to_maturity,
                risk_free_rate=contract.risk_free_rate + rate_bump,
                volatility=contract.volatility,
                steps=contract.steps,
                option_type=contract.option_type,
            )
        )
        price_rho_down = self.pricer.price(
            OptionContract(
                spot=contract.spot,
                strike=contract.strike,
                time_to_maturity=contract.time_to_maturity,
                risk_free_rate=contract.risk_free_rate - rate_bump,
                volatility=contract.volatility,
                steps=contract.steps,
                option_type=contract.option_type,
            )
        )
        rho = (price_rho_up - price_rho_down) / (2.0 * rate_bump)
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
