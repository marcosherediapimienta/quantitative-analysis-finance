from math import erf, exp, log, pi, sqrt
from typing import Dict

from ...tools.config import MIN_POSITIVE_VALUE
from ...tools.types import Contract


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def get_d_values(contract: Contract) -> Dict[str, float]:
    spot = max(contract["spot"], MIN_POSITIVE_VALUE)
    strike = max(contract["strike"], MIN_POSITIVE_VALUE)
    time_to_maturity = max(contract["time_to_maturity"], MIN_POSITIVE_VALUE)
    volatility = max(contract["volatility"], MIN_POSITIVE_VALUE)
    rate = contract["risk_free_rate"]

    d1 = (log(spot / strike) + (rate + 0.5 * volatility * volatility) * time_to_maturity) / (
        volatility * sqrt(time_to_maturity)
    )
    d2 = d1 - volatility * sqrt(time_to_maturity)
    return {"d1": d1, "d2": d2}
