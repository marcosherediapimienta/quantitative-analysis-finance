from math import exp, sqrt
from typing import Dict
from .helper import get_d_values, normal_cdf, normal_pdf
from .types import Contract

def analytical_greeks(contract: Contract) -> Dict[str, float]:
    d_values = get_d_values(contract)
    d1 = d_values["d1"]
    d2 = d_values["d2"]

    spot = contract["spot"]
    strike = contract["strike"]
    time_to_maturity = contract["time_to_maturity"]
    rate = contract["risk_free_rate"]
    volatility = contract["volatility"]
    option_type = contract["option_type"]

    discount = exp(-rate * time_to_maturity)
    pdf_d1 = normal_pdf(d1)
    sqrt_t = sqrt(time_to_maturity)

    delta_by_type = {
        "call": normal_cdf(d1),
        "put": normal_cdf(d1) - 1.0,
    }
    theta_by_type = {
        "call": -(spot * pdf_d1 * volatility) / (2.0 * sqrt_t) - rate * strike * discount * normal_cdf(d2),
        "put": -(spot * pdf_d1 * volatility) / (2.0 * sqrt_t) + rate * strike * discount * normal_cdf(-d2),
    }
    rho_by_type = {
        "call": strike * time_to_maturity * discount * normal_cdf(d2),
        "put": -strike * time_to_maturity * discount * normal_cdf(-d2),
    }

    return {
        "delta": float(delta_by_type[option_type]),
        "gamma": float(pdf_d1 / (spot * volatility * sqrt_t)),
        "vega": float(spot * pdf_d1 * sqrt_t),
        "theta": float(theta_by_type[option_type]),
        "rho": float(rho_by_type[option_type]),
    }
