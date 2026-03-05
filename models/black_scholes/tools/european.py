from math import exp
from typing import Callable, Dict
from .helper import get_d_values, normal_cdf
from .types import Contract, OptionType

PriceFn = Callable[[Contract], float]

def _call_price(contract: Contract) -> float:
    d_values = get_d_values(contract)
    spot = contract["spot"]
    strike = contract["strike"]
    rate = contract["risk_free_rate"]
    maturity = contract["time_to_maturity"]
    return float(spot * normal_cdf(d_values["d1"]) - strike * exp(-rate * maturity) * normal_cdf(d_values["d2"]))


def _put_price(contract: Contract) -> float:
    d_values = get_d_values(contract)
    spot = contract["spot"]
    strike = contract["strike"]
    rate = contract["risk_free_rate"]
    maturity = contract["time_to_maturity"]
    return float(strike * exp(-rate * maturity) * normal_cdf(-d_values["d2"]) - spot * normal_cdf(-d_values["d1"]))


PRICE_BY_OPTION_TYPE: Dict[OptionType, PriceFn] = {
    "call": _call_price,
    "put": _put_price,
}


def price_european_black_scholes(contract: Contract) -> float:
    return PRICE_BY_OPTION_TYPE[contract["option_type"]](contract)
