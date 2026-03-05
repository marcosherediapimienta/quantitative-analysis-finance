from typing import Callable, Dict, Mapping, Optional
from .config import DEFAULT_DIFF_BUMPS, MIN_POSITIVE_VALUE
from .types import Contract

PricerFn = Callable[[Contract], float]

def _with_updates(contract: Contract, **updates: float) -> Contract:
    updated = dict(contract)
    updated.update(updates)
    return updated 

def finite_difference_greeks(
    pricer: PricerFn,
    contract: Contract,
    bumps: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    bump_values = {**DEFAULT_DIFF_BUMPS, **(bumps or {})}
    base_price = pricer(contract)

    spot_bump = max(bump_values["spot_bump_rel"] * contract["spot"], bump_values["spot_bump_min"])
    price_spot_up = pricer(_with_updates(contract, spot=contract["spot"] + spot_bump))
    price_spot_down = pricer(_with_updates(contract, spot=max(MIN_POSITIVE_VALUE, contract["spot"] - spot_bump)))
    delta = (price_spot_up - price_spot_down) / (2.0 * spot_bump)
    gamma = (price_spot_up - 2.0 * base_price + price_spot_down) / (spot_bump**2)

    vol_bump = bump_values["vol_bump_abs"]
    price_vol_up = pricer(_with_updates(contract, volatility=contract["volatility"] + vol_bump))
    price_vol_down = pricer(
        _with_updates(contract, volatility=max(MIN_POSITIVE_VALUE, contract["volatility"] - vol_bump))
    )
    vega = (price_vol_up - price_vol_down) / (2.0 * vol_bump)

    time_bump = min(
        bump_values["time_bump_abs"],
        max(MIN_POSITIVE_VALUE, contract["time_to_maturity"] - MIN_POSITIVE_VALUE),
    )
    price_time_down = pricer(
        _with_updates(contract, time_to_maturity=max(MIN_POSITIVE_VALUE, contract["time_to_maturity"] - time_bump))
    )
    theta = (price_time_down - base_price) / time_bump

    rate_bump = bump_values["rate_bump_abs"]
    price_rate_up = pricer(_with_updates(contract, risk_free_rate=contract["risk_free_rate"] + rate_bump))
    price_rate_down = pricer(_with_updates(contract, risk_free_rate=contract["risk_free_rate"] - rate_bump))
    rho = (price_rate_up - price_rate_down) / (2.0 * rate_bump)

    return {
        "price": base_price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
