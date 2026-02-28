from dataclasses import dataclass

from tools.config import DEFAULT_DIFF_BUMPS, DEFAULT_IV_CALIBRATION

@dataclass(frozen=True)
class FiniteDifferenceConfig:
    spot_bump_rel: float = DEFAULT_DIFF_BUMPS["spot_bump_rel"]
    spot_bump_min: float = DEFAULT_DIFF_BUMPS["spot_bump_min"]
    vol_bump_abs: float = DEFAULT_DIFF_BUMPS["vol_bump_abs"]
    time_bump_abs: float = DEFAULT_DIFF_BUMPS["time_bump_abs"]
    rate_bump_abs: float = DEFAULT_DIFF_BUMPS["rate_bump_abs"]


@dataclass(frozen=True)
class ImpliedVolatilityConfig:
    initial_sigma: float = DEFAULT_IV_CALIBRATION["initial_sigma"]
    sigma_min: float = DEFAULT_IV_CALIBRATION["sigma_min"]
    sigma_max: float = DEFAULT_IV_CALIBRATION["sigma_max"]
    min_abs_vega: float = DEFAULT_IV_CALIBRATION["min_abs_vega"]
    tolerance: float = DEFAULT_IV_CALIBRATION["tolerance"]
    max_iterations: int = DEFAULT_IV_CALIBRATION["max_iterations"]
