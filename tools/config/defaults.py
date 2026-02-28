TRADING_DAYS_PER_YEAR = 252
DEFAULT_FALLBACK_VOLATILITY = 0.2

DEFAULT_MC_SEED = 42

DEFAULT_DIFF_BUMPS = {
    "spot_bump_rel": 0.01,
    "spot_bump_min": 0.01,
    "vol_bump_abs": 0.01,
    "time_bump_abs": 0.01,
    "rate_bump_abs": 0.01,
}

DEFAULT_IV_CALIBRATION = {
    "initial_sigma": 0.2,
    "sigma_min": 1e-6,
    "sigma_max": 5.0,
    "min_abs_vega": 1e-8,
    "tolerance": 1e-6,
    "max_iterations": 100,
}
