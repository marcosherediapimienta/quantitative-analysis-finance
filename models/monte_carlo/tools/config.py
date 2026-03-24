from .types import BumpConfig, ExerciseStyle, RegressionType

DEFAULT_NUM_SIMULATIONS: int = 10000
DEFAULT_NUM_STEPS: int = 50
DEFAULT_REGRESSION_TYPE: RegressionType = "quadratic"
DEFAULT_REGRESSION_DEGREE: int = 2
DEFAULT_EXERCISE_STYLE: ExerciseStyle = "european"
MIN_POSITIVE_VALUE: float = 1e-8

DEFAULT_DIFF_BUMPS: BumpConfig = {
    "spot_bump_rel": 0.01,
    "spot_bump_min": 0.01,
    "vol_bump_abs": 0.01,
    "time_bump_abs": 0.01,
    "rate_bump_abs": 0.01,
}
