from .types import BumpConfig, ExerciseStyle, FactorScheme

DEFAULT_STEPS: int = 500
DEFAULT_SCHEME: FactorScheme = "crr"
DEFAULT_EXERCISE_STYLE: ExerciseStyle = "european"
MIN_POSITIVE_VALUE: float = 1e-8

DEFAULT_DIFF_BUMPS: BumpConfig = {
    "spot_bump_rel": 0.01,
    "spot_bump_min": 0.01,
    "vol_bump_abs": 0.01,
    "time_bump_abs": 0.01,
    "rate_bump_abs": 0.01,
}
