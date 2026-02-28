import numpy as np
from typing import Tuple

def value_at_risk_expected_shortfall(pnl: np.ndarray, alpha: float = 0.01) -> Tuple[float, float]:
    pnl_sorted = np.sort(pnl)
    var = -np.percentile(pnl_sorted, alpha * 100)
    es = -pnl_sorted[pnl_sorted <= -var].mean()
    return float(var), float(es)
