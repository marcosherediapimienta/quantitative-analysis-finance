from typing import Callable, Dict

import numpy as np

class MonteCarloGreeksCalculator:
    def __init__(
        self,
        pricing_fn: Callable[..., float],
        spot_bump_rel: float = 0.01,
        vol_bump_abs: float = 0.01,
        rate_bump_abs: float = 0.01,
        time_bump_abs: float = 1 / 365,
    ) -> None:
        self.pricing_fn = pricing_fn
        self.spot_bump_rel = spot_bump_rel
        self.vol_bump_abs = vol_bump_abs
        self.rate_bump_abs = rate_bump_abs
        self.time_bump_abs = time_bump_abs

    def calculate(self, S: float, K: float, T: float, r: float, sigma: float, **kwargs) -> Dict[str, float]:
        base_rng = np.random.default_rng(kwargs.get("seed", None))
        seeds = base_rng.integers(0, 1e9, size=7)

        eps_s = max(self.spot_bump_rel * S, 1e-8)
        price_up = self.pricing_fn(S=S + eps_s, K=K, T=T, r=r, sigma=sigma, seed=int(seeds[0]), **kwargs)
        price_down = self.pricing_fn(S=S - eps_s, K=K, T=T, r=r, sigma=sigma, seed=int(seeds[1]), **kwargs)
        price = self.pricing_fn(S=S, K=K, T=T, r=r, sigma=sigma, seed=int(seeds[2]), **kwargs)
        delta = (price_up - price_down) / (2 * eps_s)
        gamma = (price_up - 2 * price + price_down) / (eps_s**2)

        eps_sigma = self.vol_bump_abs
        price_vega_up = self.pricing_fn(S=S, K=K, T=T, r=r, sigma=sigma + eps_sigma, seed=int(seeds[3]), **kwargs)
        price_vega_down = self.pricing_fn(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=max(1e-8, sigma - eps_sigma),
            seed=int(seeds[4]),
            **kwargs,
        )
        vega = (price_vega_up - price_vega_down) / (2 * eps_sigma)

        dt = min(self.time_bump_abs, max(1e-8, T - 1e-8))
        if T - dt > 0:
            price_t = self.pricing_fn(S=S, K=K, T=T - dt, r=r, sigma=sigma, seed=int(seeds[5]), **kwargs)
            theta = (price_t - price) / dt
        else:
            theta = float("nan")

        dr = self.rate_bump_abs
        price_rho_up = self.pricing_fn(S=S, K=K, T=T, r=r + dr, sigma=sigma, seed=int(seeds[6]), **kwargs)
        price_rho_down = self.pricing_fn(S=S, K=K, T=T, r=r - dr, sigma=sigma, seed=int(seeds[2]), **kwargs)
        rho = (price_rho_up - price_rho_down) / (2 * dr)
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
