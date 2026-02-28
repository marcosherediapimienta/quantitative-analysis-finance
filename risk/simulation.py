import numpy as np
from typing import Dict, List, Optional

from tools.config import DEFAULT_FALLBACK_VOLATILITY
from portfolio import price_option_monte_carlo, solve_option_implied_volatility

def simulate_monte_carlo_portfolio_pricing(
    portfolio: List[Dict],
    n_sims: int = 1000,
    n_steps: int = 50,
    horizon: Optional[float] = None,
    vol_shock_sigma: float = 0.1,
    rho: float = -0.5,
) -> Dict[str, np.ndarray]:
    base_val = sum(price_option_monte_carlo(opt, n_sim=1000, n_steps=n_steps) * opt["qty"] for opt in portfolio)
    pnl = []
    shocks_dict: Dict[str, List[float]] = {}
    for opt in portfolio:
        key = opt.get("ticker", opt["S"])
        shocks_dict.setdefault(key, [])

    for _ in range(n_sims):
        shocked_portfolio = []
        sigma_shocks = []
        for opt in portfolio:
            T_sim = horizon if horizon is not None else opt["T"]

            z1, z2 = np.random.normal(size=2)
            z_spot = z1
            z_vol = rho * z1 + np.sqrt(1 - rho**2) * z2
            key = opt.get("ticker", opt["S"])
            shocks_dict[key].append(float(z_spot))

            iv = 0.2
            if "market_price" in opt:
                iv_calc = solve_option_implied_volatility(
                    opt["market_price"],
                    opt["S"],
                    opt["K"],
                    opt["T"],
                    opt["r"],
                    opt["type"],
                )
                if iv_calc is not None:
                    iv = iv_calc
                else:
                    iv = DEFAULT_FALLBACK_VOLATILITY

            vol_shock = np.random.lognormal(mean=0, sigma=vol_shock_sigma) * np.exp(z_vol * vol_shock_sigma)
            sigma_shocks.append(iv * vol_shock)
            s_t = opt["S"] * np.exp((opt["r"] - 0.5 * iv**2) * T_sim + iv * np.sqrt(T_sim) * z_spot)
            shocked_opt = opt.copy()
            shocked_opt["S"] = s_t
            shocked_portfolio.append(shocked_opt)

        shocked_val = sum(
            price_option_monte_carlo(opt, n_sim=500, n_steps=n_steps) * opt["qty"]
            for opt, _sigma in zip(shocked_portfolio, sigma_shocks)
        )
        pnl.append(shocked_val - base_val)

    return {"pnl": np.array(pnl), "shocks": shocks_dict}
