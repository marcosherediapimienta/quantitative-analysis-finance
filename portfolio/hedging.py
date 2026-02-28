from typing import Dict, List

import numpy as np

from .valuation import (
    option_greeks_monte_carlo,
    portfolio_greeks_monte_carlo,
    price_option_monte_carlo,
    solve_option_implied_volatility,
)
from risk.metrics import value_at_risk_expected_shortfall

def run_monte_carlo_hedging_analysis(
    portfolio: List[Dict],
    pnl_mc: np.ndarray,
    shocks_mc: Dict,
    base_value: float,
    var_mc: float,
    es_mc: float,
    n_sim_greeks: int,
    n_steps: int,
    simulation_horizon: float,
    vis_dir: str,
) -> Dict[str, float]:
    from visualization import VisualizationOrchestrator

    delta_hedge_fraction_mc = 0.7
    underlying_positions: Dict = {}
    for opt in portfolio:
        key = opt.get("ticker", opt["S"])
        greeks_mc = option_greeks_monte_carlo(opt, n_sim=n_sim_greeks, n_steps=n_steps)
        underlying_positions.setdefault(key, {"S0": opt["S"], "delta": 0.0})
        underlying_positions[key]["delta"] += greeks_mc["delta"] * opt["qty"] * delta_hedge_fraction_mc

    pnl_mc_hedged = []
    for i in range(len(pnl_mc)):
        hedge_pnl = 0.0
        for key, v in underlying_positions.items():
            s0 = v["S0"]
            delta = v["delta"]
            z = shocks_mc[key][i]
            for opt in portfolio:
                if opt.get("ticker", opt["S"]) == key:
                    iv = solve_option_implied_volatility(
                        opt["market_price"], opt["S"], opt["K"], opt["T"], opt["r"], opt["type"]
                    )
                    iv = iv if iv is not None else 0.2
                    r = opt["r"]
                    t_sim = simulation_horizon if simulation_horizon is not None else opt["T"]
                    break
            s_t = s0 * np.exp((r - 0.5 * iv**2) * t_sim + iv * np.sqrt(t_sim) * z)
            hedge_pnl += -delta * (s_t - s0)
        pnl_mc_hedged.append(pnl_mc[i] + hedge_pnl)
    pnl_mc_hedged = np.array(pnl_mc_hedged)
    var_mc_hedged, es_mc_hedged = value_at_risk_expected_shortfall(pnl_mc_hedged, alpha=0.01)

    greeks_total_mc = portfolio_greeks_monte_carlo(portfolio, n_sim=n_sim_greeks, n_steps=n_steps)
    portfolio_gamma_mc = greeks_total_mc["gamma"]
    hedge_opt_mc = {
        "type": "call",
        "style": "european",
        "S": portfolio[0]["S"],
        "K": portfolio[0]["K"],
        "T": portfolio[0]["T"],
        "r": portfolio[0]["r"],
        "qty": 0,
        "market_price": portfolio[0]["market_price"],
    }
    greeks_hedge_mc = option_greeks_monte_carlo(hedge_opt_mc, n_sim=n_sim_greeks, n_steps=n_steps)
    qty_gamma_hedge_mc = -portfolio_gamma_mc * 0.7 / greeks_hedge_mc["gamma"] if greeks_hedge_mc["gamma"] != 0 else 0
    hedge_opt_mc["qty"] = qty_gamma_hedge_mc
    portfolio_gamma_hedged_mc = portfolio + [hedge_opt_mc]
    delta_gamma_hedged_mc = portfolio_greeks_monte_carlo(portfolio_gamma_hedged_mc, n_sim=n_sim_greeks, n_steps=n_steps)["delta"]

    pnl_gamma_delta_hedged_mc = []
    for i in range(len(pnl_mc)):
        shocked_portfolio = []
        for opt in portfolio_gamma_hedged_mc:
            key = opt.get("ticker", opt["S"])
            s0 = opt["S"]
            iv = solve_option_implied_volatility(opt["market_price"], opt["S"], opt["K"], opt["T"], opt["r"], opt["type"])
            iv = iv if iv is not None else 0.2
            r = opt["r"]
            t_sim = simulation_horizon if simulation_horizon is not None else opt["T"]
            z = shocks_mc[key][i] if key in shocks_mc else np.random.normal(0, 1)
            s_t = s0 * np.exp((r - 0.5 * iv**2) * t_sim + iv * np.sqrt(t_sim) * z)
            shocked_opt = opt.copy()
            shocked_opt["S"] = s_t
            shocked_portfolio.append(shocked_opt)

        val = sum(price_option_monte_carlo(opt, n_sim=n_sim_greeks, n_steps=n_steps) for opt in shocked_portfolio)
        s0 = portfolio[0]["S"]
        z = shocks_mc[portfolio[0].get("ticker", portfolio[0]["S"])][i]
        s_t = s0 * np.exp((portfolio[0]["r"] - 0.5 * iv**2) * t_sim + iv * np.sqrt(t_sim) * z)
        hedge_pnl = -delta_gamma_hedged_mc * (s_t - s0)
        pnl_gamma_delta_hedged_mc.append(val - base_value + hedge_pnl)
    pnl_gamma_delta_hedged_mc = np.array(pnl_gamma_delta_hedged_mc)
    var_gamma_delta_hedged_mc, es_gamma_delta_hedged_mc = value_at_risk_expected_shortfall(pnl_gamma_delta_hedged_mc, alpha=0.01)

    vega_total_mc = portfolio_greeks_monte_carlo(portfolio, n_sim=n_sim_greeks, n_steps=n_steps)["vega"]
    hedge_opt_vega_mc = {
        "type": "call",
        "style": "european",
        "S": portfolio[0]["S"],
        "K": portfolio[0]["K"],
        "T": portfolio[0]["T"],
        "r": portfolio[0]["r"],
        "qty": 0,
        "market_price": portfolio[0]["market_price"],
    }
    greeks_hedge_vega_mc = option_greeks_monte_carlo(hedge_opt_vega_mc, n_sim=n_sim_greeks, n_steps=n_steps)
    qty_vega_hedge_mc = -vega_total_mc * 0.7 / greeks_hedge_vega_mc["vega"] if greeks_hedge_vega_mc["vega"] != 0 else 0
    hedge_opt_vega_mc["qty"] = qty_vega_hedge_mc

    pnl_vega_hedged_mc = []
    for i in range(len(pnl_mc)):
        hedge_pnl = 0.0
        for key, v in underlying_positions.items():
            s0 = v["S0"]
            delta = v["delta"]
            z = shocks_mc[key][i]
            for opt in portfolio:
                if opt.get("ticker", opt["S"]) == key:
                    iv = solve_option_implied_volatility(
                        opt["market_price"], opt["S"], opt["K"], opt["T"], opt["r"], opt["type"]
                    )
                    iv = iv if iv is not None else 0.2
                    r = opt["r"]
                    t_sim = simulation_horizon if simulation_horizon is not None else opt["T"]
                    break
            s_t = s0 * np.exp((r - 0.5 * iv**2) * t_sim + iv * np.sqrt(t_sim) * z)
            hedge_pnl += -delta * (s_t - s0)
        pnl_vega_hedged_mc.append(pnl_mc[i] + hedge_pnl)
    pnl_vega_hedged_mc = np.array(pnl_vega_hedged_mc)
    var_vega_hedged_mc, es_vega_hedged_mc = value_at_risk_expected_shortfall(pnl_vega_hedged_mc, alpha=0.01)

    VisualizationOrchestrator().render_hedging_report(
        output_dir=vis_dir,
        pnl_original=pnl_mc,
        pnl_delta=pnl_mc_hedged,
        pnl_gamma_delta=pnl_gamma_delta_hedged_mc,
        pnl_vega=pnl_vega_hedged_mc,
        var_es_levels={
            "var_original": float(var_mc),
            "es_original": float(es_mc),
            "var_delta": float(var_mc_hedged),
            "es_delta": float(es_mc_hedged),
            "var_gamma_delta": float(var_gamma_delta_hedged_mc),
            "es_gamma_delta": float(es_gamma_delta_hedged_mc),
            "var_vega": float(var_vega_hedged_mc),
            "es_vega": float(es_vega_hedged_mc),
        },
    )

    return {
        "var_delta": float(var_mc_hedged),
        "es_delta": float(es_mc_hedged),
        "var_gamma_delta": float(var_gamma_delta_hedged_mc),
        "es_gamma_delta": float(es_gamma_delta_hedged_mc),
        "var_vega": float(var_vega_hedged_mc),
        "es_vega": float(es_vega_hedged_mc),
    }
