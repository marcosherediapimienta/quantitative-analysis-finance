import copy
from typing import Dict, List

import numpy as np

from models import BlackScholesPricer
from .valuation import (
    option_greeks_monte_carlo,
    portfolio_greeks_monte_carlo,
    price_option_monte_carlo,
    solve_option_implied_volatility,
)


def run_monte_carlo_sensitivity_analysis(portfolio: List[Dict], n_steps: int, n_sim_sens: int, vis_dir: str) -> None:
    from visualization import VisualizationOrchestrator

    n_sim_greeks = n_sim_sens
    visualization_orchestrator = VisualizationOrchestrator()

    def price_portfolio(portfolio_slice: List[Dict], n_sim: int, steps: int) -> float:
        return sum(price_option_monte_carlo(opt, n_sim=n_sim, n_steps=steps) * opt["qty"] for opt in portfolio_slice)

    base_portfolio = portfolio[0]
    hedge_opt_mc = {
        "type": "call",
        "style": "european",
        "S": base_portfolio["S"],
        "K": base_portfolio["K"],
        "T": base_portfolio["T"],
        "r": base_portfolio["r"],
        "qty": 0,
        "market_price": base_portfolio["market_price"],
    }
    greeks_hedge_mc = option_greeks_monte_carlo(hedge_opt_mc, n_sim=n_sim_greeks, n_steps=n_steps)
    greeks_hedge_vega_mc = option_greeks_monte_carlo(hedge_opt_mc, n_sim=n_sim_greeks, n_steps=n_steps)

    hedge_strategies = [
        ("Original", copy.deepcopy(portfolio)),
        (
            "Delta Hedge",
            copy.deepcopy(portfolio)
            + [
                {
                    "type": "call",
                    "style": "european",
                    "S": base_portfolio["S"],
                    "K": base_portfolio["K"],
                    "T": base_portfolio["T"],
                    "r": base_portfolio["r"],
                    "qty": -portfolio_greeks_monte_carlo(portfolio, n_sim=n_sim_greeks, n_steps=n_steps)["delta"],
                    "market_price": base_portfolio["market_price"],
                }
            ],
        ),
        (
            "Gamma-Delta Hedge",
            copy.deepcopy(portfolio)
            + [
                {
                    "type": "call",
                    "style": "european",
                    "S": base_portfolio["S"],
                    "K": base_portfolio["K"],
                    "T": base_portfolio["T"],
                    "r": base_portfolio["r"],
                    "qty": -portfolio_greeks_monte_carlo(portfolio, n_sim=n_sim_greeks, n_steps=n_steps)["gamma"]
                    / greeks_hedge_mc["gamma"],
                    "market_price": base_portfolio["market_price"],
                }
            ],
        ),
        (
            "Vega Hedge",
            copy.deepcopy(portfolio)
            + [
                {
                    "type": "call",
                    "style": "european",
                    "S": base_portfolio["S"],
                    "K": base_portfolio["K"],
                    "T": base_portfolio["T"],
                    "r": base_portfolio["r"],
                    "qty": -portfolio_greeks_monte_carlo(portfolio, n_sim=n_sim_greeks, n_steps=n_steps)["vega"]
                    / greeks_hedge_vega_mc["vega"],
                    "market_price": base_portfolio["market_price"],
                }
            ],
        ),
    ]

    initial_portfolio_value = price_portfolio(portfolio, n_sim_greeks, n_steps)

    def plot_sensitivity(
        x_range: np.ndarray,
        x_label: str,
        title: str,
        file_name: str,
        csv_name: str,
        low_label: str,
        high_label: str,
    ) -> None:
        rows = []
        series_by_strategy = {}
        for name, port in hedge_strategies:
            values = []
            for x in x_range:
                port_mod = copy.deepcopy(port)
                if x_label == "Spot":
                    for opt in port_mod:
                        opt["S"] = x
                elif x_label == "Risk-free rate (r)":
                    for opt in port_mod:
                        opt["r"] = x
                elif x_label == "Volatility multiplier":
                    for opt in port_mod:
                        iv = solve_option_implied_volatility(
                            opt["market_price"], opt["S"], opt["K"], opt["T"], opt["r"], opt["type"]
                        )
                        iv = iv if iv is not None else 0.2
                        if opt["type"] == "call":
                            opt["market_price"] = BlackScholesPricer.call_price(
                                opt["S"], opt["K"], opt["T"], opt["r"], iv * x
                            )
                        else:
                            opt["market_price"] = BlackScholesPricer.put_price(
                                opt["S"], opt["K"], opt["T"], opt["r"], iv * x
                            )
                values.append(price_portfolio(port_mod, n_sim_sens, n_steps))

            series_by_strategy[name] = values
            base_idx, low_idx, high_idx = 10, 0, 20
            base = initial_portfolio_value if name == "Original" else values[base_idx]
            rows.append(
                {
                    "Strategy": name,
                    "Base": base,
                    low_label: values[low_idx],
                    high_label: values[high_idx],
                    f"Δ{low_label}": values[low_idx] - base,
                    f"Δ{high_label}": values[high_idx] - base,
                }
            )

        visualization_orchestrator.render_sensitivity_report(
            output_dir=vis_dir,
            x_range=x_range,
            x_label=x_label,
            title=title,
            file_name=file_name,
            csv_name=csv_name,
            series_by_strategy=series_by_strategy,
            rows=rows,
        )

    spot_base = base_portfolio["S"]
    plot_sensitivity(
        x_range=np.linspace(spot_base * 0.9, spot_base * 1.1, 21),
        x_label="Spot",
        title="Sensitivity to Spot - All Strategies",
        file_name="sensitivity_spot_mc.png",
        csv_name="sensitivity_spot_mc.csv",
        low_label="-10%",
        high_label="+10%",
    )

    r_base = base_portfolio["r"]
    plot_sensitivity(
        x_range=np.linspace(r_base - 0.01, r_base + 0.01, 21),
        x_label="Risk-free rate (r)",
        title="Sensitivity to r - All Strategies",
        file_name="sensitivity_r_mc.png",
        csv_name="sensitivity_r_mc.csv",
        low_label="-1%",
        high_label="+1%",
    )

    plot_sensitivity(
        x_range=np.linspace(0.8, 1.2, 21),
        x_label="Volatility multiplier",
        title="Sensitivity to Volatility - All Strategies",
        file_name="sensitivity_vol_mc.png",
        csv_name="sensitivity_vol_mc.csv",
        low_label="-20%",
        high_label="+20%",
    )
