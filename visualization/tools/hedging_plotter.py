from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .base_plotter import BasePlotter

class HedgingPlotter(BasePlotter):
    def __init__(self) -> None:
        super().__init__(fig_size=(14, 8))

    def plot_pnl_distribution_comparison(
        self,
        output_dir: str,
        file_name: str,
        pnl_original: np.ndarray,
        pnl_delta: np.ndarray,
        pnl_gamma_delta: np.ndarray,
        pnl_vega: np.ndarray,
        var_es_levels: Dict[str, float],
    ) -> None:
        plt.figure(figsize=self.fig_size)
        plt.hist(pnl_original, bins=50, color="skyblue", edgecolor="k", alpha=0.5, density=True, label="Original")
        plt.hist(pnl_delta, bins=50, color="orange", edgecolor="k", alpha=0.5, density=True, label="Delta Hedge")
        plt.hist(
            pnl_gamma_delta,
            bins=50,
            color="green",
            edgecolor="k",
            alpha=0.5,
            density=True,
            label="Gamma+Delta Hedge",
        )
        plt.hist(pnl_vega, bins=50, color="purple", edgecolor="k", alpha=0.5, density=True, label="Vega Hedge")

        plt.axvline(-var_es_levels["var_original"], color="blue", linestyle="--", label=f"VaR Original ({-var_es_levels['var_original']:.0f})")
        plt.axvline(-var_es_levels["es_original"], color="blue", linestyle=":", label=f"ES Original ({-var_es_levels['es_original']:.0f})")
        plt.axvline(-var_es_levels["var_delta"], color="orange", linestyle="--", label=f"VaR Delta ({-var_es_levels['var_delta']:.0f})")
        plt.axvline(-var_es_levels["es_delta"], color="orange", linestyle=":", label=f"ES Delta ({-var_es_levels['es_delta']:.0f})")
        plt.axvline(
            -var_es_levels["var_gamma_delta"],
            color="green",
            linestyle="--",
            label=f"VaR Gamma+Delta ({-var_es_levels['var_gamma_delta']:.0f})",
        )
        plt.axvline(
            -var_es_levels["es_gamma_delta"],
            color="green",
            linestyle=":",
            label=f"ES Gamma+Delta ({-var_es_levels['es_gamma_delta']:.0f})",
        )
        plt.axvline(-var_es_levels["var_vega"], color="purple", linestyle="--", label=f"VaR Vega ({-var_es_levels['var_vega']:.0f})")
        plt.axvline(-var_es_levels["es_vega"], color="purple", linestyle=":", label=f"ES Vega ({-var_es_levels['es_vega']:.0f})")

        plt.title("P&L Distribution Comparison (MONTE CARLO)")
        plt.xlabel("P&L")
        plt.ylabel("Density")
        plt.legend(loc="upper left", fontsize=10, ncol=2)
        plt.grid(True, alpha=0.3)
        self._save_figure(output_dir, file_name)
