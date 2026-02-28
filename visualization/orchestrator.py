from typing import Dict, List

import numpy as np

from .tools.hedging_plotter import HedgingPlotter
from .tools.sensitivity_plotter import SensitivityPlotter

class VisualizationOrchestrator:
    def __init__(self) -> None:
        self.hedging_plotter = HedgingPlotter()
        self.sensitivity_plotter = SensitivityPlotter()

    def render_hedging_report(
        self,
        output_dir: str,
        pnl_original: np.ndarray,
        pnl_delta: np.ndarray,
        pnl_gamma_delta: np.ndarray,
        pnl_vega: np.ndarray,
        var_es_levels: Dict[str, float],
    ) -> None:
        self.hedging_plotter.plot_pnl_distribution_comparison(
            output_dir=output_dir,
            file_name="histogram_compare_mc.png",
            pnl_original=pnl_original,
            pnl_delta=pnl_delta,
            pnl_gamma_delta=pnl_gamma_delta,
            pnl_vega=pnl_vega,
            var_es_levels=var_es_levels,
        )

    def render_sensitivity_report(
        self,
        output_dir: str,
        x_range: np.ndarray,
        x_label: str,
        title: str,
        file_name: str,
        csv_name: str,
        series_by_strategy: Dict[str, List[float]],
        rows: List[Dict],
    ) -> None:
        self.sensitivity_plotter.plot_sensitivity_curves(
            output_dir=output_dir,
            x_range=x_range,
            x_label=x_label,
            title=title,
            file_name=file_name,
            series_by_strategy=series_by_strategy,
        )
        self.sensitivity_plotter.save_sensitivity_table(
            output_dir=output_dir,
            csv_name=csv_name,
            rows=rows,
        )
