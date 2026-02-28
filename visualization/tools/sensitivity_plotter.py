import os

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base_plotter import BasePlotter

class SensitivityPlotter(BasePlotter):
    def plot_sensitivity_curves(
        self,
        output_dir: str,
        x_range: np.ndarray,
        x_label: str,
        title: str,
        file_name: str,
        series_by_strategy: Dict[str, List[float]],
    ) -> None:
        plt.figure(figsize=self.fig_size)
        for strategy_name, values in series_by_strategy.items():
            plt.plot(x_range, values, label=strategy_name)
        plt.xlabel(x_label)
        plt.ylabel("Portfolio Value")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        self._save_figure(output_dir, file_name)

    def save_sensitivity_table(self, output_dir: str, csv_name: str, rows: List[Dict]) -> None:
        self._ensure_output_dir(output_dir)
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, csv_name), index=False)
