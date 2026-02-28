import os
from typing import Tuple

import matplotlib.pyplot as plt

class BasePlotter:
    def __init__(self, fig_size: Tuple[int, int] = (12, 7)) -> None:
        self.fig_size = fig_size

    def _ensure_output_dir(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

    def _save_figure(self, output_dir: str, file_name: str) -> None:
        self._ensure_output_dir(output_dir)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, file_name))
        plt.close()
