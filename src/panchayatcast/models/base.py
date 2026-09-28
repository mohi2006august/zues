"""Common model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.dataset import RegionContext


@dataclass
class VarPrediction:
    value: np.ndarray  # (n_days, n_cells)
    p10: np.ndarray | None = None
    p90: np.ndarray | None = None

    def copy(self) -> VarPrediction:
        return VarPrediction(
            self.value.copy(),
            None if self.p10 is None else self.p10.copy(),
            None if self.p90 is None else self.p90.copy(),
        )


Prediction = dict[str, VarPrediction]


class Downscaler(ABC):
    model_id: str
    name: str

    @abstractmethod
    def predict(
        self, ctx: RegionContext, blk: dict[str, np.ndarray], dates: pd.DatetimeIndex
    ) -> Prediction:
        """Map block fields {var: (n_days, n_block)} to cell fields {var: (n_days, n_active)}."""
