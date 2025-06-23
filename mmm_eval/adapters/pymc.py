"""
PyMC MMM framework adapter.

N.B. expects control variables to be scaled to 0-1 using maxabs scaling.
"""

from typing import Dict

import pandas as pd

from mmm_eval.adapters.base import BaseAdapter

# TODO: update this class once PyMCAdapter is promoted out of experimental
class PyMCAdapter(BaseAdapter):
    def __init__(self, config: dict):
        pass

    def fit(self: str):
        pass

    def predict(self) -> pd.Series:
        pass

    def get_channel_roi(self) -> Dict[str, float]:
        """Return the ROIs for each channel."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before computing ROI.")
        return self._channel_rois

    def _compute_rois(self, data: pd.DataFrame):
        """Estimate ROIs based on the marginal contribution of each channel."""
        pass
