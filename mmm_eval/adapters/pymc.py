"""
PyMC MMM framework adapter.

N.B. expects control variables to be scaled to 0-1 using maxabs scaling.
"""

from typing import Dict

import pandas as pd

from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.configs import PyMCConfig
from pymc_marketing.mmm import MMM

# TODO: update this class once PyMCAdapter is promoted out of experimental
class PyMCAdapter(BaseAdapter):
    def __init__(self, config: PyMCConfig, data: pd.DataFrame):
        self.config = config
        self.data = data
        self.is_fitted = False

    def fit(self: str):
        X = self.data.drop(columns=[self.config.target_column])
        y = self.data[self.config.target_column]

        MMM(**self.config.model_config.config).fit(X=X, y=y, **self.config.fit_config.config)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        pass

    def get_channel_roi(self) -> Dict[str, float]:
        """Return the ROIs for each channel."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before computing ROI.")
        return self._channel_rois

    def _compute_rois(self, data: pd.DataFrame):
        """Estimate ROIs based on the marginal contribution of each channel."""
        pass
