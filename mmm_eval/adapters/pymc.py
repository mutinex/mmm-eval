"""Legacy PyMC adapter for MMM evaluation."""

from typing import Any

import numpy as np
import pandas as pd

from mmm_eval.adapters.base import BaseAdapter


# TODO: update this class once PyMCAdapter is promoted out of experimental
class PyMCAdapter(BaseAdapter):
    """Legacy adapter for PyMC MMM framework."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the legacy PyMC adapter.

        Args:
            config: Configuration dictionary

        """
        super().__init__(config)
        self._channel_rois: dict[str, float] = {}

    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the PyMC model to data.

        Args:
            data: Training data
            **kwargs: Additional fitting parameters

        """
        # Placeholder implementation
        self.is_fitted = True

    def predict(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """Make predictions using the fitted model.

        Args:
            data: Input data for prediction
            **kwargs: Additional prediction parameters

        Returns:
            Predicted values

        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before prediction")
        # Placeholder implementation
        return np.zeros(len(data))

    def get_channel_roi(
        self,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.Series:
        """Get channel ROI estimates.

        Args:
            start_date: Optional start date for ROI calculation
            end_date: Optional end date for ROI calculation

        Returns:
            Series containing ROI estimates for each channel

        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before computing ROI")
        return pd.Series(self._channel_rois)

    def _compute_rois(self, data: pd.DataFrame) -> None:
        """Estimate ROIs based on the marginal contribution of each channel.

        Args:
            data: Input data for ROI calculation

        """
        pass
