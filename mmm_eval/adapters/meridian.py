"""Meridian adapter for MMM evaluation."""

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseAdapter


class MeridianAdapter(BaseAdapter):
    """Adapter for Google Meridian MMM framework."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the Meridian adapter.

        Args:
            config: Configuration dictionary

        """
        super().__init__(config)
        self.media_columns = config.get("media_columns", []) if config else []
        self.base_columns = config.get("base_columns", []) if config else []
        # TODO: Add Meridian-specific initialization

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the Meridian model to data.

        Args:
            data: Training data

        """
        # Placeholder implementation - simple linear combination
        self.is_fitted = True

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted model.

        Args:
            data: Input data for prediction

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
        # Placeholder implementation
        return pd.Series()
