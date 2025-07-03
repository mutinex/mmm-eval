"""Base adapter class for MMM frameworks."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseAdapter(ABC):
    """Base class for MMM framework adapters."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the base adapter.

        Args:
            config: Configuration dictionary

        """
        self.config = config or {}
        self.is_fitted = False
        self.channel_spend_columns: list[str] = []
        self.date_column: str

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the model to the data.

        Args:
            data: Training data

        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.

        Args:
            data: Input data for prediction

        Returns:
            Predicted values

        """
        pass


    @abstractmethod
    def fit_and_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """Fit on training data and make predictions on test data.

        Args:
            data: Input data for prediction

        Returns:
            Predicted values

        """
        pass

    @abstractmethod
    def fit_and_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """Fit on training data and make predictions on test data.

        Args:
            train: dataset to train model on
            test: dataset to make predictions using

        Returns:
            Predicted values.

        """
        pass

    @abstractmethod
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
        pass
