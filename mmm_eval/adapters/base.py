"""Base adapter class for MMM frameworks."""

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd


class PrimaryMediaRegressor(StrEnum):
    """Enum for primary media regressor types used in MMM frameworks."""

    SPEND = "spend"
    IMPRESSIONS = "impressions"
    REACH_AND_FREQUENCY = "reach_and_frequency"


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

    @property
    @abstractmethod
    def media_channels(self) -> list[str]:
        """Return the channel names used by this adapter.

        This property provides a consistent way to get channel names across different adapters.
        For most frameworks, this will be human-readable channel names, but for PyMC it may
        be the column names themselves.

        Returns
            List of channel names used by this adapter

        """
        pass

    @property
    @abstractmethod
    def primary_media_regressor_type(self) -> PrimaryMediaRegressor:
        """Return the type of primary media regressors used by this adapter.

        This property indicates what type of regressors are used as primary inputs
        to the model, which determines what should be perturbed in tests.

        Returns
            PrimaryMediaRegressor enum value indicating the type of primary media regressors

        """
        pass

    @property
    @abstractmethod
    def primary_media_regressor_columns(self) -> list[str]:
        """Return the primary media regressor columns that should be perturbed in tests.

        This property returns the columns that are actually used as regressors in the model.
        For most frameworks, this will be the spend columns, but for e.g. Meridian it could
        be impressions or reach/frequency columns depending on the configuration.

        Returns
            List of column names that are used as primary media regressors in the model

        """
        pass

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the model to the data.

        Args:
            data: Training data

        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame | None = None) -> np.ndarray:
        """Make predictions on new data.

        Args:
            data: Input data for prediction. Behavior varies by adapter:
                - Some adapters (e.g., PyMC) require this parameter and will raise
                  an error if None is passed
                - Other adapters (e.g., Meridian) ignore this parameter and use
                  the fitted model state instead

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
    def fit_and_predict_in_sample(self, data: pd.DataFrame) -> np.ndarray:
        """Fit the model on data and return predictions for the same data.

        Args:
            data: dataset to train model on and make predictions for

        Returns:
            Predicted values for the training data.

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

    @abstractmethod
    def get_channel_names(self) -> list[str]:  # pyright: ignore[reportReturnType]
        """Get the channel names that would be used as the index in channel ROI results.

        This method provides a consistent way to get channel names across different adapters
        without needing to call get_channel_roi() (which requires the model to be fitted).

        Returns
            List of channel names that would be used as the index in get_channel_roi results

        """
        pass

    @abstractmethod
    def copy(self) -> "BaseAdapter":
        """Create a deep copy of this adapter with all configuration.

        This method creates a complete copy of the adapter including all configuration,
        but without any fitted state (model, trace, etc.).

        Returns:
            A new adapter instance with the same configuration as this one

        """
        pass

    @abstractmethod
    def add_channels(self, new_channel_names: list[str]) -> dict[str, list[str]]:
        """Add new channels to the adapter's configuration.

        This method modifies the adapter's configuration to include additional channels.
        The adapter should not be fitted when this method is called.

        Args:
            new_channel_names: List of new channel names to add

        Returns:
            Dictionary mapping channel names to lists of column names that were added for each channel.
            For example: {"TV_shuffled": ["tv_shuffled_spend", "tv_shuffled_impressions"]}

        """
        pass

    @abstractmethod
    def get_primary_media_regressor_columns_for_channels(self, channel_names: list[str]) -> list[str]:
        """Get the primary media regressor columns for specific channels.

        This method returns the column names that should be used as primary media regressors
        for the given channels. This is useful for tests that need to perturb specific channels.

        Args:
            channel_names: List of channel names to get regressor columns for

        Returns:
            List of column names that are used as primary media regressors for the given channels

        """
        pass
