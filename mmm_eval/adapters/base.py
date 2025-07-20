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

        Returns
            A new adapter instance with the same configuration as this one

        """
        pass

    @abstractmethod
    def get_primary_media_regressor_columns_for_channels(self, channel_names: list[str]) -> list[str]:
        """Get the primary media regressor columns for specific channels.

        This method returns the column names that should be used as primary media
        regressors for the given channels. This is useful for tests that need to perturb
        specific channels.

        Args:
            channel_names: List of channel names to get regressor columns for

        Returns:
            List of column names that are used as primary media regressors for the given
            channels

        """
        pass

    def add_placebo_channel(
        self, original_channel_name: str, data_to_shuffle: pd.DataFrame, shuffled_indices: np.ndarray
    ) -> tuple["BaseAdapter", pd.DataFrame]:
        """Template method for adding a placebo channel to the adapter and data.

        This method creates a shuffled version of an existing channel and returns a new
        adapter configured to use the shuffled channel. The template method pattern
        ensures consistent behavior across adapters while allowing adapter-specific
        implementations.

        Args:
            original_channel_name: Name of the original channel to create a placebo version
                of
            data_to_shuffle: DataFrame containing the data to add shuffled columns to
            shuffled_indices: Array of shuffled indices to use for creating the placebo
                channel

        Returns:
            Tuple of (new_adapter, updated_data) where:
            - new_adapter: A new adapter instance configured to use the placebo channel
            - updated_data: DataFrame with the shuffled columns added

        """
        # Step 1: Get original columns to shuffle (subclass-specific)
        original_columns = self._get_original_channel_columns(original_channel_name)

        # Step 2: Add shuffled columns to the data
        shuffled_channel_name = f"{original_channel_name}_shuffled"
        updated_data = self._create_shuffled_columns(
            data_to_shuffle, original_columns, shuffled_indices, shuffled_channel_name
        )

        # Step 3: Create new adapter (subclass-specific)
        new_adapter = self._create_adapter_with_placebo_channel(
            original_channel_name, shuffled_channel_name, original_columns
        )

        return new_adapter, updated_data

    def _create_shuffled_columns(
        self,
        data: pd.DataFrame,
        original_columns: dict[str, str],
        shuffled_indices: np.ndarray,
        shuffled_channel_name: str,
    ) -> pd.DataFrame:
        """Create shuffled columns in the data for the placebo channel.

        This is a common implementation that creates shuffled versions of the original columns.
        The column naming is handled by each adapter according to their conventions.

        Args:
            data: DataFrame to add shuffled columns to
            original_columns: Dictionary mapping column types to original column names
            shuffled_indices: Array of shuffled indices to use
            shuffled_channel_name: Name for the new shuffled channel

        Returns:
            DataFrame with shuffled columns added

        """
        updated_data = data.copy()

        for column_type, original_col in original_columns.items():
            if original_col in data.columns:
                # Let each adapter determine the column naming convention
                shuffled_col = self._get_shuffled_col_name(shuffled_channel_name, column_type, original_col)
                updated_data[shuffled_col] = data[original_col].iloc[shuffled_indices].values

        return updated_data

    @abstractmethod
    def _get_shuffled_col_name(self, shuffled_channel_name: str, column_type: str, original_col: str) -> str:
        """Get the name for a shuffled column based on the adapter's naming convention.

        This method should be implemented by each adapter to return the correct column name
        for the shuffled channel according to their naming conventions.

        Args:
            shuffled_channel_name: Name of the shuffled channel
            column_type: Type of column (e.g., "spend", "impressions")
            original_col: Original column name

        Returns:
            Name for the shuffled column

        """
        pass

    @abstractmethod
    def _get_original_channel_columns(self, channel_name: str) -> dict[str, str]:
        """Get the original column names for a channel.

        This method should return a dictionary mapping column types to actual column names
        in the data. For example:
        - {"spend": "tv_spend", "impressions": "tv_impressions"} for impressions-based models
        - {"spend": "tv_spend"} for spend-only models

        Args:
            channel_name: Name of the channel to get columns for

        Returns:
            Dictionary mapping column types to actual column names in the data

        """
        pass

    @abstractmethod
    def _create_adapter_with_placebo_channel(
        self, original_channel: str, shuffled_channel: str, original_columns: dict[str, str]
    ) -> "BaseAdapter":
        """Create a new adapter instance configured to use the placebo channel.

        This method should create a new adapter instance that includes the placebo channel
        in its configuration. The adapter should be configured to use the shuffled column
        names that were created in the data.

        Args:
            original_channel: Name of the original channel
            shuffled_channel: Name of the new shuffled channel
            original_columns: Dictionary mapping column types to original column names

        Returns:
            New adapter instance configured to use the placebo channel

        """
        pass
