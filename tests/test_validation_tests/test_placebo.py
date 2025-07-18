"""Unit tests for PlaceboTest."""

import numpy as np
import pandas as pd
import pytest

from mmm_eval.adapters.base import BaseAdapter, PrimaryMediaRegressor
from mmm_eval.core.validation_tests import PlaceboTest
from mmm_eval.core.validation_tests_models import ValidationTestNames
from mmm_eval.data.constants import InputDataframeConstants
from mmm_eval.metrics.metric_models import PlaceboMetricResults


class MockAdapter(BaseAdapter):
    """Mock adapter for testing PlaceboTest."""

    def __init__(self, date_column: str = "date"):
        """Initialize mock adapter."""
        self.date_column = date_column
        self.channel_spend_columns = ["tv_spend", "radio_spend"]
        self._media_channels = ["TV", "Radio"]
        self.is_fitted = False
        self._roi_results = None

    @property
    def media_channels(self) -> list[str]:
        """Return the channel names used by this adapter."""
        return self._media_channels

    def fit(self, data: pd.DataFrame) -> None:
        """Mock fit method."""
        self.is_fitted = True
        # Set ROI results based on the data columns
        if "tv_spend_shuffled" in data.columns:
            # If shuffled channel is present, give it a low ROI
            self._roi_results = pd.Series({"TV": 1.5, "Radio": 2.0, "TV_shuffled": 0.1})
        else:
            self._roi_results = pd.Series({"TV": 1.5, "Radio": 2.0})

    def predict(self, data: pd.DataFrame | None = None) -> np.ndarray:
        """Mock predict method."""
        return np.array([1.0, 2.0, 3.0])

    def fit_and_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """Mock fit_and_predict method."""
        return np.array([1.0, 2.0])

    def fit_and_predict_in_sample(self, data: pd.DataFrame) -> np.ndarray:
        """Mock fit_and_predict_in_sample method."""
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def get_channel_roi(self, start_date=None, end_date=None) -> pd.Series:
        """Mock get_channel_roi method."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before computing ROI")
        return self._roi_results

    def get_channel_names(self) -> list[str]:
        """Mock get_channel_names method."""
        return self.media_channels

    @property
    def primary_media_regressor_type(self) -> PrimaryMediaRegressor:
        """Return the type of primary media regressors."""
        return PrimaryMediaRegressor.SPEND

    @property
    def primary_media_regressor_columns(self) -> list[str]:
        """Return the primary media regressor columns."""
        return self.channel_spend_columns

    def copy(self) -> "MockAdapter":
        """Create a deep copy of this adapter."""
        new_adapter = MockAdapter(self.date_column)
        new_adapter.channel_spend_columns = self.channel_spend_columns.copy()
        new_adapter._media_channels = self._media_channels.copy()
        return new_adapter

    def add_channels(self, new_channel_names: list[str]) -> None:
        """Add new channels to the adapter."""
        if self.is_fitted:
            raise RuntimeError("Cannot add channels to a fitted adapter")
        
        # For mock adapter, assume channel names are the same as column names
        self.channel_spend_columns.extend(new_channel_names)
        self._media_channels.extend(new_channel_names)


class TestPlaceboTest:
    """Test cases for PlaceboTest."""

    def test_placebo_test_creation(self):
        """Test that PlaceboTest can be created."""
        test = PlaceboTest(date_column="date")
        assert test.test_name == ValidationTestNames.PLACEBO

    def test_placebo_test_run(self):
        """Test that PlaceboTest runs successfully."""
        # Create test data
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "response": np.random.randn(10) + 100,
            "revenue": np.random.randn(10) + 1000,
            "tv_spend": np.random.randn(10) + 50,
            "radio_spend": np.random.randn(10) + 30,
        })

        # Create adapter and test
        adapter = MockAdapter(date_column="date")
        test = PlaceboTest(date_column="date")
        
        # Run the test
        result = test.run(adapter, data)
        
        # Verify the result
        assert result.test_name == ValidationTestNames.PLACEBO
        assert isinstance(result.test_scores, PlaceboMetricResults)
        assert result.test_scores.shuffled_channel_roi == 0.1
        assert result.test_scores.shuffled_channel_name.endswith("_shuffled")
        assert len(result.metric_names) == 1
        assert "shuffled_channel_roi" in result.metric_names

    def test_placebo_test_metric_threshold(self):
        """Test that the metric threshold checking works correctly."""
        # Create metric results with low ROI (should pass)
        results = PlaceboMetricResults(
            shuffled_channel_roi=0.1,
            shuffled_channel_name="TV_shuffled"
        )
        
        # Test threshold checking
        assert results._check_metric_threshold("shuffled_channel_roi", 0.1) is True
        assert results._check_metric_threshold("shuffled_channel_roi", 0.3) is False

    def test_placebo_test_dataframe_output(self):
        """Test that the metric results can be converted to DataFrame."""
        results = PlaceboMetricResults(
            shuffled_channel_roi=0.1,
            shuffled_channel_name="TV_shuffled"
        )
        
        df = results.to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "shuffled_channel_roi" in df["general_metric_name"].values
        assert df["metric_value"].iloc[0] == 0.1
        assert df["metric_pass"].iloc[0] == True  # noqa: E712 