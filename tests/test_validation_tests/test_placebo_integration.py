"""Integration tests for PlaceboTest with validation orchestrator."""

import numpy as np
import pandas as pd
import pytest

from mmm_eval.adapters.base import BaseAdapter, PrimaryMediaRegressor
from mmm_eval.core.validation_test_orchestrator import ValidationTestOrchestrator
from mmm_eval.core.validation_tests_models import ValidationTestNames


class MockAdapter(BaseAdapter):
    """Mock adapter for testing PlaceboTest integration."""

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


class TestPlaceboTestIntegration:
    """Integration tests for PlaceboTest."""

    def test_placebo_test_with_orchestrator(self):
        """Test that PlaceboTest works with the validation orchestrator."""
        # Create test data
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "response": np.random.randn(10) + 100,
            "revenue": np.random.randn(10) + 1000,
            "tv_spend": np.random.randn(10) + 50,
            "radio_spend": np.random.randn(10) + 30,
        })

        # Create adapter and orchestrator
        adapter = MockAdapter(date_column="date")
        orchestrator = ValidationTestOrchestrator()
        
        # Run only the placebo test
        results = orchestrator.validate(
            adapter=adapter,
            data=data,
            test_names=[ValidationTestNames.PLACEBO],
        )
        
        # Verify the results
        assert ValidationTestNames.PLACEBO in results.test_results
        placebo_result = results.get_test_result(ValidationTestNames.PLACEBO)
        assert placebo_result.test_name == ValidationTestNames.PLACEBO
        assert placebo_result.test_scores.shuffled_channel_roi == 0.1
        assert placebo_result.test_scores.shuffled_channel_name.endswith("_shuffled")

    def test_placebo_test_in_dataframe_output(self):
        """Test that PlaceboTest results are properly included in DataFrame output."""
        # Create test data
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "response": np.random.randn(10) + 100,
            "revenue": np.random.randn(10) + 1000,
            "tv_spend": np.random.randn(10) + 50,
            "radio_spend": np.random.randn(10) + 30,
        })

        # Create adapter and orchestrator
        adapter = MockAdapter(date_column="date")
        orchestrator = ValidationTestOrchestrator()
        
        # Run only the placebo test
        results = orchestrator.validate(
            adapter=adapter,
            data=data,
            test_names=[ValidationTestNames.PLACEBO],
        )
        
        # Convert to DataFrame
        df = results.to_df()
        
        # Verify the DataFrame contains placebo test results
        placebo_rows = df[df["test_name"] == ValidationTestNames.PLACEBO.value]
        assert len(placebo_rows) == 1
        assert placebo_rows["general_metric_name"].iloc[0] == "shuffled_channel_roi"
        assert placebo_rows["metric_value"].iloc[0] == 0.1
        assert placebo_rows["metric_pass"].iloc[0] == True  # noqa: E712 