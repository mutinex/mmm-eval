"""Unit tests for PlaceboTest."""

import numpy as np
import pandas as pd

from mmm_eval.adapters.base import BaseAdapter, PrimaryMediaRegressor
from mmm_eval.core.validation_tests import PlaceboTest
from mmm_eval.core.validation_tests_models import ValidationTestNames
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
        # Set ROI results based on the data columns and current media channels
        roi_dict = {}
        for channel in self._media_channels:
            if channel.endswith("_shuffled"):
                # Give shuffled channels a low ROI (should pass threshold of -50%)
                roi_dict[channel] = -60.0
            else:
                # Give original channels normal ROI
                roi_dict[channel] = 1.5 if channel == "TV" else 2.0
        self._roi_results = pd.Series(roi_dict)

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

    def get_primary_media_regressor_columns_for_channels(self, channel_names: list[str]) -> list[str]:
        """Get the primary media regressor columns for specific channels."""
        return channel_names

    def _get_original_channel_columns(self, channel_name: str) -> dict[str, str]:
        """Get the original column names for a channel."""
        # For mock adapter, map channel names to their actual column names
        channel_mapping = {
            "TV": {"spend": "tv_spend"},
            "Radio": {"spend": "radio_spend"},
        }
        return channel_mapping.get(channel_name, {"spend": f"{channel_name.lower()}_spend"})

    def _get_shuffled_col_name(self, shuffled_channel_name: str, column_type: str) -> str:
        """Get the name for a shuffled column based on the mock adapter's naming convention."""
        # For mock adapter, use the same convention as Meridian (with suffix)
        return f"{shuffled_channel_name}_{column_type}"

    def _create_adapter_with_placebo_channel(
        self,
        shuffled_channel: str,
    ) -> "MockAdapter":
        """Create a new adapter instance configured to use the placebo channel."""
        new_adapter = MockAdapter(self.date_column)
        new_adapter.channel_spend_columns = self.channel_spend_columns + [f"{shuffled_channel}_spend"]
        new_adapter._media_channels = self._media_channels + [shuffled_channel]
        return new_adapter


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
        data = pd.DataFrame(
            {
                "date": dates,
                "response": np.random.randn(10) + 100,
                "revenue": np.random.randn(10) + 1000,
                "tv_spend": np.random.randn(10) + 50,
                "radio_spend": np.random.randn(10) + 30,
            }
        )

        # Create adapter and test
        adapter = MockAdapter(date_column="date")
        test = PlaceboTest(date_column="date")

        # Run the test
        result = test.run(adapter, data)

        # Verify the result
        assert result.test_name == ValidationTestNames.PLACEBO
        assert isinstance(result.test_scores, PlaceboMetricResults)
        assert result.test_scores.shuffled_channel_roi == -60.0
        assert result.test_scores.shuffled_channel_name.endswith("_shuffled")
        assert len(result.metric_names) == 1
        assert "shuffled_channel_roi" in result.metric_names

    def test_placebo_test_metric_threshold(self):
        """Test that the metric threshold checking works correctly."""
        # Create metric results with low ROI (should pass)
        results = PlaceboMetricResults(shuffled_channel_roi=-60.0, shuffled_channel_name="TV_shuffled")

        # Test threshold checking
        # ROI of -60% should pass (≤ -50% threshold)
        assert results._check_metric_threshold("shuffled_channel_roi", -60.0) is True
        # ROI of -30% should fail (> -50% threshold)
        assert results._check_metric_threshold("shuffled_channel_roi", -30.0) is False

    def test_placebo_test_dataframe_output(self):
        """Test that the metric results can be converted to DataFrame."""
        results = PlaceboMetricResults(shuffled_channel_roi=-60.0, shuffled_channel_name="TV_shuffled")

        df = results.to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "shuffled_channel_roi" in df["general_metric_name"].values
        assert df["metric_value"].iloc[0] == -60.0
        assert df["metric_pass"].iloc[0] == True  # noqa: E712

    def test_placebo_test_with_impressions_regressors(self):
        """Test that PlaceboTest correctly handles impressions regressors."""

        # Create a mock adapter that simulates Meridian with impressions regressors
        class MeridianStyleAdapter(BaseAdapter):
            def __init__(self):
                self.date_column = "date"
                self.channel_spend_columns = ["tv_spend", "radio_spend"]
                self._media_channels = ["TV", "Radio"]
                self.is_fitted = False
                self._roi_results = None

            @property
            def media_channels(self) -> list[str]:
                return self._media_channels

            @property
            def primary_media_regressor_type(self) -> PrimaryMediaRegressor:
                return PrimaryMediaRegressor.IMPRESSIONS

            @property
            def primary_media_regressor_columns(self) -> list[str]:
                return ["tv_impressions", "radio_impressions"]

            def fit(self, data: pd.DataFrame) -> None:
                self.is_fitted = True
                roi_dict = {}
                for channel in self._media_channels:
                    if channel.endswith("_shuffled"):
                        roi_dict[channel] = -60.0
                    else:
                        roi_dict[channel] = 1.5 if channel == "TV" else 2.0
                self._roi_results = pd.Series(roi_dict)

            def predict(self, data: pd.DataFrame | None = None) -> np.ndarray:
                return np.array([1.0, 2.0, 3.0])

            def fit_and_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
                return np.array([1.0, 2.0])

            def fit_and_predict_in_sample(self, data: pd.DataFrame) -> np.ndarray:
                return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

            def get_channel_roi(self, start_date=None, end_date=None) -> pd.Series:
                if not self.is_fitted:
                    raise RuntimeError("Model must be fit before computing ROI")
                return self._roi_results

            def get_channel_names(self) -> list[str]:
                return self.media_channels

            def copy(self) -> "MeridianStyleAdapter":
                new_adapter = MeridianStyleAdapter()
                new_adapter.channel_spend_columns = self.channel_spend_columns.copy()
                new_adapter._media_channels = self._media_channels.copy()
                return new_adapter

            def get_primary_media_regressor_columns_for_channels(self, channel_names: list[str]) -> list[str]:
                return [f"{channel.lower()}_impressions" for channel in channel_names]

            def _get_original_channel_columns(self, channel_name: str) -> dict[str, str]:
                """Get the original column names for a channel."""
                # For this mock adapter, map channel names to their spend and impressions columns
                channel_mapping = {
                    "TV": {"spend": "tv_spend", "impressions": "tv_impressions"},
                    "Radio": {"spend": "radio_spend", "impressions": "radio_impressions"},
                }
                return channel_mapping.get(
                    channel_name,
                    {"spend": f"{channel_name.lower()}_spend", "impressions": f"{channel_name.lower()}_impressions"},
                )

            def _get_shuffled_col_name(self, shuffled_channel_name: str, column_type: str) -> str:
                """Get the name for a shuffled column based on the mock adapter's naming convention."""
                # For this mock adapter, use the same convention as Meridian (with suffix)
                return f"{shuffled_channel_name}_{column_type}"

            def _create_adapter_with_placebo_channel(
                self,
                shuffled_channel: str,
            ) -> "MeridianStyleAdapter":
                """Create a new adapter instance configured to use the placebo channel."""
                new_adapter = MeridianStyleAdapter()
                new_adapter.channel_spend_columns = self.channel_spend_columns + [f"{shuffled_channel.lower()}_spend"]
                new_adapter._media_channels = self._media_channels + [shuffled_channel]
                return new_adapter

        # Create test data with both spend and impressions columns
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            {
                "date": dates,
                "response": np.random.randn(10) + 100,
                "revenue": np.random.randn(10) + 1000,
                "tv_spend": np.random.randn(10) + 50,
                "radio_spend": np.random.randn(10) + 30,
                "tv_impressions": np.random.randn(10) + 50000,
                "radio_impressions": np.random.randn(10) + 25000,
            }
        )

        # Create adapter and test
        adapter = MeridianStyleAdapter()
        test = PlaceboTest(date_column="date")

        # Run the test
        result = test.run(adapter, data)

        # Verify the result
        assert result.test_name == ValidationTestNames.PLACEBO
        assert isinstance(result.test_scores, PlaceboMetricResults)
        assert result.test_scores.shuffled_channel_roi == -60.0
        assert result.test_scores.shuffled_channel_name.endswith("_shuffled")

        # The test should have worked correctly, indicating that the PlaceboTest
        # can handle impressions regressors properly by shuffling the correct columns
        # and using the adapter's column naming logic
