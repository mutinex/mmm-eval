"""Integration tests for PlaceboTest with validation orchestrator."""

import numpy as np
import pandas as pd

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

    def get_primary_media_regressor_columns_for_channels(self, channel_names: list[str]) -> list[str]:
        """Get the primary media regressor columns for specific channels."""
        return channel_names

    def _get_original_channel_columns(self, channel_name: str) -> dict[str, str]:
        """Get the original column names for a channel."""
        # For mock adapter, map channel names to their actual column names
        # The data has columns like "tv_spend", "radio_spend"
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


class TestPlaceboTestIntegration:
    """Integration tests for PlaceboTest."""

    def test_placebo_test_with_orchestrator(self):
        """Test that PlaceboTest works with the validation orchestrator."""
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
        assert placebo_result.test_scores.shuffled_channel_roi == -60.0
        assert placebo_result.test_scores.shuffled_channel_name.endswith("_shuffled")

    def test_placebo_test_in_dataframe_output(self):
        """Test that PlaceboTest results are properly included in DataFrame output."""
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
        assert placebo_rows["metric_value"].iloc[0] == -60.0
        assert placebo_rows["metric_pass"].iloc[0] == True  # noqa: E712
