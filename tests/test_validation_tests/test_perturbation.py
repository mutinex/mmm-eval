"""Tests for the PerturbationTest validation test."""

import numpy as np
import pandas as pd
import pytest

from mmm_eval.adapters.base import BaseAdapter, PrimaryMediaRegressor
from mmm_eval.core.validation_tests import PerturbationTest


class MockAdapter(BaseAdapter):
    """Mock adapter for testing PerturbationTest."""

    def __init__(
        self,
        primary_media_regressor_type: PrimaryMediaRegressor,
        primary_media_regressor_columns: list[str],
        channel_spend_columns: list[str] | None = None,
        media_channels: list[str] | None = None,
    ):
        """Initialize the MockAdapter.

        Args:
            primary_media_regressor_type: The type of primary media regressor.
            primary_media_regressor_columns: The columns used as primary media regressors.
            channel_spend_columns: The spend columns for channels.
            media_channels: The channel names.

        """
        self._primary_media_regressor_type = primary_media_regressor_type
        self._primary_media_regressor_columns = primary_media_regressor_columns
        self.channel_spend_columns = channel_spend_columns or []
        self._media_channels = media_channels  # can be None
        self.date_column = "date"
        self.is_fitted = False
        self._force_not_fitted_error = False

    @property
    def media_channels(self) -> list[str]:
        """Return the channel names used by this adapter."""
        if self._media_channels is not None:
            return self._media_channels
        return self.channel_spend_columns

    @property
    def primary_media_regressor_type(self) -> PrimaryMediaRegressor:
        """Return the type of primary media regressor."""
        return self._primary_media_regressor_type

    @property
    def primary_media_regressor_columns(self) -> list[str]:
        """Return the primary media regressor columns."""
        return self._primary_media_regressor_columns

    def fit(self, data: pd.DataFrame) -> None:
        """Mock fit method."""
        self.is_fitted = True

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
        if not self.is_fitted or self._force_not_fitted_error:
            raise RuntimeError("Model must be fit before computing ROI")
        channel_names = self.media_channels
        if self.primary_media_regressor_type == PrimaryMediaRegressor.REACH_AND_FREQUENCY:
            return pd.Series({ch: np.nan for ch in channel_names})
        return pd.Series({ch: 1.0 for ch in channel_names})

    def get_channel_names(self) -> list[str]:
        """Get the channel names that would be used as the index in get_channel_roi results."""
        return self.media_channels

    def copy(self) -> "MockAdapter":
        """Create a deep copy of this adapter."""
        new_adapter = MockAdapter(
            primary_media_regressor_type=self._primary_media_regressor_type,
            primary_media_regressor_columns=self._primary_media_regressor_columns.copy(),
            channel_spend_columns=self.channel_spend_columns.copy() if self.channel_spend_columns else None,
            media_channels=self._media_channels.copy() if self._media_channels else None,
        )
        return new_adapter

    def add_channels(self, new_channel_names: list[str]) -> dict[str, list[str]]:
        """Add new channels to the adapter."""
        if self.is_fitted:
            raise RuntimeError("Cannot add channels to a fitted adapter")

        # For mock adapter, assume channel names are the same as column names
        added_columns = {}
        for channel_name in new_channel_names:
            if self.channel_spend_columns is not None:
                self.channel_spend_columns.append(channel_name)
            if self._media_channels is not None:
                self._media_channels.append(channel_name)
            added_columns[channel_name] = [channel_name]

        return added_columns

    def get_primary_media_regressor_columns_for_channels(self, channel_names: list[str]) -> list[str]:
        """Get the primary media regressor columns for specific channels."""
        return channel_names

    def _get_original_channel_columns(self, channel_name: str) -> dict[str, str]:
        """Get the original column names for a channel."""
        # For mock adapter, assume channel names are the same as column names
        return {"spend": channel_name}

    def _get_shuffled_col_name(self, shuffled_channel_name: str, column_type: str, original_col: str) -> str:
        """Get the name for a shuffled column based on the mock adapter's naming convention."""
        # For mock adapter, use the same convention as Meridian (with suffix)
        return f"{shuffled_channel_name}_{column_type}"

    def _create_adapter_with_placebo_channel(
        self, original_channel: str, shuffled_channel: str, original_columns: dict[str, str]
    ) -> "MockAdapter":
        """Create a new adapter instance configured to use the placebo channel."""
        new_adapter = MockAdapter(
            primary_media_regressor_type=self._primary_media_regressor_type,
            primary_media_regressor_columns=self._primary_media_regressor_columns.copy(),
            channel_spend_columns=(
                self.channel_spend_columns + [f"{shuffled_channel}_spend"] if self.channel_spend_columns else None
            ),
            media_channels=self._media_channels + [shuffled_channel] if self._media_channels else None,
        )
        return new_adapter


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "response": np.random.randn(10) + 10,
            "revenue": np.random.randn(10) + 100,
            "tv_spend": np.random.randn(10) + 1000,
            "radio_spend": np.random.randn(10) + 500,
            "tv_impressions": np.random.randn(10) + 50000,
            "radio_impressions": np.random.randn(10) + 25000,
        }
    )


@pytest.fixture
def perturbation_test():
    """Create a PerturbationTest instance."""
    return PerturbationTest(date_column="date")


class TestPerturbationTest:
    """Test cases for PerturbationTest."""

    def test_perturbation_test_with_spend_regressors(self, perturbation_test, sample_data):
        """Test perturbation with spend regressors."""
        adapter = MockAdapter(
            primary_media_regressor_type=PrimaryMediaRegressor.SPEND,
            primary_media_regressor_columns=["tv_spend", "radio_spend"],
            channel_spend_columns=["tv_spend", "radio_spend"],
            media_channels=["tv", "radio"],
        )
        result = perturbation_test.run(adapter, sample_data)
        assert set(result.test_scores.percentage_change_for_each_channel.index) == {"tv", "radio"}
        assert (result.test_scores.percentage_change_for_each_channel == 0.0).all()

    def test_perturbation_test_with_impressions_regressors(self, perturbation_test, sample_data):
        """Test perturbation with impressions regressors."""
        adapter = MockAdapter(
            primary_media_regressor_type=PrimaryMediaRegressor.IMPRESSIONS,
            primary_media_regressor_columns=["tv_impressions", "radio_impressions"],
            channel_spend_columns=["tv_spend", "radio_spend"],
            media_channels=["tv", "radio"],
        )
        result = perturbation_test.run(adapter, sample_data)
        assert set(result.test_scores.percentage_change_for_each_channel.index) == {"tv", "radio"}
        assert (result.test_scores.percentage_change_for_each_channel == 0.0).all()

    def test_perturbation_test_with_reach_frequency_regressors(self, perturbation_test, sample_data):
        """Test perturbation with reach and frequency regressors."""
        adapter = MockAdapter(
            primary_media_regressor_type=PrimaryMediaRegressor.REACH_AND_FREQUENCY,
            primary_media_regressor_columns=[],
            channel_spend_columns=["tv_spend", "radio_spend"],
            media_channels=["tv", "radio"],
        )
        result = perturbation_test.run(adapter, sample_data)
        assert set(result.test_scores.percentage_change_for_each_channel.index) == {"tv", "radio"}
        assert result.test_scores.percentage_change_for_each_channel.isna().all()

    def test_perturbation_test_fallback_to_channel_spend_columns(self, perturbation_test, sample_data):
        """Test fallback to channel_spend_columns when media_channels is None."""
        adapter = MockAdapter(
            primary_media_regressor_type=PrimaryMediaRegressor.REACH_AND_FREQUENCY,
            primary_media_regressor_columns=[],
            channel_spend_columns=["tv_spend", "radio_spend"],
            media_channels=None,
        )
        result = perturbation_test.run(adapter, sample_data)
        assert set(result.test_scores.percentage_change_for_each_channel.index) == {"tv_spend", "radio_spend"}
        assert result.test_scores.percentage_change_for_each_channel.isna().all()

    def test_add_gaussian_noise_to_primary_regressors(self, perturbation_test, sample_data):
        """Test adding Gaussian noise to primary regressors."""
        original_data = sample_data.copy()
        regressor_cols = ["tv_spend", "radio_spend"]
        noisy_data = perturbation_test._add_gaussian_noise_to_primary_regressors(
            df=sample_data, regressor_cols=regressor_cols
        )
        pd.testing.assert_frame_equal(original_data, sample_data)
        for col in regressor_cols:
            assert not np.array_equal(noisy_data[col], sample_data[col])
        for col in sample_data.columns:
            if col not in regressor_cols:
                pd.testing.assert_series_equal(noisy_data[col], sample_data[col])

    def test_get_percent_gaussian_noise(self, perturbation_test, sample_data):
        """Test Gaussian noise generation."""
        noise = perturbation_test._get_percent_gaussian_noise(sample_data)
        assert len(noise) == len(sample_data)
        assert isinstance(noise, np.ndarray)
        assert np.all(np.abs(noise) < 0.3)

    def test_perturbation_test_adapter_not_fitted_error(self, perturbation_test, sample_data):
        """Test error when adapter is not fitted."""
        adapter = MockAdapter(
            primary_media_regressor_type=PrimaryMediaRegressor.SPEND,
            primary_media_regressor_columns=["tv_spend", "radio_spend"],
            channel_spend_columns=["tv_spend", "radio_spend"],
            media_channels=["tv", "radio"],
        )
        adapter.is_fitted = False
        adapter._force_not_fitted_error = True
        with pytest.raises(RuntimeError, match="Model must be fit before computing ROI"):
            perturbation_test.run(adapter, sample_data)

    def test_perturbation_test_with_single_channel(self, perturbation_test, sample_data):
        """Test perturbation with a single channel."""
        adapter = MockAdapter(
            primary_media_regressor_type=PrimaryMediaRegressor.SPEND,
            primary_media_regressor_columns=["tv_spend"],
            channel_spend_columns=["tv_spend"],
            media_channels=["tv"],
        )
        result = perturbation_test.run(adapter, sample_data)
        assert list(result.test_scores.percentage_change_for_each_channel.index) == ["tv"]
        assert (result.test_scores.percentage_change_for_each_channel == 0.0).all()

    def test_perturbation_test_with_many_channels(self, perturbation_test, sample_data):
        """Test perturbation with many channels."""
        channels = [f"channel_{i}" for i in range(10)]
        spend_cols = [f"spend_{i}" for i in range(10)]
        data = sample_data.copy()
        for col in spend_cols:
            data[col] = np.random.randn(len(data)) + 100
        adapter = MockAdapter(
            primary_media_regressor_type=PrimaryMediaRegressor.SPEND,
            primary_media_regressor_columns=spend_cols,
            channel_spend_columns=spend_cols,
            media_channels=channels,
        )
        result = perturbation_test.run(adapter, data)
        assert set(result.test_scores.percentage_change_for_each_channel.index) == set(channels)
        assert (result.test_scores.percentage_change_for_each_channel == 0.0).all()
