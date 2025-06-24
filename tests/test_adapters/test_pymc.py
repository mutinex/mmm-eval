"""Test the PyMC adapter."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior

# TODO: update this import once PyMCAdapter is promoted out of experimental
from mmm_eval.adapters.experimental.pymc import (
    PyMCAdapter,
    _check_columns_in_data,
    _validate_start_end_dates,
)
from mmm_eval.configs import PyMCConfig


@pytest.fixture(scope="function")
def valid_pymc_config():
    """Create a valid PyMC configuration for testing.

    Returns
        PyMCConfig: A valid PyMC configuration object with all required fields.

    """
    model_config = {
        "intercept": Prior("Normal", mu=0.5, sigma=0.2),
        "saturation_beta": Prior("HalfNormal", sigma=[0.321, 0.123]),
        "gamma_control": Prior("Normal", mu=0, sigma=0.05),
        "gamma_fourier": Prior("Laplace", mu=0, b=0.2),
    }
    config_dict = {
        "model_config": {
            "date_column": "date_week",
            "channel_columns": ["channel_1", "channel_2"],
            "control_columns": ["price", "event_1", "event_2"],
            "adstock": GeometricAdstock(l_max=4),
            "saturation": LogisticSaturation(),
            "yearly_seasonality": 2,
            "model_config": model_config,
            "sampler_config": {
                "chains": 1,
                "draws": 10,
                "tune": 5,
                "random_seed": 42,
            },
        },
        "fit_config": {
            "target_accept": 0.9,
        },
        "response_column": "quantity",
        "revenue_column": "revenue",
    }
    return PyMCConfig.from_dict(config_dict)


@pytest.fixture(scope="module")
def invalid_pymc_config():
    """Create an invalid PyMC configuration for testing error handling.

    Returns
        PyMCConfig: A PyMC configuration object that will fail during model creation.

    """
    # Create a config that will pass initial validation but fail when MMM model tries to use it
    config_dict = {
        "model_config": {
            "date_column": "date_week",
            "channel_columns": ["channel_1", "channel_2"],
            "control_columns": ["price", "event_1", "event_2"],
            "adstock": GeometricAdstock(l_max=4),
            "saturation": LogisticSaturation(),
            # Add an invalid field that will cause the MMM model to fail
            "invalid_field": "this_will_cause_an_error",
        },
        "fit_config": {},
        "response_column": "quantity",
        "revenue_column": "revenue",
    }
    return PyMCConfig.from_dict(config_dict)


def create_sample_data():
    """Create sample data for testing.

    Returns
        pd.DataFrame: A DataFrame with sample data for testing.

    """
    dates = pd.date_range("2023-01-01", periods=10, freq="W-MON")
    return pd.DataFrame(
        {
            "date_week": dates,
            "channel_1": np.random.uniform(50, 200, len(dates)),
            "channel_2": np.random.uniform(30, 150, len(dates)),
            "quantity": np.random.uniform(800, 1200, len(dates)),
            "price": np.random.uniform(8, 12, len(dates)),
            "revenue": np.random.uniform(6000, 12000, len(dates)),
            "event_1": np.random.choice([0, 1], len(dates)),
            "event_2": np.random.choice([0, 1], len(dates)),
        }
    )


@pytest.fixture(scope="function")
def realistic_test_data():
    """Create more realistic test data for PyMC integration tests."""
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range("2023-01-01", periods=20, freq="W-MON")

    # Create correlated data
    channel_1 = np.random.uniform(50, 200, len(dates))
    channel_2 = np.random.uniform(30, 150, len(dates))

    # Create response with some correlation to channels
    base_response = 1000
    trend = np.linspace(0, 50, len(dates))
    seasonality = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)

    quantity = (
        base_response + trend + seasonality + 0.3 * channel_1 + 0.2 * channel_2 + np.random.normal(0, 30, len(dates))
    )

    price = 10 + 0.1 * np.arange(len(dates)) + np.random.normal(0, 0.5, len(dates))
    revenue = price * quantity

    return pd.DataFrame(
        {
            "date_week": dates,
            "channel_1": channel_1,
            "channel_2": channel_2,
            "quantity": quantity,
            "price": price,
            "revenue": revenue,
            "event_1": np.random.choice([0, 1], len(dates), p=[0.9, 0.1]),
            "event_2": np.random.choice([0, 1], len(dates), p=[0.95, 0.05]),
        }
    )


def test_adapter_instantiation(valid_pymc_config):
    """Test adapter instantiation with valid config.

    Args:
        valid_pymc_config: A valid PyMC configuration fixture.

    """
    adapter = PyMCAdapter(valid_pymc_config)
    assert adapter is not None
    assert adapter.config is not None
    assert adapter.date_column == "date_week"
    assert adapter.channel_spend_columns == ["channel_1", "channel_2"]
    assert adapter.response_column == "quantity"
    assert adapter.revenue_column == "revenue"
    assert adapter.fit_config == {"target_accept": 0.9}


def test_adapter_instantiation_config_copy(valid_pymc_config):
    """Test that adapter doesn't mutate the original config.

    Args:
        valid_pymc_config: A valid PyMC configuration fixture.

    """
    config = valid_pymc_config
    # Store original attribute values since PyMCConfig objects don't have .copy()
    original_model_config = config.model_config.config.copy() if config.model_config else None
    original_fit_config = config.fit_config.config.copy() if config.fit_config else None
    original_response_column = config.response_column
    original_revenue_column = config.revenue_column

    _ = PyMCAdapter(config)

    # Check that original config is unchanged (the adapter modifies its copy)
    if config.model_config:
        assert config.model_config.config == original_model_config
    if config.fit_config:
        assert config.fit_config.config == original_fit_config
    assert config.response_column == original_response_column
    assert config.revenue_column == original_revenue_column


@pytest.mark.integration
def test_fit_method_real_pymc(valid_pymc_config, realistic_test_data):
    """Test the fit method with real PyMC (integration test)."""
    config = valid_pymc_config
    adapter = PyMCAdapter(config)
    data = realistic_test_data

    # This should actually fit a PyMC model
    adapter.fit(data)

    # Verify the model was fitted
    assert adapter.is_fitted is True
    assert adapter.model is not None
    assert adapter.trace is not None
    assert adapter._channel_roi_df is not None

    # Verify the trace has the expected structure
    assert hasattr(adapter.trace, "posterior")
    assert "chain" in adapter.trace.posterior.dims
    assert "draw" in adapter.trace.posterior.dims


@pytest.mark.integration
def test_predict_method_real_pymc(valid_pymc_config, realistic_test_data):
    """Test the predict method with real PyMC (integration test)."""
    config = valid_pymc_config
    adapter = PyMCAdapter(config)
    data = realistic_test_data

    # Fit the model first
    adapter.fit(data)

    # Test prediction
    result = adapter.predict(data)

    # Verify prediction results
    assert isinstance(result, np.ndarray)
    assert len(result) == len(data)
    assert not np.all(np.isnan(result))  # Should have some non-NaN predictions


@pytest.mark.integration
def test_get_channel_roi_real_pymc(valid_pymc_config, realistic_test_data):
    """Test the get_channel_roi method with real PyMC (integration test)."""
    config = valid_pymc_config
    adapter = PyMCAdapter(config)
    data = realistic_test_data

    # Fit the model first
    adapter.fit(data)

    # Test ROI calculation
    result = adapter.get_channel_roi()

    # Verify ROI results
    assert isinstance(result, pd.Series)
    assert len(result) == len(adapter.channel_spend_columns)
    assert all(channel in result.index for channel in adapter.channel_spend_columns)
    assert not np.all(np.isnan(result))  # Should have some non-NaN ROIs


@pytest.mark.integration
def test_get_channel_roi_with_date_range_real_pymc(valid_pymc_config, realistic_test_data):
    """Test get_channel_roi with date range using real PyMC (integration test)."""
    config = valid_pymc_config
    adapter = PyMCAdapter(config)
    data = realistic_test_data

    # Fit the model first
    adapter.fit(data)

    # Test ROI calculation with date range
    start_date = data["date_week"].min() + pd.Timedelta(days=7)
    end_date = data["date_week"].max() - pd.Timedelta(days=7)

    result = adapter.get_channel_roi(start_date=start_date, end_date=end_date)

    # Verify ROI results
    assert isinstance(result, pd.Series)
    assert len(result) == len(adapter.channel_spend_columns)
    assert all(channel in result.index for channel in adapter.channel_spend_columns)


@pytest.mark.integration
def test_adapter_integration_real_pymc(valid_pymc_config, realistic_test_data):
    """Test full integration workflow with real PyMC."""
    config = valid_pymc_config
    adapter = PyMCAdapter(config)
    data = realistic_test_data

    # Test full workflow: fit → predict → get_roi
    adapter.fit(data)
    assert adapter.is_fitted is True

    predictions = adapter.predict(data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(data)

    rois = adapter.get_channel_roi()
    assert isinstance(rois, pd.Series)
    assert len(rois) == len(adapter.channel_spend_columns)

    # Test that we can get ROIs for different date ranges
    mid_point = len(data) // 2
    start_date = data["date_week"].iloc[0]
    end_date = data["date_week"].iloc[mid_point]

    rois_first_half = adapter.get_channel_roi(start_date=start_date, end_date=end_date)
    assert isinstance(rois_first_half, pd.Series)
    assert len(rois_first_half) == len(adapter.channel_spend_columns)


def test_fit_method_missing_columns(valid_pymc_config):
    """Test fit method with missing columns.

    Args:
        valid_pymc_config: A valid PyMC configuration fixture.

    """
    config = valid_pymc_config
    adapter = PyMCAdapter(config)
    data = create_sample_data().drop(columns=["channel_1"])

    with pytest.raises(ValueError, match="Not all column\\(s\\) in"):
        adapter.fit(data)


def test_predict_method_not_fitted(valid_pymc_config):
    """Test predict method when model is not fitted.

    Args:
        valid_pymc_config: A valid PyMC configuration fixture.

    """
    config = valid_pymc_config
    adapter = PyMCAdapter(config)
    data = create_sample_data()

    with pytest.raises(RuntimeError, match="Model must be fit before prediction"):
        adapter.predict(data)


def test_predict_method_missing_columns(valid_pymc_config):
    """Test predict method with missing columns.

    Args:
        valid_pymc_config: A valid PyMC configuration fixture.

    """
    config = valid_pymc_config
    adapter = PyMCAdapter(config)
    data = create_sample_data().drop(columns=["channel_1"])

    # Mock the adapter as fitted to test the column validation
    adapter.is_fitted = True
    adapter.model = Mock()

    with pytest.raises(ValueError, match="Not all column\\(s\\) in"):
        adapter.predict(data)


def test_get_channel_roi_method_not_fitted(valid_pymc_config):
    """Test get_channel_roi method when model is not fitted.

    Args:
        valid_pymc_config: A valid PyMC configuration fixture.

    """
    config = valid_pymc_config
    adapter = PyMCAdapter(config)

    with pytest.raises(RuntimeError, match="Model must be fit before computing ROI"):
        adapter.get_channel_roi()


def test_get_channel_roi_invalid_date_range(valid_pymc_config):
    """Test get_channel_roi method with invalid date range.

    Args:
        valid_pymc_config: A valid PyMC configuration fixture.

    """
    config = valid_pymc_config
    adapter = PyMCAdapter(config)

    # Mock the adapter as fitted
    adapter.is_fitted = True
    adapter._channel_roi_df = pd.DataFrame(index=pd.date_range("2023-01-01", periods=10))

    start_date = pd.Timestamp("2023-01-15")
    end_date = pd.Timestamp("2023-01-01")  # End before start

    with pytest.raises(ValueError, match="Start date must be before end date"):
        adapter.get_channel_roi(start_date=start_date, end_date=end_date)


def test_calculate_rois(valid_pymc_config):
    """Test the _calculate_rois method.

    Args:
        valid_pymc_config: A valid PyMC configuration fixture.

    """
    config = valid_pymc_config
    adapter = PyMCAdapter(config)

    # Create sample contribution data
    dates = pd.date_range("2023-01-01", periods=5, freq="W-MON")
    contribution_df = pd.DataFrame(
        {
            "channel_1_response_units": [10, 15, 20, 25, 30],
            "channel_2_response_units": [5, 10, 15, 20, 25],
            "channel_1": [100, 150, 200, 250, 300],
            "channel_2": [50, 100, 150, 200, 250],
            "quantity": [1000, 1100, 1200, 1300, 1400],
            "revenue": [10000, 11000, 12000, 13000, 14000],
        },
        index=dates,
    )

    result = adapter._calculate_rois(contribution_df)

    assert isinstance(result, dict)
    assert "channel_1" in result
    assert "channel_2" in result
    assert all(isinstance(v, float) for v in result.values())


def test_calculate_rois_zero_total_spend(valid_pymc_config):
    """Test _calculate_rois method when total spend for a channel is zero.

    Args:
        valid_pymc_config: A valid PyMC configuration fixture.

    """
    config = valid_pymc_config
    adapter = PyMCAdapter(config)

    # Create sample contribution data with zero spend for channel_2
    dates = pd.date_range("2023-01-01", periods=5, freq="W-MON")
    contribution_df = pd.DataFrame(
        {
            "channel_1_response_units": [10, 15, 20, 25, 30],
            "channel_2_response_units": [5, 10, 15, 20, 25],  # Non-zero attribution
            "channel_1": [100, 150, 200, 250, 300],
            "channel_2": [0, 0, 0, 0, 0],  # Zero spend
            "quantity": [1000, 1100, 1200, 1300, 1400],
            "revenue": [10000, 11000, 12000, 13000, 14000],
        },
        index=dates,
    )

    result = adapter._calculate_rois(contribution_df)

    assert isinstance(result, dict)
    assert "channel_1" in result
    assert "channel_2" in result
    assert isinstance(result["channel_1"], float)
    assert np.isnan(result["channel_2"])  # Should be NaN for zero total spend


def test_fit_drops_zero_spend_channels(valid_pymc_config, realistic_test_data):
    """Test that channels with zero spend are dropped during fit.

    Args:
        valid_pymc_config: A valid PyMC configuration fixture.
        realistic_test_data: A realistic test data fixture.

    """
    config = valid_pymc_config
    adapter = PyMCAdapter(config)

    # Create data with one channel having zero spend
    data = realistic_test_data.copy()
    data["channel_1"] = 0  # Set channel_1 to zero spend

    # Fit the model
    adapter.fit(data)

    # Verify that channel_1 was dropped from channel_spend_columns
    assert "channel_1" not in adapter.channel_spend_columns
    assert "channel_2" in adapter.channel_spend_columns
    assert len(adapter.channel_spend_columns) == 1

    # Verify that the model config was updated
    assert adapter.model_config["channel_columns"] == ["channel_2"]


def test_fit_keeps_non_zero_spend_channels(valid_pymc_config, realistic_test_data):
    """Test that channels with non-zero spend are kept during fit.

    Args:
        valid_pymc_config: A valid PyMC configuration fixture.
        realistic_test_data: A realistic test data fixture.

    """
    config = valid_pymc_config
    adapter = PyMCAdapter(config)

    # Create data with all channels having non-zero spend
    data = realistic_test_data.copy()
    # Ensure both channels have non-zero spend for all periods
    data["channel_1"] = np.random.uniform(100, 300, len(data))  # Non-zero spend
    data["channel_2"] = np.random.uniform(50, 250, len(data))  # Non-zero spend

    # Verify the data has non-zero spend before fitting
    assert data["channel_1"].sum() > 0
    assert data["channel_2"].sum() > 0

    # Fit the model
    adapter.fit(data)

    # Verify that all channels are still present
    assert "channel_1" in adapter.channel_spend_columns
    assert "channel_2" in adapter.channel_spend_columns
    assert len(adapter.channel_spend_columns) == 2

    # Verify that the model config was not changed
    assert adapter.model_config["channel_columns"] == ["channel_1", "channel_2"]


def test_fit_drops_multiple_zero_spend_channels(valid_pymc_config, realistic_test_data):
    """Test that multiple channels with zero spend are dropped during fit.

    Args:
        valid_pymc_config: A valid PyMC configuration fixture.
        realistic_test_data: A realistic test data fixture.

    """
    config = valid_pymc_config
    adapter = PyMCAdapter(config)

    # Create data with both channels having zero spend
    data = realistic_test_data.copy()
    data["channel_1"] = 0  # Set channel_1 to zero spend
    data["channel_2"] = 0  # Set channel_2 to zero spend

    # Fit the model - this should raise an error since no channels remain
    with pytest.raises((ValueError, RuntimeError)):
        adapter.fit(data)


def test_validate_start_end_dates_valid():
    """Test _validate_start_end_dates with valid dates."""
    date_range = pd.date_range("2023-01-01", periods=10, freq="D")
    start_date = pd.Timestamp("2023-01-02")
    end_date = pd.Timestamp("2023-01-05")

    # Should not raise any exception
    _validate_start_end_dates(start_date, end_date, date_range)


def test_validate_start_end_dates_invalid_order():
    """Test _validate_start_end_dates with invalid date order."""
    date_range = pd.date_range("2023-01-01", periods=10, freq="D")
    start_date = pd.Timestamp("2023-01-05")
    end_date = pd.Timestamp("2023-01-02")

    with pytest.raises(ValueError, match="Start date must be before end date"):
        _validate_start_end_dates(start_date, end_date, date_range)


def test_validate_start_end_dates_same_date():
    """Test _validate_start_end_dates with same start and end date."""
    date_range = pd.date_range("2023-01-01", periods=10, freq="D")
    start_date = pd.Timestamp("2023-01-05")
    end_date = pd.Timestamp("2023-01-05")

    with pytest.raises(ValueError, match="Start date must be before end date"):
        _validate_start_end_dates(start_date, end_date, date_range)


def test_validate_start_end_dates_none_values():
    """Test _validate_start_end_dates with None values."""
    date_range = pd.date_range("2023-01-01", periods=10, freq="D")

    # Should not raise any exception
    _validate_start_end_dates(None, None, date_range)
    _validate_start_end_dates(pd.Timestamp("2023-01-02"), None, date_range)
    _validate_start_end_dates(None, pd.Timestamp("2023-01-05"), date_range)


def test_validate_start_end_dates_outside_range():
    """Test _validate_start_end_dates with dates outside the data range."""
    date_range = pd.date_range("2023-01-01", periods=10, freq="D")
    start_date = pd.Timestamp("2022-12-01")  # Before data range
    end_date = pd.Timestamp("2023-02-01")  # After data range

    # Should log info but not raise exception
    _validate_start_end_dates(start_date, end_date, date_range)


def test_check_columns_in_data_valid():
    """Test _check_columns_in_data with valid columns."""
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})

    # Should not raise any exception
    _check_columns_in_data(data, ["col1", "col2"])
    _check_columns_in_data(data, ["col1"])
    _check_columns_in_data(data, ["col1", "col2", "col3"])


def test_check_columns_in_data_missing_columns():
    """Test _check_columns_in_data with missing columns."""
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    with pytest.raises(ValueError, match="Not all column\\(s\\) in"):
        _check_columns_in_data(data, ["col1", "col3"])


def test_check_columns_in_data_empty_dataframe():
    """Test _check_columns_in_data with empty DataFrame."""
    data = pd.DataFrame()

    with pytest.raises(ValueError, match="Not all column\\(s\\) in"):
        _check_columns_in_data(data, ["col1"])


def test_check_columns_in_data_mixed_column_sets():
    """Test _check_columns_in_data with mixed column sets."""
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})

    # Test with string and list mixed
    _check_columns_in_data(data, ["col1", ["col2", "col3"]])

    # Test with missing columns in one set
    with pytest.raises(ValueError, match="Not all column\\(s\\) in"):
        _check_columns_in_data(data, ["col1", ["col2", "col4"]])
