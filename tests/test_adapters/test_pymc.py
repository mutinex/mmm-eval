"""Test the PyMC adapter."""

import pandas as pd
import pytest
from unittest.mock import Mock
import numpy as np

# TODO: update this import once PyMCAdapter is promoted out of experimental
from mmm_eval.adapters.experimental.pymc import (
    PyMCAdapter,
    _validate_start_end_dates,
    _check_columns_in_data,
)
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior


def valid_pymc_config_1():
    return {
        "date_column": "date_week",
        "channel_columns": ["channel_1", "channel_2"],
        "response_column": "quantity",
        "control_columns": ["price", "event_1", "event_2"],
        "revenue_column": "revenue",
        "adstock": GeometricAdstock(l_max=4),
        "saturation": LogisticSaturation(),
        "yearly_seasonality": 2,
        "model_config": {
            "intercept": Prior("Normal", mu=0.5, sigma=0.2),
            "saturation_beta": Prior("HalfNormal", sigma=[0.321, 0.123]),
            "gamma_control": Prior("Normal", mu=0, sigma=0.05),
            "gamma_fourier": Prior("Laplace", mu=0, b=0.2),
        },
        "fit_kwargs": {"target_accept": 0.9},
        "sampler_config": {
            "chains": 1,
            "draws": 10,
            "tune": 5,
            "random_seed": 42,
        },
    }


def valid_pymc_config_2():
    model_config = {
        "intercept": Prior("Normal", mu=0.5, sigma=0.2),
        "saturation_beta": Prior("HalfNormal", sigma=[0.321, 0.123]),
        "gamma_control": Prior("Normal", mu=0, sigma=0.05),
        "gamma_fourier": Prior("Laplace", mu=0, b=0.2),
    }
    return {
        "date_column": "date_week",
        "channel_columns": ["channel_1", "channel_2"],
        "response_column": "quantity",
        "control_columns": ["price", "event_1", "event_2"],
        "revenue_column": "revenue",
        "adstock": GeometricAdstock(l_max=4),
        "saturation": LogisticSaturation(),
        "yearly_seasonality": 2,
        "model_config": model_config,
        "fit_kwargs": {"target_accept": 0.9},
        "sampler_config": {
            "chains": 1,
            "draws": 10,
            "tune": 5,
            "random_seed": 42,
        },
    }


def invalid_pymc_config():
    return {
        "date_column": "date_week",
        "channel_columns": ["channel_1", "channel_2"],
        "response_column": "quantity",
        "control_columns": ["price", "event_1", "event_2"],
        "adstock": "NotARealAdstock",
    }


def create_sample_data():
    """Create sample data for testing."""
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


def create_realistic_test_data():
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
        base_response
        + trend
        + seasonality
        + 0.3 * channel_1
        + 0.2 * channel_2
        + np.random.normal(0, 30, len(dates))
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


@pytest.mark.parametrize(
    "config, is_valid",
    [
        (valid_pymc_config_1(), True),
        (valid_pymc_config_2(), True),
        (invalid_pymc_config(), False),
    ],
)
def test_adapter_instantiation(config, is_valid):
    """Test adapter instantiation with valid and invalid configs."""
    if not is_valid:
        with pytest.raises(ValueError):
            PyMCAdapter(config)
    else:
        adapter = PyMCAdapter(config)
        assert adapter is not None
        assert adapter.config is not None
        assert adapter.date_col == "date_week"
        assert adapter.channel_spend_cols == ["channel_1", "channel_2"]
        assert adapter.response_col == "quantity"
        assert adapter.revenue_col == "revenue"
        assert adapter.fit_kwargs == {"target_accept": 0.9}


def test_adapter_instantiation_config_copy():
    """Test that adapter doesn't mutate the original config."""
    config = valid_pymc_config_1()
    original_config = config.copy()

    _ = PyMCAdapter(config)

    # Check that original config is unchanged (the adapter modifies its copy)
    assert config == original_config


@pytest.mark.integration
def test_fit_method_real_pymc():
    """Test the fit method with real PyMC (integration test)."""
    config = valid_pymc_config_1()
    adapter = PyMCAdapter(config)
    data = create_realistic_test_data()

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
def test_predict_method_real_pymc():
    """Test the predict method with real PyMC (integration test)."""
    config = valid_pymc_config_1()
    adapter = PyMCAdapter(config)
    data = create_realistic_test_data()

    # Fit the model first
    adapter.fit(data)

    # Test prediction
    result = adapter.predict(data)

    # Verify prediction results
    assert isinstance(result, np.ndarray)
    assert len(result) == len(data)
    assert not np.all(np.isnan(result))  # Should have some non-NaN predictions


@pytest.mark.integration
def test_get_channel_roi_real_pymc():
    """Test the get_channel_roi method with real PyMC (integration test)."""
    config = valid_pymc_config_1()
    adapter = PyMCAdapter(config)
    data = create_realistic_test_data()

    # Fit the model first
    adapter.fit(data)

    # Test ROI calculation
    result = adapter.get_channel_roi()

    # Verify ROI results
    assert isinstance(result, pd.Series)
    assert len(result) == len(config["channel_columns"])
    assert all(channel in result.index for channel in config["channel_columns"])
    assert not np.all(np.isnan(result))  # Should have some non-NaN ROIs


@pytest.mark.integration
def test_get_channel_roi_with_date_range_real_pymc():
    """Test get_channel_roi with date range using real PyMC (integration test)."""
    config = valid_pymc_config_1()
    adapter = PyMCAdapter(config)
    data = create_realistic_test_data()

    # Fit the model first
    adapter.fit(data)

    # Test ROI calculation with date range
    start_date = data["date_week"].min() + pd.Timedelta(days=7)
    end_date = data["date_week"].max() - pd.Timedelta(days=7)

    result = adapter.get_channel_roi(start_date=start_date, end_date=end_date)

    # Verify ROI results
    assert isinstance(result, pd.Series)
    assert len(result) == len(config["channel_columns"])
    assert all(channel in result.index for channel in config["channel_columns"])


@pytest.mark.integration
def test_adapter_integration_real_pymc():
    """Test full integration workflow with real PyMC."""
    config = valid_pymc_config_1()
    adapter = PyMCAdapter(config)
    data = create_realistic_test_data()

    # Test full workflow: fit → predict → get_roi
    adapter.fit(data)
    assert adapter.is_fitted is True

    predictions = adapter.predict(data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(data)

    rois = adapter.get_channel_roi()
    assert isinstance(rois, pd.Series)
    assert len(rois) == len(config["channel_columns"])

    # Test that we can get ROIs for different date ranges
    mid_point = len(data) // 2
    start_date = data["date_week"].iloc[0]
    end_date = data["date_week"].iloc[mid_point]

    rois_first_half = adapter.get_channel_roi(start_date=start_date, end_date=end_date)
    assert isinstance(rois_first_half, pd.Series)
    assert len(rois_first_half) == len(config["channel_columns"])


def test_fit_method_missing_columns():
    """Test fit method with missing columns."""
    config = valid_pymc_config_1()
    adapter = PyMCAdapter(config)
    data = create_sample_data().drop(columns=["channel_1"])

    with pytest.raises(ValueError, match="Not all column\\(s\\) in"):
        adapter.fit(data)


def test_predict_method_not_fitted():
    """Test predict method when model is not fitted."""
    config = valid_pymc_config_1()
    adapter = PyMCAdapter(config)
    data = create_sample_data()

    with pytest.raises(RuntimeError, match="Model must be fit before prediction"):
        adapter.predict(data)


def test_predict_method_missing_columns():
    """Test predict method with missing columns."""
    config = valid_pymc_config_1()
    adapter = PyMCAdapter(config)
    data = create_sample_data().drop(columns=["channel_1"])

    # Mock the adapter as fitted to test the column validation
    adapter.is_fitted = True
    adapter.model = Mock()

    with pytest.raises(ValueError, match="Not all column\\(s\\) in"):
        adapter.predict(data)


def test_get_channel_roi_method_not_fitted():
    """Test get_channel_roi method when model is not fitted."""
    config = valid_pymc_config_1()
    adapter = PyMCAdapter(config)

    with pytest.raises(RuntimeError, match="Model must be fit before computing ROI"):
        adapter.get_channel_roi()


def test_get_channel_roi_invalid_date_range():
    """Test get_channel_roi method with invalid date range."""
    config = valid_pymc_config_1()
    adapter = PyMCAdapter(config)

    # Mock the adapter as fitted
    adapter.is_fitted = True
    adapter._channel_roi_df = pd.DataFrame(
        index=pd.date_range("2023-01-01", periods=10)
    )

    start_date = pd.Timestamp("2023-01-15")
    end_date = pd.Timestamp("2023-01-01")  # End before start

    with pytest.raises(ValueError, match="Start date must be before end date"):
        adapter.get_channel_roi(start_date=start_date, end_date=end_date)


def test_calculate_rois():
    """Test the _calculate_rois method."""
    config = valid_pymc_config_1()
    adapter = PyMCAdapter(config)

    # Create sample contribution data
    dates = pd.date_range("2023-01-01", periods=5, freq="W-MON")
    contribution_df = pd.DataFrame(
        {
            "channel_1_units": [10, 15, 20, 25, 30],
            "channel_2_units": [5, 10, 15, 20, 25],
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
