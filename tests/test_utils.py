"""Test utility functions."""

import json

import pytest
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior

from mmm_eval.utils import PyMCConfigRehydrator



def valid_hydration_config_1():
    """Create a valid hydration configuration for testing."""
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
        "fit_kwargs": {
            "target_accept": 0.9,
            "draws": 50,
            "tune": 50,
            "chains": 1,
            "random_seed": 123,
        },
    }


def valid_hydration_config_2():
    """Create another valid hydration configuration for testing."""
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
    }


def invalid_hydration_config():
    """Create an invalid hydration configuration for testing."""
    return {
        "date_column": "date_week",
        "channel_columns": ["channel_1", "channel_2"],
        "response_column": "quantity",
        "control_columns": ["price", "event_1", "event_2"],
        "revenue_column": "revenue",
        "adstock": "GeometricAdstock(l_max=4)",  # String instead of object
        "saturation": "LogisticSaturation()",  # String instead of object
        "yearly_seasonality": 2,
        "fit_kwargs": {"target_accept": 0.9},
    }


@pytest.mark.parametrize(
    "config, equal_to_original",
    [
        (valid_hydration_config_1(), True),
        (valid_hydration_config_2(), True),
        (invalid_hydration_config(), False),
    ],
)
def test_config_rehydration(config, equal_to_original, tmp_path):
    """Test configuration rehydration functionality."""
    # Save config to temporary file
    config_path = tmp_path / "pymc_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, default=str, indent=2)

    # Load config back in
    with open(config_path) as f:
        loaded_config = json.load(f)

    # Create rehydrator with loaded config
    rehydrated_config = PyMCConfigRehydrator(loaded_config).rehydrate_config()

    # Compare rehydrated configs
    if equal_to_original:
        assert rehydrated_config == config
    else:
        assert rehydrated_config != config
