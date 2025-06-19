import pytest
from mmm_eval.adapters.pymc import PyMCAdapter
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
    }


def invalid_pymc_config():
    return {
        "date_column": "date_week",
        "channel_columns": ["channel_1", "channel_2"],
        "response_column": "quantity",
        "control_columns": ["price", "event_1", "event_2"],
        "adstock": "NotARealAdstock",
    }


@pytest.mark.parametrize(
    "config, is_valid",
    [
        (valid_pymc_config_1(), True),
        (valid_pymc_config_2(), True),
        (invalid_pymc_config(), False),
    ],
)
def test_adapter_instantiation(config, is_valid):
    # Instantiate the adapter

    if not is_valid:
        with pytest.raises(ValueError):
            PyMCAdapter(config)
    else:
        adapter = PyMCAdapter(config)
        assert adapter is not None
        assert adapter.config is not None
