import os
import tempfile

import pytest
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior

from mmm_eval.adapters.experimental.schemas import PyMCFitSchema, PyMCModelSchema
from mmm_eval.configs.configs import PyMCConfig


class MockModelObject:
    """Mock model object for testing."""

    def __init__(self):
        """Initialize the MockModelObject."""
        self.date_column = "date_week"
        self.channel_columns = ["channel_1", "channel_2"]
        self.control_columns = ["price", "event_1", "event_2"]
        self.adstock = GeometricAdstock(l_max=4)
        self.saturation = LogisticSaturation()
        self.yearly_seasonality = 2
        self.extra_field = "should_be_filtered_out"


def valid_hydration_config_1():
    """Create a valid hydration configuration for testing."""
    return {
        "response_column": "quantity",
        "revenue_column": "revenue",
        "model_config": {
            "intercept": Prior("Normal", mu=0.5, sigma=0.2),
            "saturation_beta": Prior("HalfNormal", sigma=[0.321, 0.123]),
            "gamma_control": Prior("Normal", mu=0, sigma=0.05),
            "gamma_fourier": Prior("Laplace", mu=0, b=0.2),
            "adstock": GeometricAdstock(l_max=4),
            "saturation": LogisticSaturation(),
            "yearly_seasonality": 2,
            "control_columns": ["price", "event_1", "event_2"],
            "date_column": "date_week",
            "channel_columns": ["channel_1", "channel_2"],
        },
        "fit_config": {
            "target_accept": 0.9,
            "draws": 5,
            "tune": 10,
            "chains": 1,
            "random_seed": 123,
        },
    }


# JSON config for e2e testing
SAMPLE_CONFIG_JSON = {
    "pymc_model_config": {
        "date_column": "date_week",
        "channel_columns": ["channel_1", "channel_2"],
        "control_columns": ["price", "event_1", "event_2"],
        "adstock": "GeometricAdstock(l_max=4)",
        "saturation": "LogisticSaturation()",
        "yearly_seasonality": "2",
    },
    "fit_config": {
        "target_accept": "0.9",
        "draws": 5,
        "tune": 10,
        "chains": 1,
        "random_seed": 123,
    },
    "revenue_column": "revenue",
    "response_column": "quantity",
}


def test_pymc_config_from_model_object():
    """Test PyMCConfig from model object."""
    mock_model = MockModelObject()
    fit_kwargs = {"target_accept": 0.9}
    revenue_column = "revenue"
    response_column = "quantity"
    config = PyMCConfig.from_model_object(mock_model, fit_kwargs, revenue_column, response_column)
    assert config.pymc_model_config is not None
    assert config.fit_config is not None
    assert config.response_column == response_column
    assert config.revenue_column == revenue_column
    assert "extra_field" not in config.pymc_model_config.model_dump()
    assert "date_column" in config.pymc_model_config.model_dump()


def test_pymc_config_auto_response_column():
    """Test auto-setting of response column."""
    mock_model = MockModelObject()
    fit_kwargs = {"target_accept": 0.9}
    revenue_column = "revenue"
    config = PyMCConfig.from_model_object(mock_model, fit_kwargs, revenue_column)
    assert config.response_column == revenue_column
    # Test auto-setting of date and channel columns
    assert config.date_column == mock_model.date_column
    assert config.channel_columns == mock_model.channel_columns


def test_pymc_config_save_and_load_json():
    """Test saving and loading a PyMCConfig to and from a JSON file."""
    mock_model = MockModelObject()
    fit_kwargs = {"target_accept": 0.9}
    response_column = "quantity"
    config = PyMCConfig.from_model_object(mock_model, fit_kwargs, "revenue", response_column)
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = temp_dir
        file_name = "test_config"
        config.save_model_object_to_json(save_path, file_name)
        file_path = os.path.join(save_path, file_name + ".json")
        assert os.path.exists(file_path)
        loaded_config = PyMCConfig.load_model_config_from_json(file_path)
        assert loaded_config.pymc_model_config is not None
        assert loaded_config.fit_config is not None
        assert loaded_config.response_column == response_column
        # Verify auto-set fields are preserved
        assert loaded_config.date_column == mock_model.date_column
        assert loaded_config.channel_columns == mock_model.channel_columns


def test_pymc_config_validation():
    """Test validation of PyMCConfig."""
    mock_model = MockModelObject()
    with pytest.raises(ValueError, match="`model_object` is required"):
        PyMCConfig.from_model_object(None, {"target_accept": 0.9}, "revenue")
    with pytest.raises(ValueError, match="`fit_kwargs` is required"):
        PyMCConfig.from_model_object(mock_model, {}, "revenue")
    with pytest.raises(ValueError, match="`revenue_column` is required"):
        PyMCConfig.from_model_object(mock_model, {"target_accept": 0.9}, "")


def test_pymc_config_direct_instantiation():
    """Test direct instantiation of PyMCConfig."""
    pymc_model_config = PyMCModelSchema(
        date_column="date_week",
        channel_columns=["channel_1", "channel_2"],
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
    )
    fit_config = PyMCFitSchema(target_accept=0.9)
    config = PyMCConfig(
        pymc_model_config=pymc_model_config,
        fit_config=fit_config,
        revenue_column="revenue",
        response_column="quantity",
    )
    assert config.pymc_model_config == pymc_model_config
    assert config.fit_config == fit_config
    assert config.revenue_column == "revenue"
    assert config.response_column == "quantity"
