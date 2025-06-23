import os
import tempfile

import pytest
from pydantic import ValidationError
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation

from mmm_eval.adapters.experimental.schemas import PyMCModelSchema
from mmm_eval.configs.configs import Config, EvalConfig, PyMCConfig


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


def test_config_validation():
    """Test Config class validation."""
    valid_config = {
        "date_column": "date_week",
        "channel_columns": ["channel_1", "channel_2"],
        "adstock": GeometricAdstock(l_max=4),
        "saturation": LogisticSaturation(),
        "yearly_seasonality": 2,
    }

    config = Config(valid_config, PyMCModelSchema)
    assert config.config is not None
    assert "date_column" in config.config


def test_config_validation_invalid():
    """Test Config class validation with invalid config."""
    invalid_config = {"channel_columns": []}  # Empty list should raise ValueError

    with pytest.raises(ValidationError):
        Config(invalid_config, PyMCModelSchema)


def test_eval_config():
    """Test EvalConfig base class."""
    mock_model = MockModelObject()
    eval_config = EvalConfig(mock_model)
    assert eval_config.model_object == mock_model


def test_pymc_config_with_model_object():
    """Test PyMCConfig initialization with model object."""
    mock_model = MockModelObject()
    fit_kwargs = {"target_accept": 0.9}
    target_column = "quantity"

    config = PyMCConfig(mock_model, fit_kwargs, target_column)

    assert config.model_config is not None
    assert config.fit_config is not None
    assert config.target_column == target_column

    # Check that extra fields are filtered out
    assert "extra_field" not in config.model_config.config
    assert "date_column" in config.model_config.config


def test_pymc_config_save_and_load():
    """Test PyMCConfig save and load functionality."""
    mock_model = MockModelObject()
    fit_kwargs = {"target_accept": 0.9}
    target_column = "quantity"

    config = PyMCConfig(mock_model, fit_kwargs, target_column)

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = temp_dir
        file_name = "test_config"

        # Save config
        config.save_config(save_path, file_name)

        # Check file exists
        file_path = os.path.join(save_path, file_name + ".json")
        assert os.path.exists(file_path)

        # Load config
        loaded_config = PyMCConfig.load_config(file_path)

        # Verify loaded config has same structure
        assert loaded_config.model_config is not None
        assert loaded_config.fit_config is not None
        assert loaded_config.target_column == target_column


def test_pymc_config_filter_input_to_schema():
    """Test filtering of input config to schema keys."""
    mock_model = MockModelObject()
    config = PyMCConfig(mock_model)

    # Test that extra fields are filtered out
    filtered_config = config._filter_input_to_schema(mock_model.__dict__, PyMCModelSchema)

    assert "extra_field" not in filtered_config
    assert "date_column" in filtered_config
    assert "channel_columns" in filtered_config


def test_pymc_config_partial_initialization():
    """Test PyMCConfig with partial initialization."""
    # Test with only model_object
    mock_model = MockModelObject()
    config = PyMCConfig(model_object=mock_model)
    assert config.model_config is not None
    assert not hasattr(config, "fit_config")
    assert not hasattr(config, "target_column")

    # Test with only fit_kwargs
    fit_kwargs = {"target_accept": 0.9}
    config = PyMCConfig(fit_kwargs=fit_kwargs)
    assert config.fit_config is not None
    assert not hasattr(config, "model_config")
    assert not hasattr(config, "target_column")

    # Test with only target_column
    config = PyMCConfig(target_column="quantity")
    assert config.target_column == "quantity"
    assert not hasattr(config, "model_config")
    assert not hasattr(config, "fit_config")
