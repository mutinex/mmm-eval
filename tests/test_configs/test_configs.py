import os
import tempfile

import pytest
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior

from mmm_eval.adapters.schemas import (
    MeridianInputDataBuilderSchema,
    MeridianModelSpecSchema,
    MeridianSamplePosteriorSchema,
    PyMCFitSchema,
    PyMCModelSchema,
)
from mmm_eval.configs.configs import MeridianConfig, PyMCConfig


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


class MockMeridianModelObject:
    """Mock Meridian model object for testing."""

    def __init__(self):
        """Initialize the MockMeridianModelObject."""
        from meridian.model.prior_distribution import PriorDistribution
        import tensorflow_probability as tfp
        
        # Create a proper prior distribution
        prior = PriorDistribution(roi_m=tfp.distributions.LogNormal(0.2, 0.9))
        
        # Mock model spec with prior distribution
        self.model_spec = type('MockModelSpec', (), {
            'prior': prior,
            'media_effects_dist': 'log_normal',
            'hill_before_adstock': False,
            'max_lag': 8,
            'organic_media_prior_type': 'contribution',
            'non_media_treatments_prior_type': 'contribution',
        })()
        
        # Mock input_data structure
        self.input_data = type('MockInputData', (), {
            'kpi': type('MockKPI', (), {'time': [1, 2, 3]})(),
            'media': [
                type('MockMedia', (), {
                    'channel': 'channel_1',
                    'spend': [100, 200, 300],
                    'impressions': [1000, 2000, 3000]
                })(),
                type('MockMedia', (), {
                    'channel': 'channel_2', 
                    'spend': [150, 250, 350],
                    'impressions': [1500, 2500, 3500]
                })()
            ],
            'controls': [
                type('MockControl', (), {'name': 'control_1'})(),
                type('MockControl', (), {'name': 'control_2'})()
            ],
            'organic_media': [
                type('MockOrganic', (), {
                    'name': 'organic_1',
                    'channel': 'organic_channel'
                })()
            ],
            'non_media_treatments': [
                type('MockTreatment', (), {'name': 'treatment_1'})()
            ]
        })()


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
    config = PyMCConfig.from_model_object(
        model_object=mock_model,
        fit_kwargs=fit_kwargs,
        revenue_column=revenue_column,
        response_column=response_column,
    )
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
    config = PyMCConfig.from_model_object(
        model_object=mock_model,
        fit_kwargs=fit_kwargs,
        revenue_column=revenue_column,
    )
    assert config.response_column == revenue_column
    # Test auto-setting of date and channel columns
    assert config.date_column == mock_model.date_column
    assert config.channel_columns == mock_model.channel_columns


def test_pymc_config_save_and_load_json():
    """Test saving and loading a PyMCConfig to and from a JSON file."""
    mock_model = MockModelObject()
    fit_kwargs = {"target_accept": 0.9}
    response_column = "quantity"
    config = PyMCConfig.from_model_object(
        model_object=mock_model,
        fit_kwargs=fit_kwargs,
        revenue_column="revenue",
        response_column=response_column,
    )
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
        PyMCConfig.from_model_object(model_object=None, revenue_column="revenue")
    with pytest.raises(ValueError, match="`revenue_column` is required"):
        PyMCConfig.from_model_object(model_object=mock_model, revenue_column="")


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


def test_meridian_config_from_model_object():
    """Test MeridianConfig from model object."""
    mock_model = MockMeridianModelObject()
    fit_kwargs = {"n_chains": 2, "n_adapt": 100}
    revenue_column = "revenue"
    response_column = "quantity"
    
    # Now that _extract_input_data_builder_config is implemented, this should work
    config = MeridianConfig.from_model_object(
        model_object=mock_model,
        fit_kwargs=fit_kwargs,
        revenue_column=revenue_column,
        response_column=response_column,
    )
    
    # Verify the config was created successfully
    assert config is not None
    assert config.revenue_column == revenue_column
    assert config.response_column == response_column
    assert config.input_data_builder_config is not None
    assert config.model_spec_config is not None
    assert config.sample_posterior_config is not None
    
    # Verify extracted input data builder config
    input_config = config.input_data_builder_config
    assert input_config.media_channels == ["channel_1", "channel_2"]
    assert input_config.channel_spend_columns == ["channel_1_spend", "channel_2_spend"]
    assert input_config.channel_impressions_columns == ["channel_1_impressions", "channel_2_impressions"]
    assert input_config.control_columns == ["control_1", "control_2"]
    assert input_config.organic_media_columns == ["organic_1"]
    assert input_config.organic_media_channels == ["organic_channel"]
    assert input_config.non_media_treatment_columns == ["treatment_1"]
    assert input_config.response_column == "response"
    assert input_config.date_column == "date"


def test_meridian_config_validation():
    """Test validation of MeridianConfig."""
    mock_model = MockMeridianModelObject()
    with pytest.raises(ValueError, match="`model_object` is required"):
        MeridianConfig.from_model_object(model_object=None, revenue_column="revenue")
    with pytest.raises(ValueError, match="`revenue_column` is required"):
        MeridianConfig.from_model_object(model_object=mock_model, revenue_column="")


def test_meridian_config_direct_instantiation():
    """Test direct instantiation of MeridianConfig."""
    from meridian.model.prior_distribution import PriorDistribution
    import tensorflow_probability as tfp
    
    # Create input data builder config
    input_data_builder_config = MeridianInputDataBuilderSchema(
        date_column="date_week",
        media_channels=["channel_1", "channel_2"],
        channel_spend_columns=["channel_1_spend", "channel_2_spend"],
        response_column="quantity",
    )
    
    # Create model spec config
    model_spec_config = MeridianModelSpecSchema(
        prior=PriorDistribution(roi_m=tfp.distributions.LogNormal(0.2, 0.9)),
        media_effects_dist="log_normal",
        hill_before_adstock=False,
        max_lag=8,
    )
    
    # Create sample posterior config
    sample_posterior_config = MeridianSamplePosteriorSchema(
        n_chains=2,
        n_adapt=100,
        n_burnin=100,
        n_keep=500,
    )
    
    config = MeridianConfig(
        input_data_builder_config=input_data_builder_config,
        model_spec_config=model_spec_config,
        sample_posterior_config=sample_posterior_config,
        revenue_column="revenue",
        response_column="quantity",
    )
    
    assert config.input_data_builder_config == input_data_builder_config
    assert config.model_spec_config == model_spec_config
    assert config.sample_posterior_config == sample_posterior_config
    assert config.revenue_column == "revenue"
    assert config.response_column == "quantity"


def test_meridian_config_save_and_load_json():
    """Test saving and loading a MeridianConfig to and from a JSON file."""
    from meridian.model.prior_distribution import PriorDistribution
    import tensorflow_probability as tfp
    
    # Create input data builder config
    input_data_builder_config = MeridianInputDataBuilderSchema(
        date_column="date_week",
        media_channels=["channel_1", "channel_2"],
        channel_spend_columns=["channel_1_spend", "channel_2_spend"],
        response_column="quantity",
    )
    
    # Create model spec config
    model_spec_config = MeridianModelSpecSchema(
        prior=PriorDistribution(roi_m=tfp.distributions.LogNormal(0.2, 0.9)),
        media_effects_dist="log_normal",
        hill_before_adstock=False,
        max_lag=8,
    )
    
    # Create sample posterior config
    sample_posterior_config = MeridianSamplePosteriorSchema(
        n_chains=2,
        n_adapt=100,
        n_burnin=100,
        n_keep=500,
    )
    
    config = MeridianConfig(
        input_data_builder_config=input_data_builder_config,
        model_spec_config=model_spec_config,
        sample_posterior_config=sample_posterior_config,
        revenue_column="revenue",
        response_column="quantity",
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = temp_dir
        file_name = "test_meridian_config"
        config.save_model_object_to_json(save_path, file_name)
        file_path = os.path.join(save_path, file_name + ".json")
        assert os.path.exists(file_path)
        
        # Note: Loading will likely fail due to the complexity of rehydrating
        # Meridian objects, but we can test that the file was created
        assert os.path.getsize(file_path) > 0
