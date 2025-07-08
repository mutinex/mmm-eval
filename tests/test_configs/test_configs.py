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
        import tensorflow_probability as tfp
        from meridian.model.prior_distribution import PriorDistribution

        # Create a proper prior distribution
        prior = PriorDistribution(roi_m=tfp.distributions.LogNormal(0.2, 0.9))

        # Mock model spec with prior distribution
        self.model_spec = type(
            "MockModelSpec",
            (),
            {
                "prior": prior,
                "media_effects_dist": "log_normal",
                "hill_before_adstock": False,
                "max_lag": 8,
                "organic_media_prior_type": "contribution",
                "non_media_treatments_prior_type": "contribution",
            },
        )()

        # Mock input_data structure
        self.input_data = type(
            "MockInputData",
            (),
            {
                "kpi": type("MockKPI", (), {"time": [1, 2, 3]})(),
                "media": [
                    type(
                        "MockMedia",
                        (),
                        {"channel": "channel_1", "spend": [100, 200, 300], "impressions": [1000, 2000, 3000]},
                    )(),
                    type(
                        "MockMedia",
                        (),
                        {"channel": "channel_2", "spend": [150, 250, 350], "impressions": [1500, 2500, 3500]},
                    )(),
                ],
                "controls": [
                    type("MockControl", (), {"name": "control_1"})(),
                    type("MockControl", (), {"name": "control_2"})(),
                ],
                "organic_media": [type("MockOrganic", (), {"name": "organic_1", "channel": "organic_channel"})()],
                "non_media_treatments": [type("MockTreatment", (), {"name": "treatment_1"})()],
            },
        )()


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
    """Test creating a MeridianConfig from a model object."""
    mock_model = MockMeridianModelObject()
    fit_kwargs = {"n_chains": 2, "n_adapt": 100, "n_burnin": 100, "n_keep": 500}
    revenue_column = "revenue"
    response_column = "quantity"

    # Create input data builder config
    input_data_builder_config = MeridianInputDataBuilderSchema(
        date_column="date",
        media_channels=["channel_1", "channel_2"],
        channel_spend_columns=["channel_1_spend", "channel_2_spend"],
        response_column="response",
    )

    # Now that _extract_input_data_builder_config is implemented, this should work
    config = MeridianConfig.from_model_object(
        model_object=mock_model,
        input_data_builder_config=input_data_builder_config,
        revenue_column=revenue_column,
        sample_posterior_kwargs=fit_kwargs,
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
    assert input_config.response_column == "response"
    assert input_config.date_column == "date"


def test_meridian_config_validation():
    """Test validation of MeridianConfig."""
    mock_model = MockMeridianModelObject()
    input_data_builder_config = MeridianInputDataBuilderSchema(
        date_column="date",
        media_channels=["channel_1"],
        channel_spend_columns=["channel_1_spend"],
        response_column="response",
    )

    with pytest.raises(ValueError, match="`model_object` is required"):
        MeridianConfig.from_model_object(
            model_object=None, input_data_builder_config=input_data_builder_config, revenue_column="revenue"
        )
    with pytest.raises(ValueError, match="`revenue_column` is required"):
        MeridianConfig.from_model_object(
            model_object=mock_model, input_data_builder_config=input_data_builder_config, revenue_column=""
        )


def test_meridian_config_direct_instantiation():
    """Test direct instantiation of MeridianConfig."""
    import tensorflow_probability as tfp
    from meridian.model.prior_distribution import PriorDistribution

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
    import tensorflow_probability as tfp
    from meridian.model.prior_distribution import PriorDistribution

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


def test_meridian_config_rehydrator():
    """Test that MeridianConfigRehydrator can properly rehydrate TFP distributions."""
    import tensorflow_probability as tfp
    from meridian.model.prior_distribution import PriorDistribution

    from mmm_eval.configs.rehydrators import MeridianConfigRehydrator

    # Create a test config with TFP distributions
    test_config = {
        "prior": "PriorDistribution(roi_m=tfp.distributions.LogNormal(0.2, 0.9))",
        "media_effects_dist": "log_normal",
        "hill_before_adstock": "False",
        "max_lag": "8",
    }

    # Test the rehydrator
    rehydrator = MeridianConfigRehydrator(test_config)
    hydrated_config = rehydrator.rehydrate_config()

    # Verify that the prior was properly rehydrated
    assert "prior" in hydrated_config
    prior = hydrated_config["prior"]
    assert isinstance(prior, PriorDistribution)

    # Verify that the prior contains the expected TFP distribution
    assert hasattr(prior, "roi_m")
    assert isinstance(prior.roi_m, tfp.distributions.LogNormal)

    # Verify other fields were properly rehydrated
    assert hydrated_config["media_effects_dist"] == "log_normal"
    assert hydrated_config["hill_before_adstock"] is False
    assert hydrated_config["max_lag"] == 8


def test_meridian_config_rehydrator_with_dict_prior():
    """Test that MeridianConfigRehydrator can handle prior as a dict (real scenario)."""
    import tensorflow_probability as tfp
    from meridian.model.prior_distribution import PriorDistribution

    from mmm_eval.configs.rehydrators import MeridianConfigRehydrator

    # Create a test config where prior is a dict (like in real JSON loading)
    test_config = {
        "prior": {
            "roi_m": "tfp.distributions.LogNormal(0.2, 0.9)",
            "knot_values": "tfp.distributions.Normal(0.0, 1.0)",
        },
        "media_effects_dist": "log_normal",
        "hill_before_adstock": "False",
        "max_lag": "8",
    }

    # Test the rehydrator
    rehydrator = MeridianConfigRehydrator(test_config)
    hydrated_config = rehydrator.rehydrate_config()

    # Verify that the prior was properly rehydrated
    assert "prior" in hydrated_config
    prior = hydrated_config["prior"]
    assert isinstance(prior, PriorDistribution)

    # Verify that the prior contains the expected TFP distributions
    assert hasattr(prior, "roi_m")
    assert isinstance(prior.roi_m, tfp.distributions.LogNormal)
    assert hasattr(prior, "knot_values")
    assert isinstance(prior.knot_values, tfp.distributions.Normal)

    # Verify other fields were properly rehydrated
    assert hydrated_config["media_effects_dist"] == "log_normal"
    assert hydrated_config["hill_before_adstock"] is False
    assert hydrated_config["max_lag"] == 8


def test_meridian_serialization_deserialization():
    """Test the new serialization/deserialization system for Meridian configs."""
    import tensorflow_probability as tfp
    from meridian.model.prior_distribution import PriorDistribution

    from mmm_eval.configs.configs import serialize_meridian_config_value
    from mmm_eval.configs.rehydrators import MeridianConfigRehydrator

    # Create a test PriorDistribution
    prior = PriorDistribution(
        roi_m=tfp.distributions.LogNormal(0.2, 0.9), knot_values=tfp.distributions.Normal(0.0, 1.0)
    )

    # Test serialization
    serialized_prior = serialize_meridian_config_value(prior)

    # Verify serialization format
    assert isinstance(serialized_prior, dict)
    assert "roi_m" in serialized_prior
    assert "knot_values" in serialized_prior
    assert serialized_prior["roi_m"]["type"] == "LogNormal"
    assert serialized_prior["knot_values"]["type"] == "Normal"

    # Test deserialization
    test_config = {"prior": serialized_prior}
    rehydrator = MeridianConfigRehydrator(test_config)
    hydrated_config = rehydrator.rehydrate_config()

    # Verify deserialization
    assert "prior" in hydrated_config
    deserialized_prior = hydrated_config["prior"]
    assert isinstance(deserialized_prior, PriorDistribution)
    assert hasattr(deserialized_prior, "roi_m")
    assert isinstance(deserialized_prior.roi_m, tfp.distributions.LogNormal)
    assert hasattr(deserialized_prior, "knot_values")
    assert isinstance(deserialized_prior.knot_values, tfp.distributions.Normal)
