from typing import Any

from pydantic import (
    Field,
    computed_field,
)
from pymc_marketing.mmm import MMM

from mmm_eval.adapters.schemas import (
    MeridianInputDataBuilderSchema,
    MeridianModelSpecSchema,
    MeridianSamplePosteriorSchema,
    PyMCFitSchema,
    PyMCModelSchema,
)
from mmm_eval.configs.base import BaseConfig
from mmm_eval.configs.constants import ConfigConstants
from mmm_eval.configs.rehydrators import PyMCConfigRehydrator, MeridianConfigRehydrator


def serialize_tfp_distribution(dist):
    """Serialize a TFP distribution to a dict that can be reconstructed."""
    import tensorflow_probability as tfp
    
    dist_type = type(dist).__name__
    
    # Get the parameters
    if hasattr(dist, 'parameters'):
        params = dist.parameters
    else:
        # Fallback for distributions without parameters attribute
        params = {}
        for param_name in ['loc', 'scale', 'concentration', 'rate', 'low', 'high']:
            if hasattr(dist, param_name):
                params[param_name] = getattr(dist, param_name)
    
    # Recursively serialize parameters
    def serialize_param(value):
        if hasattr(value, '__class__') and 'tensorflow_probability' in str(type(value)):
            return serialize_tfp_distribution(value)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            return value
        elif isinstance(value, list):
            return [serialize_param(v) for v in value]
        elif isinstance(value, dict):
            return {k: serialize_param(v) for k, v in value.items()}
        else:
            return repr(value)

    serializable_params = {key: serialize_param(val) for key, val in params.items()}

    return {
        "type": dist_type,
        "parameters": serializable_params
    }


def serialize_prior_distribution(prior):
    """Serialize a PriorDistribution object to a dict that can be reconstructed."""
    def serialize_value(val):
        if hasattr(val, '__class__') and 'tensorflow_probability' in str(type(val)):
            return serialize_tfp_distribution(val)
        elif isinstance(val, (int, float, str, bool)) or val is None:
            return val
        elif isinstance(val, list):
            return [serialize_value(v) for v in val]
        elif isinstance(val, dict):
            return {k: serialize_value(v) for k, v in val.items()}
        else:
            return repr(val)
    serialized_prior = {}
    for attr_name, attr_value in prior.__dict__.items():
        serialized_prior[attr_name] = serialize_value(attr_value)
    return serialized_prior


def serialize_meridian_config_value(value):
    """Serialize a Meridian config value, handling special cases for TFP objects."""
    import tensorflow_probability as tfp
    from meridian.model.prior_distribution import PriorDistribution
    
    if isinstance(value, PriorDistribution):
        return serialize_prior_distribution(value)
    elif hasattr(value, '__class__') and 'tensorflow_probability' in str(type(value)):
        return serialize_tfp_distribution(value)
    else:
        # Use repr for other objects (like PyMC does)
        return repr(value)


class PyMCConfig(BaseConfig):
    """Evaluation config for the PyMC MMM framework."""

    pymc_model_config: PyMCModelSchema = Field(..., description="Model configuration")
    fit_config: PyMCFitSchema = Field(default=PyMCFitSchema(), description="Fit configuration")

    @computed_field
    @property
    def date_column(self) -> str:
        """Return the date column."""
        return self.pymc_model_config.date_column

    @computed_field
    @property
    def channel_columns(self) -> list[str]:
        """Return the channel columns."""
        return self.pymc_model_config.channel_columns

    @computed_field
    @property
    def control_columns(self) -> list[str] | None:
        """Return the control columns."""
        return self.pymc_model_config.control_columns

    @property
    def pymc_model_config_dict(self) -> dict[str, Any]:
        """Return the model configuration as a dictionary."""
        return self.pymc_model_config.model_dump()

    @property
    def fit_config_dict(self) -> dict[str, Any]:
        """Return the fit configuration as a dictionary of user provided values."""
        return self.fit_config.fit_config_dict_without_non_provided_fields

    @property
    def predict_config_dict(self) -> dict[str, Any]:
        """Return the predict configuration as a dictionary of user provided values."""
        return {
            key: value
            for key, value in self.fit_config_dict.items()
            if key in ConfigConstants.PyMCIntersectingFitPredictKwargs.all()
        }

    @classmethod
    def from_model_object(
        cls,
        model_object: MMM,
        revenue_column: str,
        fit_kwargs: dict[str, Any] | None = None,
        response_column: str | None = None,
    ) -> "PyMCConfig":
        """Create a PyMCConfig from a model object and fit kwargs.

        Args:
            model_object: The PyMC model object
            revenue_column: The column containing the revenue variable
            fit_kwargs: The arguments passed to `.fit()` (optional, will use defaults if not provided)
            response_column: The column containing the response variable (optional)

        Returns:
            A validated PyMCConfig instance

        """
        cls._validate_inputs(model_object, revenue_column)

        model_config = cls._extract_model_config(model_object)
        fit_config = cls._extract_fit_config(fit_kwargs) if fit_kwargs else PyMCFitSchema()

        return cls(
            pymc_model_config=model_config,
            fit_config=fit_config,
            revenue_column=revenue_column,
            response_column=response_column,
        )

    @staticmethod
    def _validate_inputs(model_object: Any, revenue_column: str) -> None:
        """Validate the input parameters."""
        if model_object is None:
            raise ValueError("`model_object` is required.")
        if not revenue_column:
            raise ValueError("`revenue_column` is required")

    @staticmethod
    def _extract_model_config(model_object: MMM) -> PyMCModelSchema:
        """Extract and validate model configuration from a model object."""
        model_fields = set(PyMCModelSchema.model_fields.keys())
        filtered_config = {key: value for key, value in model_object.__dict__.items() if key in model_fields}
        return PyMCModelSchema(**filtered_config)

    @staticmethod
    def _extract_fit_config(fit_kwargs: dict[str, Any]) -> PyMCFitSchema:
        """Extract and validate fit configuration from fit kwargs."""
        fit_fields = set(PyMCFitSchema.model_fields.keys())
        filtered_config = {key: value for key, value in fit_kwargs.items() if key in fit_fields}
        return PyMCFitSchema(**filtered_config)

    def save_model_object_to_json(self, save_path: str, file_name: str) -> "PyMCConfig":
        """Save the config to a JSON file."""
        config_dict = self.model_dump()
        config_dict[ConfigConstants.PyMCConfigAttributes.PYMC_MODEL_CONFIG] = {
            k: repr(v) for k, v in config_dict[ConfigConstants.PyMCConfigAttributes.PYMC_MODEL_CONFIG].items()
        }
        config_dict[ConfigConstants.PyMCConfigAttributes.FIT_CONFIG] = {
            k: repr(v) for k, v in config_dict[ConfigConstants.PyMCConfigAttributes.FIT_CONFIG].items()
        }
        BaseConfig._save_json_file(save_path, file_name, config_dict)
        return self

    @classmethod
    def load_model_config_from_json(cls, config_path: str) -> "PyMCConfig":
        """Load the config from a JSON file."""
        config_dict = cls._load_json_file(config_path)
        return cls._from_string_dict(config_dict)

    @classmethod
    def _from_string_dict(cls, config_dict: dict[str, Any]) -> "PyMCConfig":
        if ConfigConstants.PyMCConfigAttributes.PYMC_MODEL_CONFIG in config_dict:
            rehydrator = PyMCConfigRehydrator(config_dict[ConfigConstants.PyMCConfigAttributes.PYMC_MODEL_CONFIG])
            hydrated_model_config = rehydrator.rehydrate_config()
            config_dict[ConfigConstants.PyMCConfigAttributes.PYMC_MODEL_CONFIG] = PyMCModelSchema(
                **hydrated_model_config
            )
        if ConfigConstants.PyMCConfigAttributes.FIT_CONFIG in config_dict:
            rehydrator = PyMCConfigRehydrator(config_dict[ConfigConstants.PyMCConfigAttributes.FIT_CONFIG])
            hydrated_fit_config = rehydrator.rehydrate_config()
            config_dict[ConfigConstants.PyMCConfigAttributes.FIT_CONFIG] = PyMCFitSchema(**hydrated_fit_config)
        return cls.model_validate(config_dict)


class MeridianConfig(BaseConfig):
    """Evaluation config for the Google Meridian MMM framework."""

    input_data_builder_config: MeridianInputDataBuilderSchema = Field(
        ..., description="Input data builder configuration"
    )
    model_spec_config: MeridianModelSpecSchema = Field(..., description="Model specification configuration")
    sample_posterior_config: MeridianSamplePosteriorSchema = Field(
        default=MeridianSamplePosteriorSchema(), description="Sample posterior configuration"
    )

    @computed_field
    @property
    def date_column(self) -> str:
        """Return the date column."""
        return self.input_data_builder_config.date_column

    # TODO: consider renaming to "channel_spend_columns"
    @computed_field
    @property
    def channel_columns(self) -> list[str]:
        """Return the channel columns."""
        return self.input_data_builder_config.channel_spend_columns

    @computed_field
    @property
    def control_columns(self) -> list[str] | None:
        """Return the control columns."""
        return self.input_data_builder_config.control_columns

    @property
    def input_data_builder_config_dict(self) -> dict[str, Any]:
        """Return the input data builder configuration as a dictionary."""
        return self.input_data_builder_config.model_dump()

    @property
    def model_spec_config_dict(self) -> dict[str, Any]:
        """Return the model specification configuration as a dictionary."""
        return self.model_spec_config.model_dump()

    @property
    def sample_posterior_config_dict(self) -> dict[str, Any]:
        """Return the sample posterior configuration as a dictionary of user provided values."""
        return self.sample_posterior_config.fit_config_dict_without_non_provided_fields

    @classmethod
    def from_model_object(
        cls,
        model_object: Any,  # Meridian model object
        input_data_builder_config,
        revenue_column: str,
        sample_posterior_kwargs: dict[str, Any] | None = None,
    ) -> "MeridianConfig":
        """Create a MeridianConfig from a model object and fit kwargs.

        Args:
            model_object: The Meridian model object
            revenue_column: The column containing the revenue variable
            sample_posterior_kwargs: The arguments passed to `.fit()` (optional, will use
                defaults if not provided)
            response_column: The column containing the response variable (optional)

        Returns:
            A validated MeridianConfig instance

        """
        cls._validate_inputs(model_object, revenue_column)

        model_spec_config = cls._extract_model_spec_config(model_object)
        sample_posterior_config = cls._extract_sample_posterior_config(sample_posterior_kwargs) if sample_posterior_kwargs else MeridianSamplePosteriorSchema()

        return cls(
            input_data_builder_config=input_data_builder_config,
            model_spec_config=model_spec_config,
            sample_posterior_config=sample_posterior_config,
            revenue_column=revenue_column,
            response_column=input_data_builder_config.response_column,
        )

    @staticmethod
    def _validate_inputs(model_object: Any, revenue_column: str) -> None:
        """Validate the input parameters."""
        if model_object is None:
            raise ValueError("`model_object` is required.")
        if not revenue_column:
            raise ValueError("`revenue_column` is required")

    @staticmethod
    def _extract_model_spec_config(model_object: Any) -> MeridianModelSpecSchema:
        """Extract and validate model specification configuration from a model object."""
        model_spec_fields = set(MeridianModelSpecSchema.model_fields.keys())
        model_spec = model_object.model_spec
        filtered_config = {}
        for key in model_spec_fields:
            if hasattr(model_spec, key):
                filtered_config[key] = getattr(model_spec, key)
        return MeridianModelSpecSchema(**filtered_config)

    @staticmethod
    def _extract_sample_posterior_config(fit_kwargs: dict[str, Any]) -> MeridianSamplePosteriorSchema:
        """Extract and validate sample posterior configuration from fit kwargs."""
        sample_posterior_fields = set(MeridianSamplePosteriorSchema.model_fields.keys())
        filtered_config = {key: value for key, value in fit_kwargs.items() if key in sample_posterior_fields}
        return MeridianSamplePosteriorSchema(**filtered_config)

    def save_model_object_to_json(self, save_path: str, file_name: str) -> "MeridianConfig":
        """Save the config to a JSON file."""
        config_dict = self.model_dump()
        
        # Serialize input_data_builder_config values directly
        input_data_dict = {}
        for key, value in self.input_data_builder_config.model_dump().items():
            input_data_dict[key] = serialize_meridian_config_value(value)
        config_dict[ConfigConstants.MeridianConfigAttributes.INPUT_DATA_BUILDER_CONFIG] = input_data_dict
        
        # Serialize model_spec_config values directly
        model_spec_dict = {}
        for key, value in self.model_spec_config.model_dump().items():
            if key == "prior":
                # Handle prior field specially - serialize the actual PriorDistribution object
                model_spec_dict[key] = serialize_meridian_config_value(self.model_spec_config.prior)
            else:
                model_spec_dict[key] = serialize_meridian_config_value(value)
        config_dict[ConfigConstants.MeridianConfigAttributes.MODEL_SPEC_CONFIG] = model_spec_dict
        
        # Serialize sample_posterior_config values directly
        sample_posterior_dict = {}
        for key, value in self.sample_posterior_config.model_dump().items():
            sample_posterior_dict[key] = serialize_meridian_config_value(value)
        config_dict[ConfigConstants.MeridianConfigAttributes.SAMPLE_POSTERIOR_CONFIG] = sample_posterior_dict
        
        BaseConfig._save_json_file(save_path, file_name, config_dict)
        return self

    @classmethod
    def load_model_config_from_json(cls, config_path: str) -> "MeridianConfig":
        """Load the config from a JSON file."""
        config_dict = cls._load_json_file(config_path)
        return cls._from_string_dict(config_dict)

    @classmethod
    def _from_string_dict(cls, config_dict: dict[str, Any]) -> "MeridianConfig":
        if ConfigConstants.MeridianConfigAttributes.INPUT_DATA_BUILDER_CONFIG in config_dict:
            rehydrator = MeridianConfigRehydrator(config_dict[ConfigConstants.MeridianConfigAttributes.INPUT_DATA_BUILDER_CONFIG])
            hydrated_input_data_builder_config = rehydrator.rehydrate_config()
            config_dict[ConfigConstants.MeridianConfigAttributes.INPUT_DATA_BUILDER_CONFIG] = MeridianInputDataBuilderSchema(
                **hydrated_input_data_builder_config
            )
        if ConfigConstants.MeridianConfigAttributes.MODEL_SPEC_CONFIG in config_dict:
            rehydrator = MeridianConfigRehydrator(config_dict[ConfigConstants.MeridianConfigAttributes.MODEL_SPEC_CONFIG])
            hydrated_model_spec_config = rehydrator.rehydrate_config()
            config_dict[ConfigConstants.MeridianConfigAttributes.MODEL_SPEC_CONFIG] = MeridianModelSpecSchema(
                **hydrated_model_spec_config
            )
        if ConfigConstants.MeridianConfigAttributes.SAMPLE_POSTERIOR_CONFIG in config_dict:
            rehydrator = MeridianConfigRehydrator(config_dict[ConfigConstants.MeridianConfigAttributes.SAMPLE_POSTERIOR_CONFIG])
            hydrated_sample_posterior_config = rehydrator.rehydrate_config()
            config_dict[ConfigConstants.MeridianConfigAttributes.SAMPLE_POSTERIOR_CONFIG] = MeridianSamplePosteriorSchema(
                **hydrated_sample_posterior_config
            )
        return cls.model_validate(config_dict)
