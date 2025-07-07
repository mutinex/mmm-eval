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

    # @staticmethod
    # def _extract_input_data_builder_config(model_object: Any, channel_spend_columns: list[str],
    #                                        channel_impressions_columns,
    #                                        channel_reach_columns, channel_frequency_columns) -> MeridianInputDataBuilderSchema:
    #     """Extract and validate input data builder configuration from a model object.
        
    #     This method examines the Meridian model's input_data structure to reconstruct
    #     the input data builder configuration that was used to create the model.
        
    #     Args:
    #         model_object: The Meridian model object
            
    #     Returns:
    #         MeridianInputDataBuilderSchema containing the extracted configuration
            
    #     Raises:
    #         ValueError: If the model object doesn't have the expected structure
    #     """
    #     if not hasattr(model_object, 'input_data'):
    #         raise ValueError("Model object must have 'input_data' attribute")
            
    #     input_data = model_object.input_data
        
    #     # Extract basic information from input_data
    #     config = {}
        
    #     # Extract date column - this is typically the time dimension
    #     if hasattr(input_data, 'kpi') and input_data.kpi is not None and hasattr(input_data.kpi, 'time'):
    #         # The time column name is not directly available, so we'll use a default
    #         # This could be enhanced by examining the original data structure
    #         config['date_column'] = 'date'  # Default assumption
            
    #     # Extract media channels and spend columns from media data
    #     config['media_channels'] = [e.item() for e in list(input_data.media.media_channel)]
    #     config['channel_spend_columns'] = channel_spend_columns
    #     config['channel_impressions_columns'] = channel_impressions_columns
    #     config['channel_reach_columns'] = channel_reach_columns
    #     config['channel_frequency_columns'] = channel_frequency_columns
    #     #if media_data is not None:
    #         # Extract media channel information
    #         # for media_component in media_data:
    #         #     if hasattr(media_component, 'channel') and media_component.channel is not None:
    #         #         media_channels.append(media_component.channel)
                    
    #         #     # Determine the type of media data (spend, impressions, reach/frequency)
    #         #     if hasattr(media_component, 'spend') and media_component.spend is not None:
    #         #         channel_spend_columns.append(f"{media_component.channel}_spend")
                    
    #         #     if hasattr(media_component, 'impressions') and media_component.impressions is not None:
    #         #         channel_impressions_columns.append(f"{media_component.channel}_impressions")
                    
    #         #     if (hasattr(media_component, 'reach') and media_component.reach is not None and 
    #         #         hasattr(media_component, 'frequency') and media_component.frequency is not None):
    #         #         channel_reach_columns.append(f"{media_component.channel}_reach")
    #         #         channel_frequency_columns.append(f"{media_component.channel}_frequency")
            
    #         # if media_channels:
    #         #     config['media_channels'] = media_channels
    #         # if channel_spend_columns:
    #         #     config['channel_spend_columns'] = channel_spend_columns
            
    #         # if channel_impressions_columns:
    #         #     config['channel_impressions_columns'] = channel_impressions_columns
    #         # if channel_reach_columns:
    #         #     config['channel_reach_columns'] = channel_reach_columns
    #         #     config['channel_frequency_columns'] = channel_frequency_columns
                
    #     # Extract control columns
    #     controls_data = getattr(input_data, 'controls', None)
    #     if controls_data is not None:
    #         control_columns = []
    #         for control in controls_data:
    #             if hasattr(control, 'name') and control.name is not None:
    #                 control_columns.append(control.name)
    #         if control_columns:
    #             config['control_columns'] = control_columns
                
    #     # Extract organic media columns
    #     organic_media_data = getattr(input_data, 'organic_media', None)
    #     if organic_media_data is not None:
    #         organic_media_columns = []
    #         organic_media_channels = []
    #         for organic in organic_media_data:
    #             if hasattr(organic, 'name') and organic.name is not None:
    #                 organic_media_columns.append(organic.name)
    #             if hasattr(organic, 'channel') and organic.channel is not None:
    #                 organic_media_channels.append(organic.channel)
    #         if organic_media_columns:
    #             config['organic_media_columns'] = organic_media_columns
    #         if organic_media_channels:
    #             config['organic_media_channels'] = organic_media_channels
                
    #     # Extract non-media treatment columns
    #     non_media_treatments_data = getattr(input_data, 'non_media_treatments', None)
    #     if non_media_treatments_data is not None:
    #         non_media_treatment_columns = []
    #         for treatment in non_media_treatments_data:
    #             if hasattr(treatment, 'name') and treatment.name is not None:
    #                 non_media_treatment_columns.append(treatment.name)
    #         if non_media_treatment_columns:
    #             config['non_media_treatment_columns'] = non_media_treatment_columns
                
    #     # Extract response column - this is typically the KPI
    #     kpi_data = getattr(input_data, 'kpi', None)
    #     if kpi_data is not None:
    #         config['response_column'] = 'response'  # Default assumption
            
    #     # Validate that we have the minimum required fields
    #     required_fields = ['media_channels', 'channel_spend_columns', 'response_column']
    #     missing_fields = [field for field in required_fields if field not in config]
    #     if missing_fields:
    #         raise ValueError(f"Could not extract required fields from model: {missing_fields}")
            
    #     return MeridianInputDataBuilderSchema(**config)

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
        config_dict[ConfigConstants.MeridianConfigAttributes.INPUT_DATA_BUILDER_CONFIG] = {
            k: repr(v) for k, v in config_dict[ConfigConstants.MeridianConfigAttributes.INPUT_DATA_BUILDER_CONFIG].items()
        }
        config_dict[ConfigConstants.MeridianConfigAttributes.MODEL_SPEC_CONFIG] = {
            k: repr(v) for k, v in config_dict[ConfigConstants.MeridianConfigAttributes.MODEL_SPEC_CONFIG].items()
        }
        config_dict[ConfigConstants.MeridianConfigAttributes.SAMPLE_POSTERIOR_CONFIG] = {
            k: repr(v) for k, v in config_dict[ConfigConstants.MeridianConfigAttributes.SAMPLE_POSTERIOR_CONFIG].items()
        }
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
