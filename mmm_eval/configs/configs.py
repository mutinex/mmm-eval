from typing import Any

from pydantic import (
    Field,
    computed_field,
)
from pymc_marketing.mmm import MMM

from mmm_eval.adapters.schemas import (
    PyMCFitSchema,
    PyMCModelSchema,
)
from mmm_eval.configs.base import BaseConfig
from mmm_eval.configs.constants import ConfigConstants
from mmm_eval.configs.rehydrators import PyMCConfigRehydrator


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
