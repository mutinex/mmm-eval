from typing import Any

from pydantic import BaseModel

from mmm_eval.adapters.experimental.schemas import (
    PyMCFitSchema,
    PyMCModelSchema,
    PyMCStringConfigSchema,
)
from mmm_eval.configs.rehydrators import PyMCConfigRehydrator
from mmm_eval.configs.utils import load_config, save_config


class Config:
    """Base class for all configs that have a schema."""

    def __init__(self, config: dict[str, Any], schema_class: BaseModel):
        """Initialize the Config.

        Args:
            config: The config to validate.
            schema_class: The schema to validate the config against.

        """
        self.config = config
        self.schema_class = schema_class
        self.validate_schema()

    def validate_schema(self) -> "Config":
        """Validate a config against the schema provided."""
        self.config = self.schema_class(**self.config).model_dump(exclude_unset=True)
        return self


class EvalConfig:
    """Base class for configs that are used to fit models and run the evaluation suite."""

    def __init__(self, model_object: Any):
        """Initialize the EvalConfig.

        Args:
            model_object: The model object to use for the config.

        """
        self.model_object = model_object


class PyMCConfig(EvalConfig):
    """Evaluation config for the PyMC MMM framework."""

    def __init__(
        self,
        model_object: Any,
        fit_kwargs: dict[str, Any],
        revenue_column: str,
        response_column: str | None = None,
    ):
        """Instantiate a PyMCConfig manually.

        Args:
            model_object: The PyMC model object.
            fit_kwargs: The arguments passed to `.fit()`.
            revenue_column: The column containing the revenue variable.
            response_column: The column containing the response variable (optional).

        """
        super().__init__(model_object)

        if model_object is None:
            raise ValueError("`model_object` is required.")
        if not fit_kwargs:
            raise ValueError("`fit_kwargs` is required.")
        if not revenue_column:
            raise ValueError("`revenue_column` is required")

        self.response_column = response_column or revenue_column
        self.revenue_column = revenue_column

        self.model_config = Config(
            self._filter_input_to_schema(model_object.__dict__, PyMCModelSchema),
            PyMCModelSchema,
        )
        self.fit_config = Config(
            self._filter_input_to_schema(fit_kwargs, PyMCFitSchema),
            PyMCFitSchema,
        )

    def save_config(self, save_path: str, file_name: str) -> "PyMCConfig":
        """Save the config to a JSON file."""
        string_config = {
            "model_config": {k: repr(v) for k, v in self.model_config.config.items()} if self.model_config else None,
            "fit_config": {k: repr(v) for k, v in self.fit_config.config.items()} if self.fit_config else None,
            "response_column": self.response_column,
            "revenue_column": self.revenue_column,
        }
        save_config(string_config, save_path, file_name)
        return self

    @classmethod
    def load_config(cls, config_path: str) -> "PyMCConfig":
        """Load the config from a JSON file and return a valid PyMCConfig object."""
        string_config = load_config(config_path)
        parsed_config = PyMCStringConfigSchema(**string_config).model_dump(exclude_unset=True)
        return cls.from_dict(parsed_config)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PyMCConfig":
        """Rehydrate a PyMCConfig from a dictionary (e.g. after JSON parsing)."""
        model_config = cls._rehydrate_config(config_dict.get("model_config", {}), PyMCModelSchema)
        fit_config = cls._rehydrate_config(config_dict.get("fit_config", {}), PyMCFitSchema)

        obj = cls.__new__(cls)  # Bypass __init__
        super(PyMCConfig, obj).__init__(model_object=None)

        obj.model_config = model_config
        obj.fit_config = fit_config
        obj.response_column = config_dict.get("response_column")
        obj.revenue_column = config_dict.get("revenue_column", obj.response_column)

        return obj

    @staticmethod
    def _rehydrate_config(config: dict[str, Any], schema_class: BaseModel) -> Config:
        """Convert stringified config values to Python objects, then validate."""
        filtered_config = PyMCConfig._filter_input_to_schema(config, schema_class)
        hydrated_config = PyMCConfigRehydrator(filtered_config).rehydrate_config()
        return Config(hydrated_config, schema_class)

    @staticmethod
    def _filter_input_to_schema(config: dict[str, Any], schema_class: BaseModel) -> dict[str, Any]:
        """Filter a dict to only include keys valid for the given schema."""
        schema_keys = set(schema_class.model_fields.keys())
        return {k: v for k, v in config.items() if k in schema_keys}
