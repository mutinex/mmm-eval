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
    """Evalulation config for PyMC MMM framework."""

    def __init__(
        self,
        model_object: Any | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        target_column: str | None = None,
    ):
        """Initialize the PyMCConfig.

        Args:
            model_object: The PyMC model object.
            fit_kwargs: The args passed to .fit().
            target_column: The column containing the response variable.

        """
        super().__init__(model_object)
        if model_object is not None:
            self.model_config = Config(
                self._filter_input_to_schema(model_object.__dict__, PyMCModelSchema),
                PyMCModelSchema,
            )
        if fit_kwargs is not None:
            self.fit_config = Config(self._filter_input_to_schema(fit_kwargs, PyMCFitSchema), PyMCFitSchema)
        if target_column is not None:
            self.target_column = target_column

    @staticmethod
    def _filter_input_to_schema(config: dict[str, Any], schema_class: BaseModel) -> dict[str, Any]:
        """Filter the input to the schema.

        Args:
            config: The config to filter.
            schema_class: The schema to filter the config against.

        """
        schema_keys = set(schema_class.model_fields.keys())
        return {k: v for k, v in config.items() if k in schema_keys}

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PyMCConfig":
        """Create a PyMCConfig from a dict.

        Args:
            config_dict (dict): Should have 'model_config', 'fit_config', and 'target_column' keys.

        Returns:
            PyMCConfig: An instance of the config class.

        """
        obj = cls()
        obj.model_config = cls._rehydrate_config(config_dict.get("model_config", {}), PyMCModelSchema)
        obj.fit_config = cls._rehydrate_config(config_dict.get("fit_config", {}), PyMCFitSchema)
        obj.target_column = config_dict.get("target_column")
        return obj

    def save_config(self, save_path: str, file_name: str) -> "PyMCConfig":
        """Save the stringified config to a JSON file.

        Args:
            save_path (str): The path to save the config to.
            file_name (str): The name of the file to save the config to.

        Returns:
            self: The config object.

        """
        string_config = {
            "model_config": {k: repr(v) for k, v in self.model_config.config.items()},
            "fit_config": {k: repr(v) for k, v in self.fit_config.config.items()},
            "target_column": self.target_column,
        }
        save_config(string_config, save_path, file_name)
        return self

    @classmethod
    def load_config(cls, config_path: str) -> "PyMCConfig":
        """Load the config from a JSON file.

        Args:
            config_path (str): The path to the config file.

        Returns:
            PyMCConfig: The config object updated with the loaded config.

        """
        string_config = load_config(config_path)
        string_config = PyMCStringConfigSchema(**string_config).model_dump(exclude_unset=True)
        return cls.from_dict(string_config)

    @staticmethod
    def _rehydrate_config(config: dict[str, Any], schema_class: BaseModel) -> Config:
        """Rehydrate the config from a stringified config.

        Args:
            config (dict[str, Any]): The config to rehydrate.
            schema_class (BaseModel): The schema to rehydrate to.

        """
        filtered_config = PyMCConfig._filter_input_to_schema(config, schema_class)
        hydrated_config = PyMCConfigRehydrator(filtered_config).rehydrate_config()
        return Config(hydrated_config, schema_class)
