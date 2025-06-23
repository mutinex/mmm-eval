from typing import Any, Optional
from pydantic import BaseModel
from mmm_eval.adapters.experimental.schemas import PyMCModelSchema, PyMCFitSchema, PyMCStringConfigSchema
from mmm_eval.configs.rehydrators import PyMCConfigRehydrator
from mmm_eval.configs.utils import save_config, load_config

class Config:
    """
    Base class for all configs that have a schema

    Args:
        config (dict[str, Any]): The config to validate.
        schema_class (BaseModel): The schema to validate the config against.
    """
    def __init__(self, config: dict[str, Any], schema_class: BaseModel):
        self.config = config
        self.schema_class = schema_class
        self.validate_schema()

    def validate_schema(self) -> "Config":
        """
        Validate a config against the schema provided.
        """
        self.config = self.schema_class(**self.config).model_dump(exclude_unset=True)
        return self


class EvalConfig:
    """
    Base class for configs that are used to fit a model and run the evaluation suite.
    """
    def __init__(self, model_object: Any):
        self.model_object = model_object
        

class PyMCConfig(EvalConfig):
    """
    Evalulation config for PyMC MMM framework.
    """
    def __init__(self, model_object: Optional[Any] = None, fit_kwargs: Optional[dict[str, Any]] = None, target_column: Optional[str] = None):
        super().__init__(model_object)
        if model_object is not None:
            self.model_config = Config(self.filter_input_to_schema(model_object.__dict__, PyMCModelSchema), PyMCModelSchema)
        if fit_kwargs is not None:
            self.fit_config = Config(self.filter_input_to_schema(fit_kwargs, PyMCFitSchema), PyMCFitSchema)
        if target_column is not None:
            self.target_column = target_column

    def filter_input_to_schema(self, config: dict[str, Any], schema_class: BaseModel) -> dict[str, Any]:
        """
        Filter keys from the config to only include keys that are in both input and schema.

        Required because model object __dict__ has extra keys which are valid but not desired.

        Args:
            config (dict): The config to filter.
            schema_class (BaseModel): The schema to reference for filtering.

        Returns:
            dict[str, Any]: A filtered config dictionary.
        """
        schema_keys = set(schema_class.model_fields.keys())
        filtered_dict = {k: v for k, v in config.items() if k in schema_keys}
        return filtered_dict

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PyMCConfig":
        """
        Create a PyMCConfig from a dict.

        Args:
            config_dict (dict): Should have 'model_config', 'fit_config', and 'target_column' keys.

        Returns:
            PyMCConfig: An instance of the config class.
        """
        obj = cls()
        obj.model_config = Config(
            config_dict.get("model_config", {}), PyMCModelSchema
        )
        obj.fit_config = Config(
            config_dict.get("fit_config", {}), PyMCFitSchema
        )
        obj.target_column = config_dict.get("target_column")
        return obj
        

    def save_config(self, save_path: str, file_name: str) -> "PyMCConfig":
        """
        Save the stringified config to a JSON file.

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
        """
        Load the config from a JSON file.
        Args:
            config_path (str): The path to the config file.

        Returns:
            PyMCConfig: The config object updated with the loaded config.
        """
        string_config = load_config(config_path)
        string_config = PyMCStringConfigSchema(**string_config).model_dump(exclude_unset=True)
        return cls.from_dict(string_config)
    
    def _rehydrate_config(self, config: dict[str, Any], schema_class: BaseModel) -> Config:
        """
        Rehydrate the config from a stringified config.

        Args:
            config (dict[str, Any]): The config to rehydrate.
            schema_class (BaseModel): The schema to rehydrate to.

        Returns:
            Config: A validated Config object with the rehydrated config.
        """
        filtered_config = self.filter_input_to_schema(config, schema_class)
        hydrated_config = PyMCConfigRehydrator(filtered_config).rehydrate_config()
        return Config(hydrated_config, schema_class)