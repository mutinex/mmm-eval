from typing import Any, Optional
from mmm_eval.adapters.experimental.schemas import PyMCModelSchema, PyMCFitSchema
from mmm_eval.configs.rehydrators import PyMCConfigRehydrator
from mmm_eval.configs.utils import save_config, load_config

class Config:
    def __init__(self, config: dict[str, Any], schema_class: Any):
        self.config = config
        self.schema_class = schema_class
        self.validate_schema()

    def validate_schema(self):
        """
        Validate a config against the schema provided.

        """
        self.config = self.schema_class(**self.config).model_dump(exclude_unset=True)


class EvalConfig:
    def __init__(self, model_object: Any):
        self.model_object = model_object
        

class PyMCConfig(EvalConfig):
    def __init__(self, model_object: Optional[Any] = None, fit_kwargs: Optional[dict[str, Any]] = None, target_column: Optional[str] = None):
        super().__init__(model_object)
        if model_object is not None:
            self.model_config = Config(self.filter_input_to_schema(model_object.__dict__, PyMCModelSchema), PyMCModelSchema)
        if fit_kwargs is not None:
            self.fit_config = Config(self.filter_input_to_schema(fit_kwargs, PyMCFitSchema), PyMCFitSchema)
        if target_column is not None:
            self.target_column = target_column

    def filter_input_to_schema(self, config: dict[str, Any], schema_class: PyMCModelSchema | PyMCFitSchema) -> Config:
        """
        Filter keys from the config to only include keys that are in both input and schema.

        Required because model object __dict__ has extra keys which are valid but not desired.

        Args:
            config (dict): The config to filter.
            schema_class (PyMCModelSchema | PyMCFitSchema): The schema to filter to.

        Returns:
            Config: A validated Config object with the filtered config.
        """
        schema_keys = set(schema_class.model_fields.keys())
        filtered_dict = {k: v for k, v in config.items() if k in schema_keys}
        return filtered_dict


    def save_config(self, save_path: str, file_name: str):
        string_config = {
        "model_config": {k: repr(v) for k, v in self.model_config.config.items()},
        "fit_config": {k: repr(v) for k, v in self.fit_config.config.items()},
        "target_column": self.target_column,
    }
        save_config(string_config, save_path, file_name)
        return self
    
    def load_config(self, config_path: str):
        string_config = load_config(config_path)
        
        self.model_config = self._rehydrate_config(string_config["model_config"], PyMCModelSchema)
        self.fit_config = self._rehydrate_config(string_config["fit_config"], PyMCFitSchema)
        self.target_column = string_config["target_column"]
        return self
    
    def _rehydrate_config(self, config: dict[str, Any], schema_class: PyMCModelSchema | PyMCFitSchema):
        filtered_config = self.filter_input_to_schema(config, schema_class)
        hydrated_config = PyMCConfigRehydrator(filtered_config).rehydrate_config()
        return Config(hydrated_config, schema_class)

class MeridianConfig(EvalConfig):
    def __init__(self):
        pass