from datetime import datetime
from typing import Any, Optional
from pathlib import Path
import json

from pydantic import BaseModel, Field, model_validator
from pymc_marketing.mmm import MMM

from mmm_eval.configs.rehydrators import PyMCConfigRehydrator
from mmm_eval.configs.base import BaseConfig

from mmm_eval.adapters.experimental.schemas import (
    PyMCFitSchema,
    PyMCModelSchema,
)

class PyMCConfig(BaseConfig):
    """Evaluation config for the PyMC MMM framework."""
    pymc_model_config: PyMCModelSchema = Field(..., description="Model configuration")
    fit_config: PyMCFitSchema = Field(..., description="Fit configuration")
    revenue_column: str = Field(..., description="Column containing the revenue variable")
    response_column: Optional[str] = Field(None, description="Column containing the response variable")
    date_column: Optional[datetime | str] = Field(None, description="Column containing the date variable")
    channel_columns: Optional[list[str]] = Field(None, description="Columns containing the channel variables")

    @model_validator(mode='after')
    def set_response_column(self) -> 'PyMCConfig':
        if self.response_column is None:
            self.response_column = self.revenue_column
        return self
    
    # Set in the config the date column so we can use it in the data pipeline
    @model_validator(mode='after')
    def set_date_column(self) -> 'PyMCConfig':
        if self.date_column is None:
            self.date_column = self.pymc_model_config.date_column
        return self

    # Set in the config the channel columns so we can use it in the data pipeline
    @model_validator(mode='after')
    def set_channel_columns(self) -> 'PyMCConfig':
        if self.channel_columns is None:
            self.channel_columns = self.pymc_model_config.channel_columns
        return self
    
    @classmethod
    def from_model_object(
        cls,
        model_object: MMM,
        fit_kwargs: dict[str, Any],
        revenue_column: str,
        response_column: Optional[str] = None
    ) -> 'PyMCConfig':
        """Create a PyMCConfig from a model object and fit kwargs.
        
        Args:
            model_object: The PyMC model object
            fit_kwargs: The arguments passed to `.fit()`
            revenue_column: The column containing the revenue variable
            response_column: The column containing the response variable (optional)
            
        Returns:
            A validated PyMCConfig instance
        """
        cls._validate_inputs(model_object, fit_kwargs, revenue_column)
        
        model_config = cls._extract_model_config(model_object)
        fit_config = cls._extract_fit_config(fit_kwargs)
        
        return cls(
            pymc_model_config=model_config,
            fit_config=fit_config,
            revenue_column=revenue_column,
            response_column=response_column
        )
    
    @staticmethod
    def _validate_inputs(
        model_object: Any, 
        fit_kwargs: dict[str, Any], 
        revenue_column: str
    ) -> None:
        """Validate the input parameters."""
        if model_object is None:
            raise ValueError("`model_object` is required.")
        if not fit_kwargs:
            raise ValueError("`fit_kwargs` is required.")
        if not revenue_column:
            raise ValueError("`revenue_column` is required")
    
    @staticmethod
    def _extract_model_config(model_object: MMM) -> PyMCModelSchema:
        """Extract and validate model configuration from a model object."""
        model_fields = set(PyMCModelSchema.model_fields.keys())
        filtered_config = {
            key: value for key, value in model_object.__dict__.items() 
            if key in model_fields
        }
        return PyMCModelSchema(**filtered_config)
    
    @staticmethod
    def _extract_fit_config(fit_kwargs: dict[str, Any]) -> PyMCFitSchema:
        """Extract and validate fit configuration from fit kwargs."""
        fit_fields = set(PyMCFitSchema.model_fields.keys())
        filtered_config = {
            key: value for key, value in fit_kwargs.items() 
            if key in fit_fields
        }
        return PyMCFitSchema(**filtered_config)

    def save_model_object_to_json(self, save_path: str, file_name: str) -> 'PyMCConfig':
        """Save the config to a JSON file."""
        config_dict = self.model_dump()
        config_dict["pymc_model_config"] = {k: repr(v) for k, v in config_dict["pymc_model_config"].items()}
        config_dict["fit_config"] = {k: repr(v) for k, v in config_dict["fit_config"].items()}
        BaseConfig._save_json_file(save_path, file_name, config_dict)
        return self

    @classmethod
    def load_model_config_from_json(cls, config_path: str) -> 'PyMCConfig':
        """Load the config from a JSON file."""
        config_dict = cls._load_json_file(config_path)
        return cls._from_string_dict(config_dict)

    @classmethod
    def _from_string_dict(cls, config_dict: dict[str, Any]) -> 'PyMCConfig':
        if "pymc_model_config" in config_dict:
            rehydrator = PyMCConfigRehydrator(config_dict["pymc_model_config"])
            hydrated_model_config = rehydrator.rehydrate_config()
            config_dict["pymc_model_config"] = PyMCModelSchema(**hydrated_model_config)
        if "fit_config" in config_dict:
            rehydrator = PyMCConfigRehydrator(config_dict["fit_config"])
            hydrated_fit_config = rehydrator.rehydrate_config()
            config_dict["fit_config"] = PyMCFitSchema(**hydrated_fit_config)
        return cls.model_validate(config_dict)
