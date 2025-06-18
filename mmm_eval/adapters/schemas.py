from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union
from datetime import datetime
import pymc_marketing.mmm.components.adstock as adstock
import pymc_marketing.mmm.components.saturation as saturation
import inspect

def build_class_registry(*modules):
    registry = {}
    for mod in modules:
        registry.update({
            name: cls
            for name, cls in inspect.getmembers(mod, inspect.isclass)
            if cls.__module__.startswith(mod.__name__)
        })
    return registry

def valid_adstock_classes() -> list:
    return list(build_class_registry(adstock).keys())

def valid_saturation_classes() -> list:
    return list(build_class_registry(saturation).keys())

# Get the actual class objects for validation
ADSTOCK_CLASSES = build_class_registry(adstock)
SATURATION_CLASSES = build_class_registry(saturation)

class PyMCInputDataSchema(BaseModel):
    """Schema for input CSV data"""
    date: datetime
    channel_spend: Dict[str, float] = Field(description="Channel spend columns")
    response: float = Field(description="Target response variable")
    control_vars: Dict[str, float] = Field(description="Control variables")
    revenue: float = Field(description="Revenue column")

class PyMCConfigSchema(BaseModel):
    """Schema for PyMC Config Dictionary"""
    date_column: Optional[str] = Field(
        default=None,
        description="Column name of the date variable."
    )
    channel_columns: Optional[List[str]] = Field(
        default=None,
        description="Column names of the media channel variables.",
        min_items=1
    )
    response_column: Optional[str] = Field(
        default=None,
        description="Column name of the response variable."
    )
    revenue_column: Optional[str] = Field(
        default=None,
        description="Column name of the revenue variable."
    )
    adstock: Optional[Union[tuple(ADSTOCK_CLASSES.values())]] = Field(
        default=None,
        description="Type of adstock transformation to apply."
    )
    saturation: Optional[Union[tuple(SATURATION_CLASSES.values())]] = Field(
        default=None,
        description="Type of saturation transformation to apply."
    )
    time_varying_intercept: bool = Field(
        default=False,
        description="Whether to consider time-varying intercept."
    )
    time_varying_media: bool = Field(
        default=False,
        description="Whether to consider time-varying media contributions."
    )
    model_config: Optional[Dict] = Field(
        default=None,
        description="Model configuration."
    )
    sampler_config: Optional[Dict] = Field(
        default=None,
        description="Sampler configuration."
    )
    validate_data: bool = Field(
        default=True,
        description="Whether to validate the data before fitting to model"
    )
    control_columns: Optional[List[str]] = Field(
        default=None,
        description="Column names of the control variables."
    )
    yearly_seasonality: Optional[int] = Field(
        default=None,
        description="Number of yearly seasonality components."
    )
    adstock_first: bool = Field(
        default=True,
        description="Whether to apply adstock first."
    )
    dag: Optional[str] = Field(
        default=None,
        description="Optional DAG provided as a string Dot format for causal identification."
    )
    treatment_nodes: Optional[Union[List[str], tuple[str]]] = Field(
        default=None,
        description="Column names of the variables of interest to identify causal effects on outcome."
    )
    outcome_node: Optional[str] = Field(
        default=None,
        description="Name of the outcome variable."
    )
    fit_kwargs: Optional[Dict] = Field(
        default=None,
        description="Additional arguments for model fitting."
    )

    @validator('channel_columns')
    def validate_channel_columns(cls, v):
        if v is not None and not v:
            raise ValueError("channel_columns must not be empty")
        return v

    @validator('adstock')
    def validate_adstock(cls, v):
        if v is not None:
            valid_classes = list(ADSTOCK_CLASSES.values())
            if not any(isinstance(v, cls) for cls in valid_classes):
                valid_names = list(ADSTOCK_CLASSES.keys())
                raise ValueError(f"adstock must be one of {valid_names}, got {type(v).__name__}")
        return v

    @validator('saturation')
    def validate_saturation(cls, v):
        if v is not None:
            valid_classes = list(SATURATION_CLASSES.values())
            if not any(isinstance(v, cls) for cls in valid_classes):
                valid_names = list(SATURATION_CLASSES.keys())
                raise ValueError(f"saturation must be one of {valid_names}, got {type(v).__name__}")
        return v

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"  # Allow extra fields not defined in schema
    }