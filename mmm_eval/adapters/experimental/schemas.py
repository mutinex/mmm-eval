"""Pydantic schemas for PyMC adapter validation."""

import inspect
from datetime import datetime
from typing import Any

import pymc_marketing.mmm.components.adstock as adstock
import pymc_marketing.mmm.components.saturation as saturation
from pydantic import BaseModel, Field, field_validator


def build_class_registry(*modules) -> dict[str, Any]:
    """Build a registry of classes from modules.

    Args:
        *modules: Modules to scan for classes

    Returns:
        Dictionary mapping class names to classes

    """
    registry = {}
    for mod in modules:
        registry.update(
            {
                name: cls
                for name, cls in inspect.getmembers(mod, inspect.isclass)
                if cls.__module__.startswith(mod.__name__)
            }
        )
    return registry


# Get the actual class objects for validation
ADSTOCK_CLASSES = build_class_registry(adstock)
SATURATION_CLASSES = build_class_registry(saturation)


class PyMCInputDataSchema(BaseModel):
    """Schema for input CSV data."""

    date: datetime
    channel_spend: dict[str, float] = Field(description="Channel spend columns")
    response: float = Field(description="Target response variable")
    control_vars: dict[str, float] = Field(description="Control variables")
    revenue: float = Field(description="Revenue column")


class PyMCConfigSchema(BaseModel):
    """Schema for PyMC Config Dictionary."""

    date_column: str | None = Field(default=None, description="Column name of the date variable.")
    channel_columns: list[str] | None = Field(
        default=None,
        description="Column names of the media channel variables.",
        min_length=1,
    )
    response_column: str | None = Field(default=None, description="Column name of the response variable.")
    revenue_column: str | None = Field(default=None, description="Column name of the revenue variable.")
    adstock: Any | None = Field(default=None, description="Type of adstock transformation to apply.")
    saturation: Any | None = Field(default=None, description="Type of saturation transformation to apply.")
    time_varying_intercept: bool = Field(default=False, description="Whether to consider time-varying intercept.")
    time_varying_media: bool = Field(
        default=False,
        description="Whether to consider time-varying media contributions.",
    )
    pymc_model_config: dict[str, Any] | None = Field(default=None, description="Model configuration.")
    sampler_config: dict[str, Any] | None = Field(default=None, description="Sampler configuration.")
    validate_data: bool = Field(default=True, description="Whether to validate the data before fitting to model")
    control_columns: list[str] | None = Field(default=None, description="Column names of the control variables.")
    yearly_seasonality: int | None = Field(default=None, description="Number of yearly seasonality components.")
    adstock_first: bool = Field(default=True, description="Whether to apply adstock first.")
    dag: str | None = Field(
        default=None,
        description="Optional DAG provided as a string Dot format for causal identification.",
    )
    treatment_nodes: list[str] | tuple[str, ...] | None = Field(
        default=None,
        description="Column names of the variables of interest to identify causal effects on outcome.",
    )
    outcome_node: str | None = Field(default=None, description="Name of the outcome variable.")
    fit_kwargs: dict[str, Any] | None = Field(default=None, description="Additional arguments for model fitting.")

    @field_validator("channel_columns")
    def validate_channel_columns(cls, v):
        """Validate channel columns are not empty.

        Args:
            v: Channel columns value

        Returns:
            Validated value

        Raises:
            ValueError: If channel columns is empty

        """
        if v is not None and not v:
            raise ValueError("channel_columns must not be empty")
        return v

    @field_validator("adstock")
    def validate_adstock(cls, v):
        """Validate adstock component.

        Args:
            v: Adstock value

        Returns:
            Validated value

        Raises:
            ValueError: If adstock is not a valid type

        """
        if v is not None:
            valid_classes = list(ADSTOCK_CLASSES.values())
            if not any(isinstance(v, cls) for cls in valid_classes):
                valid_names = list(ADSTOCK_CLASSES.keys())
                raise ValueError(f"adstock must be one of {valid_names}, got {type(v).__name__}")
        return v

    @field_validator("saturation")
    def validate_saturation(cls, v):
        """Validate saturation component.

        Args:
            v: Saturation value

        Returns:
            Validated value

        Raises:
            ValueError: If saturation is not a valid type

        """
        if v is not None:
            valid_classes = list(SATURATION_CLASSES.values())
            if not any(isinstance(v, cls) for cls in valid_classes):
                valid_names = list(SATURATION_CLASSES.keys())
                raise ValueError(f"saturation must be one of {valid_names}, got {type(v).__name__}")
        return v
