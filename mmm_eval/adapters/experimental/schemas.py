from pydantic import BaseModel, Field, validator, InstanceOf
from typing import Annotated
from pymc_marketing.mmm.components.adstock import AdstockTransformation
from pymc_marketing.mmm.components.saturation import SaturationTransformation


class PyMCInputDataSchema(BaseModel):
    """Schema for input CSV data"""

    date_column: str = (Field(..., description="Column name of the date variable."),)
    channel_columns: list[str] = (
        Field(min_length=1, description="Column names of the media channel variables."),
    )
    response_column: str = (
        Field(..., description="Column name of the response variable."),
    )
    control_columns: list[str] = (
        Field(..., description="Column names of the control variables."),
    )
    revenue_column: float = Field(description="Revenue column")


class PyMCFitSchema(BaseModel):
    """Schema for PyMC Fit Configuration Dictionary"""

    chains: int = Field(..., description="Number of chains to run.")
    target_accept: float = Field(
        ..., description="Target acceptance rate for the sampler."
    )
    tune: int = Field(..., description="Number of tuning steps.")
    draws: int = Field(..., description="Number of draws to sample.")
    init: str = Field(..., description="Initialization method.")


class PyMCMMMSchema(BaseModel):
    """Schema for PyMC Config Dictionary"""

    date_column: str = (Field(..., description="Column name of the date variable."),)
    channel_columns: list[str] = (
        Field(min_length=1, description="Column names of the media channel variables."),
    )
    adstock: InstanceOf[AdstockTransformation] = (
        Field(..., description="Type of adstock transformation to apply."),
    )
    saturation: InstanceOf[SaturationTransformation] = (
        Field(..., description="Type of saturation transformation to apply."),
    )
    time_varying_intercept: bool = (
        Field(False, description="Whether to consider time-varying intercept."),
    )
    time_varying_media: bool = (
        Field(
            False, description="Whether to consider time-varying media contributions."
        ),
    )
    model_config: dict | None = (Field(None, description="Model configuration."),)
    sampler_config: dict | None = (Field(None, description="Sampler configuration."),)
    validate_data: bool = (
        Field(True, description="Whether to validate the data before fitting to model"),
    )
    control_columns: (
        Annotated[
            list[str],
            Field(
                min_length=1,
                description="Column names of control variables to be added as additional regressors",
            ),
        ]
        | None
    ) = (None,)
    yearly_seasonality: (
        Annotated[
            int,
            Field(
                gt=0, description="Number of Fourier modes to model yearly seasonality."
            ),
        ]
        | None
    ) = (None,)
    adstock_first: bool = (Field(True, description="Whether to apply adstock first."),)
    dag: str | None = (
        Field(
            None,
            description="Optional DAG provided as a string Dot format for causal identification.",
        ),
    )
    treatment_nodes: list[str] | tuple[str] | None = (
        Field(
            None,
            description="Column names of the variables of interest to identify causal effects on outcome.",
        ),
    )
    outcome_node: str | None = (
        Field(None, description="Name of the outcome variable."),
    )

    @validator("channel_columns")
    def validate_channel_columns(cls, v):
        if v is not None and not v:
            raise ValueError("channel_columns must not be empty")
        return v

    @validator("adstock")
    def validate_adstock(cls, v):
        if v is not None:
            assert isinstance(v, AdstockTransformation)
        return v

    @validator("saturation")
    def validate_saturation(cls, v):
        if v is not None:
            assert isinstance(v, SaturationTransformation)
        return v

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",  # Allow extra fields not defined in schema
    }


class EvalConfigSchema(PyMCMMMSchema, PyMCFitSchema):
    target_column: str = Field(..., description="Name of the target column.")
    model_config: PyMCMMMSchema = Field(..., description="Model configuration.")
    fit_config: PyMCFitSchema = Field(..., description="Fit configuration.")
