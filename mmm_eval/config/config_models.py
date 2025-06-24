"""Configuration models for the MMM evaluator."""

from datetime import datetime

from pydantic import BaseModel, Field


class EvaluatorConfig(BaseModel):
    """Configuration specific to the evaluator/data pipeline.

    This contains the column mappings and data configuration that the evaluator
    needs to process input data, separate from framework-specific configuration.
    """

    # model_config = ConfigDict(
    #     extra="ignore",  # Explicitly ignore extra fields
    #     validate_assignment=True,  # Validate when attributes are set
    # )

    date_column: str | datetime = Field(
        description="Column name for date variable in the input data. Can be a string or a datetime object."
    )
    response_column: str = Field(description="Column name for response variable (target) in the input data")
    revenue_column: str = Field(description="Column name for revenue variable in the input data")
