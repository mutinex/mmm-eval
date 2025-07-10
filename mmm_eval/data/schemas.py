"""Pydantic schemas for MMM data validation."""

import pandera.pandas as pa
from pandera.typing import Series


class ValidatedDataSchema(pa.DataFrameModel):
    """Schema for MMM data validation.

    Defines the bare minimum columns for MMM evaluation.
    """

    # Required columns
    response: Series[pa.dtypes.Float64] = pa.Field(nullable=False)
    revenue: Series[pa.dtypes.Float64] = pa.Field(nullable=False)

    class Config:
        """Config for the schema."""

        coerce = True
