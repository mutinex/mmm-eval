"""Data validation for MMM evaluation."""

import pandas as pd
import pandera.pandas as pa

from .constants import DataPipelineConstants
from .exceptions import DataValidationError, EmptyDataFrameError
from .schemas import ValidatedDataSchema


class DataValidator:
    """Validator for MMM data with configurable validation rules."""

    def __init__(self, min_number_observations: int = DataPipelineConstants.MIN_NUMBER_OBSERVATIONS):
        """Initialize validator with validation rules.

        Args:
            min_number_observations: Minimum required number of observations for time series CV

        """
        self.min_number_observations = min_number_observations

    def run_validations(self, df: pd.DataFrame) -> None:
        """Run all validations on the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Validation result with all errors and warnings

        """
        # Run each validation in order
        self._validate_not_empty(df)
        self._validate_schema(df)
        self._validate_data_size(df)

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Check if DataFrame matches the schema."""
        try:
            ValidatedDataSchema.validate(df)
        except pa.errors.SchemaErrors as e:
            raise DataValidationError(f"DataFrame does not match the schema: {str(e)}") from e

    def _validate_not_empty(self, df: pd.DataFrame) -> None:
        """Check if DataFrame is empty."""
        if df.empty:
            raise EmptyDataFrameError("DataFrame is empty")

    def _validate_data_size(self, df: pd.DataFrame) -> None:
        """Check minimum data size."""
        if len(df) < self.min_number_observations:
            raise DataValidationError(
                f"Data has {len(df)} rows, but time series CV requires at least {self.min_number_observations} rows"
            )
