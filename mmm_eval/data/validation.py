"""Data validation for MMM evaluation."""

import logging

import pandas as pd
import pandera.pandas as pa

from .constants import DataPipelineConstants
from .exceptions import DataValidationError, EmptyDataFrameError
from .schemas import ValidatedDataSchema

logger = logging.getLogger(__name__)


class DataValidator:
    """Validator for MMM data with configurable validation rules."""

    def __init__(
        self,
        control_columns: list[str] | None,
        min_number_observations: int = DataPipelineConstants.MIN_NUMBER_OBSERVATIONS,
    ):
        """Initialize validator with validation rules.

        Args:
            control_columns: List of control columns
            min_number_observations: Minimum required number of observations for time series CV

        """
        self.min_number_observations = min_number_observations
        self.control_columns = control_columns

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

        if self.control_columns:
            self._check_control_variables_between_0_and_1(df=df, cols=self.control_columns)

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

    def _check_control_variables_between_0_and_1(self, df: pd.DataFrame, cols: list[str]) -> None:
        """Check if variables are in the 0-1 range.

        Args:
            df: DataFrame containing the data
            cols: List of columns to check

        """
        data_to_check = df[list(cols)]
        out_of_range_cols = data_to_check.columns[(data_to_check.min() < -1) | (data_to_check.max() > 1)]

        for col in out_of_range_cols:
            col_min = data_to_check[col].min()
            col_max = data_to_check[col].max()
            logger.warning(
                f"Control column '{col}' has values outside [-1, 1]: "
                f"min={col_min:.4f}, max={col_max:.4f}. "
                f"Consider scaling this column to [-1, 1] as per PyMC best practices."
            )
