"""Data validation for MMM evaluation."""

import logging

import pandas as pd
import pandera.pandas as pa

from .constants import DataPipelineConstants
from .exceptions import DataValidationError, EmptyDataFrameError
from .schemas import ValidatedDataSchema
from mmm_eval.core.validation_tests_models import FrameworkNames

logger = logging.getLogger(__name__)


class DataValidator:
    """Validator for MMM data with configurable validation rules."""

    def __init__(
        self,
        framework: FrameworkNames,
        date_column: str,
        response_column: str,
        revenue_column: str,
        control_columns: list[str] | None,
        min_number_observations: int = DataPipelineConstants.MIN_NUMBER_OBSERVATIONS,
    ):
        """Initialize validator with validation rules.

        Args:
            framework: a supported framework, one of `FrameworkNames`
            date_column: Name of the date column
            response_column: Name of the response column
            revenue_column: Name of the revenue column
            control_columns: List of control columns
            min_number_observations: Minimum required number of observations for time series CV

        """
        self.framework = framework
        self.date_column = date_column
        self.response_column = response_column
        self.revenue_column = revenue_column
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
        self._validate_response_and_revenue_columns_xor_zeroes(df)

        # TODO: only run for PyMC adapter
        if self.control_columns and self.framework == FrameworkNames.PYMC_MARKETING:
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

    def _validate_response_and_revenue_columns_xor_zeroes(self, df: pd.DataFrame) -> None:
        """Ensure that there are no cases where exactly one of response_column and revenue_column is non-zero."""
        if self.response_column != self.revenue_column:
            response_zero = df[self.response_column] == 0
            revenue_zero = df[self.revenue_column] == 0

            # XOR condition: exactly one is zero (one is zero AND the other is not zero)
            xor_condition = (response_zero & ~revenue_zero) | (~response_zero & revenue_zero)
            xor_entries = df[xor_condition]

            if not xor_entries.empty:
                raise DataValidationError(
                    f"Found {len(xor_entries)} entries where exactly one of "
                    f"'{self.response_column}' and '{self.revenue_column}' is zero. "
                    f"Problematic dates: {xor_entries[self.date_column].unique()}"
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
