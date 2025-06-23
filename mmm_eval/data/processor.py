"""Data processing utilities for MMM evaluation."""

import pandas as pd

from mmm_eval.data.constants import InputDataframeConstants
from mmm_eval.data.exceptions import (
    InvalidDateFormatError,
    MissingRequiredColumnsError,
)


class DataProcessor:
    """Simple data processor for MMM evaluation.

    Handles data transformations like datetime casting, column renaming, etc.
    """

    def __init__(
        self,
        date_column: str = InputDataframeConstants.DATE_COL,
        response_column: str = InputDataframeConstants.RESPONSE_COL,
        revenue_column: str = InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL,
    ):
        """Initialize data processor.

        Args:
            date_column: Name of the date column to parse and rename
            response_column: Name of the response column to parse and rename
            revenue_column: Name of the revenue column to parse and rename

        """
        self.date_column = date_column
        self.response_column = response_column
        self.revenue_column = revenue_column

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the DataFrame with configured transformations.

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame

        Raises:
            MissingRequiredColumnsError: If the required columns are not present.
            InvalidDateFormatError: If the date column cannot be parsed.

        """
        processed_df = df.copy()

        # Validate that all required columns exist
        self._validate_required_columns_present(
            df=processed_df,
            date_column=self.date_column,
            response_column=self.response_column,
            revenue_column=self.revenue_column,
        )

        # Parse date columns
        processed_df = self._parse_date_columns(processed_df, self.date_column)

        # Rename required columns
        processed_df = self._rename_required_columns(
            df=processed_df,
            date_column=self.date_column,
            response_column=self.response_column,
            revenue_column=self.revenue_column,
        )

        return processed_df

    def _validate_required_columns_present(
        self,
        df: pd.DataFrame,
        date_column: str,
        response_column: str,
        revenue_column: str,
    ) -> None:
        """Validate that all required columns are present for processing.

        Args:
            df: Input DataFrame
            date_column: Name of the date column
            response_column: Name of the response column
            revenue_column: Name of the revenue column

        Returns:
            None

        """
        if date_column not in df.columns:
            raise MissingRequiredColumnsError(f"Date column '{date_column}' required but not found in DataFrame")
        if response_column not in df.columns:
            raise MissingRequiredColumnsError(
                f"Response column '{response_column}' required but not found in DataFrame"
            )
        if revenue_column not in df.columns:
            raise MissingRequiredColumnsError(f"Revenue column '{revenue_column}' required but not found in DataFrame")

    def _parse_date_columns(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Parse date columns to datetime.

        Args:
            df: Input DataFrame
            date_column: Name of the date column

        Returns:
            DataFrame with parsed date columns

        Raises:
            InvalidDateFormatError: If the date column cannot be parsed.

        """
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors="raise")
        except Exception as e:
            raise InvalidDateFormatError(f"Failed to parse date column '{date_column}': {e}") from e

        return df

    def _rename_required_columns(
        self,
        df: pd.DataFrame,
        date_column: str,
        response_column: str,
        revenue_column: str,
    ) -> pd.DataFrame:
        """Rename required columns to the constants.

        Args:
            df: Input DataFrame
            date_column: Name of the date column
            response_column: Name of the response column
            revenue_column: Name of the revenue column

        Returns:
            DataFrame with renamed columns

        Raises:
            MissingRequiredColumnsError: If the required columns are not present.

        """
        df = df.rename(
            columns={
                date_column: InputDataframeConstants.DATE_COL,
                response_column: InputDataframeConstants.RESPONSE_COL,
                revenue_column: InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL,
            }
        )
        return df
