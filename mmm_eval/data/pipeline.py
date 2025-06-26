"""Data pipeline for MMM evaluation."""

import pandas as pd

from .constants import DataPipelineConstants
from .processor import DataProcessor
from .validation import DataValidator


class DataPipeline:
    """Data pipeline that orchestrates loading, processing, and validation.

    Provides a simple interface to go from raw data file to validated DataFrame.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        control_columns: list[str] | None,
        channel_columns: list[str],
        date_column: str,
        response_column: str,
        revenue_column: str,
        min_number_observations: int = DataPipelineConstants.MIN_NUMBER_OBSERVATIONS,
    ):
        """Initialize data pipeline.

        Args:
            data: DataFrame containing the data
            control_columns: List of control columns
            channel_columns: List of channel columns
            date_column: Name of the date column
            response_column: Name of the response column
            revenue_column: Name of the revenue column
            min_number_observations: Minimum required number of observations

        """
        # Initialize components
        self.data = data
        self.processor = DataProcessor(
            date_column=date_column,
            response_column=response_column,
            revenue_column=revenue_column,
            control_columns=control_columns,
            channel_columns=channel_columns,
        )
        self.validator = DataValidator(
            date_column=date_column,
            response_column=response_column,
            revenue_column=revenue_column,
            control_columns=control_columns,
            min_number_observations=min_number_observations,
        )

    def run(self) -> pd.DataFrame:
        """Run the complete data pipeline: process → validate.

        Returns
            Validated and processed DataFrame

        Raises
            Various exceptions processing or validation steps

        """
        processed_df = self.processor.process(self.data)

        self.validator.run_validations(processed_df)

        return processed_df
