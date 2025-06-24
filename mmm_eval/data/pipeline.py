"""Data pipeline for MMM evaluation."""

from pathlib import Path

import pandas as pd

from .constants import DataPipelineConstants, InputDataframeConstants
from .loaders import DataLoader
from .processor import DataProcessor
from .validation import DataValidator


class DataPipeline:
    """Data pipeline that orchestrates loading, processing, and validation.

    Provides a simple interface to go from raw data file to validated DataFrame.
    """

    def __init__(
        self,
        data_path: str | Path,
        control_columns: list[str] | None,
        channel_columns: list[str],
        date_column: str = InputDataframeConstants.DATE_COL,
        response_column: str = InputDataframeConstants.RESPONSE_COL,
        revenue_column: str = InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL,
        min_number_observations: int = DataPipelineConstants.MIN_NUMBER_OBSERVATIONS,
    ):
        """Initialize data pipeline.

        Args:
            data_path: Path to the data file
            control_columns: List of control columns
            channel_columns: List of channel columns
            date_column: Name of the date column
            response_column: Name of the response column
            revenue_column: Name of the revenue column
            min_number_observations: Minimum required number of observations

        """
        # Initialize components
        self.loader = DataLoader(data_path)
        self.processor = DataProcessor(
            date_column=date_column,
            response_column=response_column,
            revenue_column=revenue_column,
            control_columns=control_columns,
            channel_columns=channel_columns,
        )
        self.validator = DataValidator(
            control_columns=control_columns,
            min_number_observations=min_number_observations,
            channel_spend_columns=channel_columns,
        )

    def run(self) -> pd.DataFrame:
        """Run the complete data pipeline: load → process → validate.

        Returns
            Validated and processed DataFrame

        Raises
            Various exceptions from loading, processing, or validation steps

        """
        # Step 1: Load data
        raw_df = self.loader.load()

        # Step 2: Process data
        processed_df = self.processor.process(raw_df)

        # Step 3: Validate data
        self.validator.run_validations(processed_df)

        return processed_df
