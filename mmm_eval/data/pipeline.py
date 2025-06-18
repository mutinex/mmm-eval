"""
Data pipeline for MMM evaluation.
"""

from typing import Union, Optional
import pandas as pd
from pathlib import Path

from .loaders import DataLoader
from .processor import DataProcessor
from .validation import DataValidator


class DataPipeline:
    """
    Data pipeline that orchestrates loading, processing, and validation.
    
    Provides a simple interface to go from raw data file to validated DataFrame.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        parse_dates: bool = True,
        date_column: str = "date",
        require_no_nulls: bool = True,
        min_data_size: int = 21
    ):
        """
        Initialize data pipeline.
        
        Args:
            data_path: Path to the data file
            parse_dates: Whether to parse date columns
            date_column: Name of the date column
            require_no_nulls: Whether to require no null values
            min_data_size: Minimum required data size
        """
        self.data_path = data_path
        self.parse_dates = parse_dates
        self.date_column = date_column
        self.require_no_nulls = require_no_nulls
        self.min_data_size = min_data_size
        
        # Initialize components
        self.loader = DataLoader(data_path)
        self.processor = DataProcessor(
            parse_dates=parse_dates,
            date_column=date_column
        )
        self.validator = DataValidator(
            require_no_nulls=require_no_nulls,
            min_data_size=min_data_size
        )
    
    def run(self) -> pd.DataFrame:
        """
        Run the complete data pipeline: load → process → validate.
        
        Returns:
            Validated and processed DataFrame
            
        Raises:
            Various exceptions from loading, processing, or validation steps
        """
        # Step 1: Load data
        raw_df = self.loader.load()
        
        # Step 2: Process data
        processed_df = self.processor.process(raw_df)
        
        # Step 3: Validate data
        self.validator.run_validations(processed_df)
        
        return processed_df