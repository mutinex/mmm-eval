"""
Data processing utilities for MMM evaluation.
"""

from typing import List, Optional
import pandas as pd

from mmm_eval.data.exceptions import InvalidDateFormatError, DataValidationError
from mmm_eval.data.constants import InputDataframeConstants


class DataProcessor:
    """
    Simple data processor for MMM evaluation.
    
    Handles data transformations like datetime casting, column renaming, etc.
    """
    
    def __init__(
        self,
        parse_dates: bool = True,
        date_column: str = InputDataframeConstants.DATE_COL
    ):
        """
        Initialize data processor.
        
        Args:
            parse_dates: Whether to parse date columns
            date_columns: List of date column names to parse (defaults to ['date'])
        """
        self.parse_dates = parse_dates
        self.date_column = date_column
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the DataFrame with configured transformations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        processed_df = df.copy()
        
        # Parse date columns
        if self.parse_dates:
            processed_df = self._parse_date_columns(processed_df, self.date_column)
        
        # Rename date column
        processed_df = self._rename_date_column(processed_df, self.date_column)
        
        return processed_df
    
    def _parse_date_columns(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Parse date columns to datetime."""
        if date_column not in df.columns:
            raise DataValidationError(f"Date column '{date_column}' not found in DataFrame")
        
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        except Exception as e:
            raise InvalidDateFormatError(f"Failed to parse date column '{date_column}': {e}")
        
        return df
    
    def _rename_date_column(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Rename date column to 'date'."""
        df = df.rename(columns={date_column: InputDataframeConstants.DATE_COL})
        return df
