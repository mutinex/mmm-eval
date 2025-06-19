"""
Data validation for MMM evaluation.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import pandera.pandas as pa

from .schemas import ValidatedDataSchema
from .exceptions import DataValidationError, EmptyDataFrameError, ValidationError
from .constants import DataPipelineConstants

class DataValidator:
    """
    Validator for MMM data with configurable validation rules.
    """
    
    def __init__(
        self,
        min_data_size: int = DataPipelineConstants.MIN_DATA_SIZE
    ):
        """
        Initialize validator with validation rules.
        
        Args:
            min_data_size: Minimum required data size for time series CV
        """
        self.min_data_size = min_data_size
    
    def run_validations(self, df: pd.DataFrame) -> None:
        """
        Run all validations on the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validation result with all errors and warnings
        """
        
        # Run each validation in order
        self._validate_not_empty(df)
        self._validate_no_nulls(df)
        self._validate_schema(df)
        self._validate_data_size(df)

    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Check if DataFrame matches the schema."""
        try:
            ValidatedDataSchema.validate(df)
        except pa.errors.SchemaErrors as e:
            raise DataValidationError(f"DataFrame does not match the schema: {e}")
    
    def _validate_not_empty(self, df: pd.DataFrame) -> None:
        """Check if DataFrame is empty."""
        if df.empty:
            raise EmptyDataFrameError("DataFrame is empty")
    
    def _validate_data_size(self, df: pd.DataFrame) -> None:
        """Check minimum data size."""
        if len(df) < self.min_data_size:
            raise DataValidationError(f"Data has {len(df)} rows, but time series CV requires at least {self.min_data_size} rows")
    
    def _validate_no_nulls(self, df: pd.DataFrame) -> None:
        """Check for null values."""
        if df.isnull().any().any():
            raise DataValidationError("Found null values in DataFrame")

