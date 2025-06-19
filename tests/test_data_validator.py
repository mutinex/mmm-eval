"""
Unit tests for DataValidator.
"""

import pytest
import pandas as pd
import pandera.pandas as pa
from mmm_eval.data import DataValidator
from mmm_eval.data.exceptions import DataValidationError, EmptyDataFrameError
from mmm_eval.data.constants import InputDataframeConstants


class TestDataValidator:
    """Test DataValidator functionality."""
    
    def test_valid_data(self):
        """Test validation of valid data."""
        df = pd.DataFrame({
            InputDataframeConstants.DATE_COL: pd.date_range('2023-01-01', periods=25),
            InputDataframeConstants.MEDIA_CHANNEL_COL: ['facebook'] * 25,
            InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL: [1000.0] * 25
        })
        
        validator = DataValidator(min_data_size=21)
        validator.run_validations(df)  # Should not raise any exceptions
    
    def test_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        
        validator = DataValidator()
        with pytest.raises(EmptyDataFrameError):  # Should raise EmptyDataFrameError for empty data
            validator.run_validations(df)
    
    def test_insufficient_data_size(self):
        """Test validation with insufficient data size."""
        df = pd.DataFrame({
            InputDataframeConstants.DATE_COL: pd.date_range('2023-01-01', periods=10),
            InputDataframeConstants.MEDIA_CHANNEL_COL: ['facebook'] * 10,
            InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL: [1000.0] * 10
        })
        
        validator = DataValidator(min_data_size=21)
        with pytest.raises(DataValidationError):  # Should raise exception for small data
            validator.run_validations(df)
    
    def test_null_values(self):
        """Test validation with null values."""
        df = pd.DataFrame({
            InputDataframeConstants.DATE_COL: pd.date_range('2023-01-01', periods=25),
            InputDataframeConstants.MEDIA_CHANNEL_COL: ['facebook'] * 25,
            InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL: [1000.0] * 25
        })
        df.loc[0, InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL] = None
        
        validator = DataValidator()
        with pytest.raises(DataValidationError):  # Should raise DataValidationError for nulls
            validator.run_validations(df)