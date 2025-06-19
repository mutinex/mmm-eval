"""
Unit tests for DataProcessor.
"""

import pytest
import pandas as pd
from mmm_eval.data import DataProcessor
from mmm_eval.data.exceptions import InvalidDateFormatError, DataValidationError
from mmm_eval.data.constants import InputDataframeConstants


class TestDataProcessor:
    """Test DataProcessor functionality."""
    
    def _get_test_df(self):
        """Helper method to create test DataFrame."""
        return pd.DataFrame({
            InputDataframeConstants.DATE_COL: ['2023-01-01', '2023-01-02'],
            InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL: [1000.0, 1500.0]
        })

    def test_parse_dates(self):
        """Test date parsing functionality."""
        df = self._get_test_df()
        
        processor = DataProcessor()
        result = processor.process(df)
        
        assert pd.api.types.is_datetime64_any_dtype(result[InputDataframeConstants.DATE_COL])
        assert result[InputDataframeConstants.DATE_COL].iloc[0] == pd.Timestamp('2023-01-01')
    
    def test_custom_date_column(self):
        """Test processing with custom date column name."""
        df = self._get_test_df()
        df = df.rename(columns={InputDataframeConstants.DATE_COL: 'Date'})
        
        processor = DataProcessor(date_column='Date')
        result = processor.process(df)
        
        assert InputDataframeConstants.DATE_COL in result.columns  # Should be renamed
        assert pd.api.types.is_datetime64_any_dtype(result[InputDataframeConstants.DATE_COL])
    
    def test_missing_date_column(self):
        """Test error when date column is missing."""
        df = pd.DataFrame({
            InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL: [1000.0, 1500.0]
        })
        
        processor = DataProcessor()
        with pytest.raises(DataValidationError):  # Should raise DataValidationError for missing column
            processor.process(df)
    
    def test_invalid_date_format(self):
        """Test error when date format cannot be parsed."""
        df = pd.DataFrame({
            InputDataframeConstants.DATE_COL: ['2023-01-01', 'not-a-date', '2023-01-03'],
            InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL: [1000.0, 1500.0, 2000.0]
        })
        
        processor = DataProcessor()
        with pytest.raises(InvalidDateFormatError):  # Should raise InvalidDateFormatError for unparseable dates
            processor.process(df)