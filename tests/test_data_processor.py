"""
Unit tests for DataProcessor.
"""

import pytest
import pandas as pd
from mmm_eval.data import DataProcessor


class TestDataProcessor:
    """Test DataProcessor functionality."""
    
    def test_parse_dates(self):
        """Test date parsing functionality."""
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'value': [1, 2]
        })
        
        processor = DataProcessor(parse_dates=True)
        result = processor.process(df)
        
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
        assert result['date'].iloc[0] == pd.Timestamp('2023-01-01')
    
    def test_custom_date_column(self):
        """Test processing with custom date column name."""
        df = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02'],
            'value': [1, 2]
        })
        
        processor = DataProcessor(parse_dates=True, date_column='Date')
        result = processor.process(df)
        
        assert 'date' in result.columns  # Should be renamed
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
    
    def test_no_date_parsing(self):
        """Test when date parsing is disabled."""
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'value': [1, 2]
        })
        
        processor = DataProcessor(parse_dates=False)
        result = processor.process(df)
        
        assert result['date'].dtype == 'object'  # Should remain string
    
    def test_missing_date_column(self):
        """Test error when date column is missing."""
        df = pd.DataFrame({
            'value': [1, 2]
        })
        
        processor = DataProcessor(parse_dates=True)
        with pytest.raises(Exception):  # Should raise some exception
            processor.process(df) 