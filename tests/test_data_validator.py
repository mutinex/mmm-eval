"""
Unit tests for DataValidator.
"""

import pytest
import pandas as pd
from mmm_eval.data import DataValidator


class TestDataValidator:
    """Test DataValidator functionality."""
    
    def test_valid_data(self):
        """Test validation of valid data."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=25),
            'media_channel': ['facebook'] * 25,
            'media_channel_spend': [1000.0] * 25
        })
        
        validator = DataValidator(require_no_nulls=True, min_data_size=21)
        result = validator.run_validations(df)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        
        validator = DataValidator()
        with pytest.raises(Exception):  # Should raise exception for empty data
            validator.run_validations(df)
    
    def test_insufficient_data_size(self):
        """Test validation with insufficient data size."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'media_channel': ['facebook'] * 10,
            'media_channel_spend': [1000.0] * 10
        })
        
        validator = DataValidator(min_data_size=21)
        with pytest.raises(Exception):  # Should raise exception for small data
            validator.run_validations(df)
    
    def test_null_values(self):
        """Test validation with null values."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=25),
            'media_channel': ['facebook'] * 25,
            'media_channel_spend': [1000.0] * 25
        })
        df.loc[0, 'media_channel_spend'] = None
        
        validator = DataValidator(require_no_nulls=True)
        with pytest.raises(Exception):  # Should raise exception for nulls
            validator.run_validations(df)
    
    def test_allow_nulls(self):
        """Test validation when nulls are allowed."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=25),
            'media_channel': ['facebook'] * 25,
            'media_channel_spend': [1000.0] * 25
        })
        df.loc[0, 'media_channel_spend'] = None
        
        validator = DataValidator(require_no_nulls=False, min_data_size=10)
        result = validator.run_validations(df)
        
        assert result.is_valid is True  # Should pass when nulls are allowed 