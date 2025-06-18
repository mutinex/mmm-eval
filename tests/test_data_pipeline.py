"""
Unit tests for DataPipeline.
"""

import pytest
import pandas as pd
from pathlib import Path
from mmm_eval.data import DataPipeline


class TestDataPipeline:
    """Test DataPipeline functionality."""
    
    def test_complete_pipeline(self, tmp_path):
        """Test complete pipeline with valid data."""
        # Create test CSV
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=25).strftime('%Y-%m-%d'),
            'media_channel': ['facebook'] * 25,
            'media_channel_spend': [1000.0] * 25
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        # Run pipeline
        pipeline = DataPipeline(
            data_path=csv_path,
            parse_dates=True,
            require_no_nulls=True,
            min_data_size=21
        )
        result = pipeline.run()
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (25, 3)
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
    
    def test_pipeline_with_custom_settings(self, tmp_path):
        """Test pipeline with custom settings."""
        # Create test CSV
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=25).strftime('%Y-%m-%d'),
            'media_channel': ['facebook'] * 25,
            'media_channel_spend': [1000.0] * 25
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        # Run pipeline with custom date column
        pipeline = DataPipeline(
            data_path=csv_path,
            parse_dates=True,
            date_column='Date',
            require_no_nulls=False,
            min_data_size=10
        )
        result = pipeline.run()
        
        assert isinstance(result, pd.DataFrame)
        assert 'date' in result.columns  # Should be renamed
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
    
    def test_pipeline_fails_with_invalid_data(self, tmp_path):
        """Test pipeline fails with invalid data."""
        # Create test CSV with insufficient data
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10).strftime('%Y-%m-%d'),
            'media_channel': ['facebook'] * 10,
            'media_channel_spend': [1000.0] * 10
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        # Run pipeline with strict requirements
        pipeline = DataPipeline(
            data_path=csv_path,
            parse_dates=True,
            require_no_nulls=True,
            min_data_size=21
        )
        
        with pytest.raises(Exception):  # Should fail due to insufficient data
            pipeline.run()
    
    def test_pipeline_components_initialized(self, tmp_path):
        """Test that pipeline components are properly initialized."""
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({'test': [1]}).to_csv(csv_path, index=False)
        
        pipeline = DataPipeline(
            data_path=csv_path,
            parse_dates=True,
            date_column='custom_date',
            require_no_nulls=False,
            min_data_size=10
        )
        
        # Check components are initialized
        assert hasattr(pipeline, 'loader')
        assert hasattr(pipeline, 'processor')
        assert hasattr(pipeline, 'validator')
        
        # Check settings are passed through
        assert pipeline.processor.date_column == 'custom_date'
        assert pipeline.validator.require_no_nulls is False
        assert pipeline.validator.min_data_size == 10 