"""
Unit tests for DataPipeline.
"""

import pytest
import pandas as pd
from pathlib import Path
from mmm_eval.data import DataPipeline
from mmm_eval.data.exceptions import DataValidationError
from mmm_eval.data.constants import InputDataframeConstants, DataLoaderConstants

class TestDataPipeline:
    """Test DataPipeline functionality."""

    def _get_test_df(self):
        """Helper method to create test DataFrame."""
        return pd.DataFrame({
            InputDataframeConstants.DATE_COL: pd.date_range('2023-01-01', periods=25).strftime('%Y-%m-%d'),
            InputDataframeConstants.MEDIA_CHANNEL_COL: ['facebook'] * 25,
            InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL: [1000.0] * 25
        })
    
    def test_complete_pipeline(self, tmp_path):
        """Test complete pipeline with valid data."""
        # Create test CSV
        df = self._get_test_df()
        csv_path = tmp_path / f"test.{DataLoaderConstants.ValidDataExtensions.CSV}"
        df.to_csv(csv_path, index=False)
        
        # Run pipeline
        pipeline = DataPipeline(
            data_path=csv_path,
            min_data_size=21
        )
        result = pipeline.run()
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (25, 3)
        assert pd.api.types.is_datetime64_any_dtype(result[InputDataframeConstants.DATE_COL])
    
    def test_pipeline_with_custom_settings(self, tmp_path):
        """Test pipeline with custom settings."""
        # Create test CSV
        df = self._get_test_df()
        csv_path = tmp_path / f"test.{DataLoaderConstants.ValidDataExtensions.CSV}"
        df.to_csv(csv_path, index=False)
        
        # Run pipeline with custom date column
        pipeline = DataPipeline(
            data_path=csv_path,
            date_column=InputDataframeConstants.DATE_COL,
            min_data_size=10
        )
        result = pipeline.run()
        
        assert isinstance(result, pd.DataFrame)
        assert InputDataframeConstants.DATE_COL in result.columns  # Should be renamed
        assert pd.api.types.is_datetime64_any_dtype(result[InputDataframeConstants.DATE_COL])
    
    def test_pipeline_fails_with_invalid_data(self, tmp_path):
        """Test pipeline fails with invalid data."""
        # Create test CSV with insufficient data
        df = self._get_test_df()
        df = df.head(10)
        csv_path = tmp_path / f"test.{DataLoaderConstants.ValidDataExtensions.CSV}"
        df.to_csv(csv_path, index=False)
        
        # Run pipeline with strict requirements
        pipeline = DataPipeline(
            data_path=csv_path,
            min_data_size=21
        )
        
        with pytest.raises(DataValidationError):  # Should fail due to insufficient data
            pipeline.run()