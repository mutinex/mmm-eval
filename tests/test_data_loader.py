"""
Unit tests for DataLoader.
"""

import pytest
import pandas as pd
from pathlib import Path
from mmm_eval.data import DataLoader
from mmm_eval.data.constants import InputDataframeConstants, DataLoaderConstants


class TestDataLoader:
    """Test DataLoader functionality."""

    def _get_test_df(self):
        """Helper method to create test DataFrame."""
        return pd.DataFrame({
            InputDataframeConstants.DATE_COL: ['2023-01-01', '2023-01-02'],
            InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL: [1000.0, 1500.0]
        })
    
    def test_load_csv(self, tmp_path):
        """Test loading CSV data."""
        # Create test CSV
        df = self._get_test_df()
        csv_path = tmp_path / f"test.{DataLoaderConstants.ValidDataExtensions.CSV}"
        df.to_csv(csv_path, index=False)
        
        # Load data
        loader = DataLoader(csv_path)
        result = loader.load()
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
        assert list(result.columns) == [InputDataframeConstants.DATE_COL, InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL]
    
    def test_load_parquet(self, tmp_path):
        """Test loading Parquet data."""
        # Create test Parquet
        df = self._get_test_df()
        parquet_path = tmp_path / f"test.{DataLoaderConstants.ValidDataExtensions.PARQUET}"
        df.to_parquet(parquet_path, index=False)
        
        # Load data
        loader = DataLoader(parquet_path)
        result = loader.load()
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
    
    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            DataLoader(f"nonexistent.{DataLoaderConstants.ValidDataExtensions.CSV}")
    
    def test_unsupported_format(self, tmp_path):
        """Test error for unsupported file format."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("test")
        
        loader = DataLoader(txt_path)
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load() 