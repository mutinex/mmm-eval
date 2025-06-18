"""
Unit tests for DataLoader.
"""

import pytest
import pandas as pd
from pathlib import Path
from mmm_eval.data import DataLoader


class TestDataLoader:
    """Test DataLoader functionality."""
    
    def test_load_csv(self, tmp_path):
        """Test loading CSV data."""
        # Create test CSV
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'value': [1, 2]
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        # Load data
        loader = DataLoader(csv_path)
        result = loader.load()
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
        assert list(result.columns) == ['date', 'value']
    
    def test_load_parquet(self, tmp_path):
        """Test loading Parquet data."""
        # Create test Parquet
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'value': [1, 2]
        })
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path, index=False)
        
        # Load data
        loader = DataLoader(parquet_path)
        result = loader.load()
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
    
    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            DataLoader("nonexistent.csv")
    
    def test_unsupported_format(self, tmp_path):
        """Test error for unsupported file format."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("test")
        
        loader = DataLoader(txt_path)
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load() 