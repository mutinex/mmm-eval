"""Data loading utilities for MMM evaluation."""

from pathlib import Path
from typing import Any, Union

import pandas as pd

from mmm_eval.data.constants import DataLoaderConstants


class DataLoader:
    """
    Simple data loader for MMM evaluation.
    
    Takes a data path and loads the data.
    """

    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize data loader with data path.
        
        Args:
            data_path: Path to the data file (CSV, Parquet, etc.)
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def load(self) -> pd.DataFrame:
        """
        Load data from the specified path.
        
        Returns:
            Loaded DataFrame

        """
        ext = self.data_path.suffix.lower().lstrip(".")
        if ext not in DataLoaderConstants.ValidDataExtensions.all():
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        if ext == DataLoaderConstants.ValidDataExtensions.CSV:
            return self._load_csv()
        elif ext == DataLoaderConstants.ValidDataExtensions.PARQUET:
            return self._load_parquet()
    
    def _load_csv(self) -> pd.DataFrame:
        """Load CSV data."""
        return pd.read_csv(self.data_path)
    
    def _load_parquet(self) -> pd.DataFrame:
        """Load Parquet data."""
        return pd.read_parquet(self.data_path)
