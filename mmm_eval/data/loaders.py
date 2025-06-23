"""Data loading utilities for MMM evaluation."""

from pathlib import Path
from typing import Any

import pandas as pd


class DataLoader:
    """Base data loader class for MMM evaluation.

    Provides utilities for loading and basic validation of MMM data.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize data loader.

        Args:
            config: Configuration for data loading

        """
        self.config = config or {}
        self.required_columns = config.get("required_columns", []) if config else []
        self.date_column = config.get("date_column", "date") if config else "date"
        self.kpi_column = config.get("kpi_column", "kpi") if config else "kpi"

    def load(self, source: Path) -> pd.DataFrame:
        """Load data from various sources.

        Args:
            source: Data source (file path)

        Returns:
            Loaded DataFrame

        """
        if source.suffix.lower() == ".csv":
            data = load_csv(source)
        else:
            raise ValueError(f"Unsupported file format: {source}")

        return self._validate_data(data)

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate MMM data for basic quality checks.

        Args:
            data: Input DataFrame

        Returns:
            Validated DataFrame

        Raises:
            ValueError: If validation fails

        """
        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check if KPI column exists
        if self.kpi_column not in data.columns:
            raise ValueError(f"KPI column '{self.kpi_column}' not found in data")

        # Check for basic data quality
        if data.empty:
            raise ValueError("Data is empty")

        # Check for null values in KPI
        if data[self.kpi_column].isnull().sum() == len(data):
            raise ValueError(f"All values in KPI column '{self.kpi_column}' are null")

        return data


def load_csv(
    file_path: str | Path,
    date_column: str | None = None,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load data from CSV file.

    Args:
        file_path: Path to CSV file
        date_column: Name of date column to parse
        parse_dates: Whether to parse date columns

    Returns:
        Loaded DataFrame

    """
    # Set default parameters for MMM data
    default_kwargs = {
        "index_col": None,
        "encoding": "utf-8",
    }

    # Load data
    data = pd.read_csv(file_path, **default_kwargs)

    # Parse dates if specified
    if parse_dates and date_column and date_column in data.columns:
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.sort_values(date_column).reset_index(drop=True)

    return data
