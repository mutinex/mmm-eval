"""
Data loading utilities for MMM evaluation.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
from mmm_eval.configs.utils import validate_path

def load_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    data_path = validate_path(data_path)
    if not data_path.suffix.lower() == ".csv":
        raise ValueError(f"Invalid data path: {data_path}. Must be a CSV file.")
    return pd.read_csv(data_path)



class DataLoader:
    """
    Base data loader class for MMM evaluation.

    Provides utilities for loading and basic validation of MMM data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data loader.

        Args:
            config: Configuration for data loading
        """
        self.config = config or {}
        self.required_columns = config.get("required_columns", []) if config else []
        self.date_column = config.get("date_column", "date") if config else "date"
        self.kpi_column = config.get("kpi_column", "kpi") if config else "kpi"

    def load(self, source: Union[str, Path, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        Load data from various sources.

        Args:
            source: Data source (file path, DataFrame, etc.)
            **kwargs: Additional loading parameters

        Returns:
            Loaded DataFrame
        """
        if isinstance(source, pd.DataFrame):
            data = source.copy()
        elif isinstance(source, (str, Path)):
            if str(source).endswith(".csv"):
                data = load_csv(source, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {source}")
        else:
            raise ValueError(f"Unsupported data source type: {type(source)}")

        return self._validate_data(data)

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Basic validation of MMM data.

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
        if data[self.kpi_column].isnull().all():
            raise ValueError(f"All values in KPI column '{self.kpi_column}' are null")

        return data


def load_csv(
    file_path: Union[str, Path],
    date_column: Optional[str] = None,
    parse_dates: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Load data from CSV file.

    Args:
        file_path: Path to CSV file
        date_column: Name of date column to parse
        parse_dates: Whether to parse date columns
        **kwargs: Additional pandas.read_csv parameters

    Returns:
        Loaded DataFrame
    """
    # Set default parameters for MMM data
    default_kwargs = {
        "index_col": None,
        "encoding": "utf-8",
    }
    default_kwargs.update(kwargs)

    # Load data
    data = pd.read_csv(file_path, **default_kwargs)

    # Parse dates if specified
    if parse_dates and date_column and date_column in data.columns:
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.sort_values(date_column).reset_index(drop=True)

    return data


def load_from_database(
    connection_string: str, query: str, date_column: Optional[str] = None, **kwargs
) -> pd.DataFrame:
    """
    Load data from database.

    Args:
        connection_string: Database connection string
        query: SQL query to execute
        date_column: Name of date column to parse
        **kwargs: Additional pandas.read_sql parameters

    Returns:
        Loaded DataFrame

    Note:
        Requires appropriate database drivers to be installed.
    """
    try:
        import sqlalchemy
    except ImportError:
        raise ImportError(
            "sqlalchemy is required for database loading. Install with: pip install sqlalchemy"
        )

    # Create engine
    engine = sqlalchemy.create_engine(connection_string)

    # Load data
    data = pd.read_sql(query, engine, **kwargs)

    # Parse dates if specified
    if date_column and date_column in data.columns:
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.sort_values(date_column).reset_index(drop=True)

    return data
