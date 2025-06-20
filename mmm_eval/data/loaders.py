"""
Data loading utilities for MMM evaluation.
"""

from typing import Dict, Any, Optional, Union
import pandas as pd
from pathlib import Path
import json
from logging import getLogger
from pymc_marketing.mmm import MMM
from mmm_eval.adapters.experimental.schemas import PyMCModelSchema, PyMCFitSchema
from mmm_eval.utils import PyMCConfigRehydrator
from pydantic import BaseModel


class Config:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.is_hydrated = self.check_hydration()

    def check_hydration(self):
        """
        Check if the config is hydrated (not all strings)

        Returns:
            bool: True if the config is hydrated, False otherwise.
        """
        return not all(isinstance(v, str) for v in self.config.values())


class ConfigLoader:
    def __init__(self, model_object: Any):
        self.schema_class = None
        self.model_object = model_object

    def extract_config_from_dict(self) -> dict[str, Any]:
        """
        Load config from dictionary using only the keys defined in the schema.

        Args:
            config (dict): Dictionary (e.g., from model.__dict__).
            schema_class (BaseModel): The Pydantic schema to validate against.

        Returns:
            dict: A schema-validated dictionary populated with values from config.
        """
        if self.schema_class is None:
            raise ValueError("schema_class must be set before calling extract_config_from_dict")
        schema_keys = self.schema_class.model_fields.keys()
        filtered_dict = {k: v for k, v in self.model_object.__dict__.items() if k in schema_keys}
        return self.schema_class(**filtered_dict).model_dump()

    

class PYMCConfig(ConfigLoader):
    def __init__(self, model_object: Any, fit_kwargs: dict[str, Any], target_column: str):
        super().__init__(model_object)
        self.schema_class = PyMCModelSchema
        self.model_config = Config(self.extract_config_from_dict())
        self.fit_kwargs = Config(fit_kwargs)
        self.target_column = target_column


    def save_config_to_json(self, save_path: str, file_name: str):
        config = {
        "model_config": {k: repr(v) for k, v in self.model_config.config.items()},
        "fit_config": {k: repr(v) for k, v in self.fit_kwargs.config.items()},
        "target_column": self.target_column,
    }
        Path(save_path).mkdir(parents=True, exist_ok=True)
        json.dump(config, open(f"{save_path}/{file_name}.json", "w"))
        return self
    
    def load_config_from_json(self, load_path: str, file_name: str):
        config = json.load(open(f"{load_path}/{file_name}.json"))
        self.model_config = Config(self._rehydrate_config(config["model_config"], PyMCModelSchema))
        self.fit_kwargs = Config(self._rehydrate_config(config["fit_config"], PyMCFitSchema))
        self.target_column = config["target_column"]
        return self
    
    def _rehydrate_config(self, config, schema_class: PyMCModelSchema | PyMCFitSchema):
        hydrated_config = PyMCConfigRehydrator(config, schema_class).rehydrate_config()
        return hydrated_config


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
