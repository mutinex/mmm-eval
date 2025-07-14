"""Dataset processor for MMM evaluation frameworks.

TODOs:
- load in externals from pipeline run, use same approach as holidays
- add suffixes to columns based on their dataset provenance
"""

import logging
import numpy as np
import pandas as pd

from mmm_eval.comparison.process import load_and_process_datasets
from mmm_eval.data.constants import InputDataframeConstants

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """Process datasets for different MMM frameworks.
    
    This class wraps the `load_and_process_datasets` function and provides
    framework-specific transformations for PyMC and Meridian.
    """
    
    def __init__(
        self,
        customer_id: str,
        data_version: str,
        holidays_whitelist: list[str],
        holidays_df_path: str,
        node_filter: str | None = None,
    ):
        """Initialize the dataset processor.
        
        Args:
            customer_id: Customer identifier
            data_version: Version of the data to load
            holidays_whitelist: List of holiday variables to include
            holidays_df_path: Path to the holidays DataFrame
            node_filter: Optional filter for specific nodes
        """
        self.customer_id = customer_id
        self.data_version = data_version
        self.holidays_whitelist = holidays_whitelist
        self.holidays_df_path = holidays_df_path
        self.node_filter = node_filter
        
        # Load and process the base dataset
        self._raw_data = load_and_process_datasets(
            customer_id=customer_id,
            data_version=data_version,
            holidays_whitelist=holidays_whitelist,
            holidays_df_path=holidays_df_path,
            node_filter=node_filter,
        )
        
        logger.info(f"Loaded dataset with shape {self._raw_data.shape}")
    
    @classmethod
    def from_raw_data(
        cls,
        raw_data: pd.DataFrame,
        customer_id: str = "test_customer",
        data_version: str = "test_version",
        holidays_whitelist: list[str] | None = None,
        holidays_df_path: str = "test_path",
        node_filter: str | None = None,
    ) -> "DatasetProcessor":
        """Create a DatasetProcessor instance from already loaded raw data.
        
        This method bypasses the `load_and_process_datasets` call, which can be
        useful for testing or when the data is already available.
        
        Args:
            raw_data: Pre-loaded and processed DataFrame
            customer_id: Customer identifier (default: "test_customer")
            data_version: Version of the data (default: "test_version")
            holidays_whitelist: List of holiday variables (default: None)
            holidays_df_path: Path to the holidays DataFrame (default: "test_path")
            node_filter: Optional filter for specific nodes (default: None)
            
        Returns:
            DatasetProcessor instance with the provided raw data
        """
        instance = cls.__new__(cls)
        
        # Set attributes
        instance.customer_id = customer_id
        instance.data_version = data_version
        instance.holidays_whitelist = holidays_whitelist or []
        instance.holidays_df_path = holidays_df_path
        instance.node_filter = node_filter
        
        # Store the raw data directly
        instance._raw_data = raw_data.copy()
        
        logger.info(f"Created DatasetProcessor from raw data with shape {instance._raw_data.shape}")
        
        return instance
    
    def get_pymc_column_map(self) -> dict[str, list[str]]:
        """Categorize columns for PyMC framework.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with keys 'response', 'revenue', 'channel_columns', 'control_columns'
            containing lists of column names for each category
        """
        df = self.raw_data
        all_columns = set(df.columns)
        
        # Response column is "quantity"
        response_columns = ["quantity"] if "quantity" in all_columns else []
        
        # Revenue column is "revenue"
        revenue_columns = ["revenue"] if "revenue" in all_columns else []
        
        # Channel columns have suffix "_brand", "_category", or "_product"
        channel_columns = [
            col for col in all_columns 
            if any(col.endswith(suffix) for suffix in ["_brand", "_category", "_product"])
        ]
        
        # Control columns are all other columns
        excluded_columns = set(response_columns + revenue_columns + channel_columns)
        control_columns = [col for col in all_columns if col not in excluded_columns]
        
        return {
            "response": response_columns,
            "revenue": revenue_columns,
            "channel_columns": channel_columns,
            "control_columns": control_columns,
        }
    
    def get_meridian_column_map(self) -> dict[str, list[str]]:
        """Categorize columns for Meridian framework.
        
        Returns:
            Dictionary with keys 'media_channels', 'control_columns', 'non_media_treatment_columns'
            containing lists of column names for each category
        """
        df = self.raw_data
        all_columns = set(df.columns)
        
        # Media channels are columns suffixed with "_brand", "_category", or "_product"
        media_channels = [
            col for col in all_columns 
            if any(col.endswith(suffix) for suffix in ["_brand", "_category", "_product"])
        ]
        
        # All binary columns are control_columns
        binary_columns = []
        for col in all_columns:
            if col in df.columns:
                unique_values = df[col].dropna().unique().tolist()
                # Handle both integer/boolean and float binary columns
                if len(unique_values) <= 2 and (
                    set(unique_values).issubset({0, 1, True, False}) or
                    set(unique_values).issubset({0.0, 1.0}) or
                    set(unique_values).issubset({0, 1.0}) or
                    set(unique_values).issubset({0.0, 1})
                ):
                    binary_columns.append(col)
        
        # Non-binary columns with "price", "offer", or "discount" in the name are non_media_treatment_columns
        non_media_treatment_columns = [
            col for col in all_columns 
            if col not in binary_columns and any(keyword in col.lower() for keyword in ["price", "offer", "discount"])
        ]
        
        # Exclude columns that are already categorized
        excluded_columns = set(media_channels + binary_columns + non_media_treatment_columns + ["date", "quantity", "revenue"])
        
        # Remaining columns are control_columns
        control_columns = binary_columns + [col for col in all_columns if col not in excluded_columns]
        
        return {
            "media_channels": media_channels,
            "control_columns": control_columns,
            "non_media_treatment_columns": non_media_treatment_columns,
        }
    
    def get_meridian_dataset(self) -> pd.DataFrame:
        """Transform the dataset for Meridian framework.
        
        Meridian requires:
        - A dummy geography column (since the data is aggregated to node level)
        - Specific column structure for media channels
        
        Returns:
            DataFrame formatted for Meridian
        """
        df = self.raw_data.copy()
        
        # Add dummy geography column for Meridian
        # Since data is aggregated to node level, we create a single geography
        df["geo"] = "national"
        logger.info(f"Transformed dataset for Meridian with shape {df.shape}")
        return df.reset_index()
    
    def get_pymc_dataset(self) -> pd.DataFrame:
        """Transform the dataset for PyMC framework.
        
        PyMC requires:
        - Control variables scaled to [-1, 1] using maxabs scaling
        - Standard column structure
        
        Returns:
            DataFrame formatted for PyMC
        """
        df = self.raw_data.copy()
        
        # Get column categorization for PyMC
        column_map = self.get_pymc_column_map()
        control_columns = column_map["control_columns"]
        
        # Scale control columns using maxabs scaling to [-1, 1] range (vectorized)
        if control_columns:
            control_data = df[control_columns]
            max_abs_values = np.abs(control_data).max()
            # Avoid division by zero by setting max_abs to 1 where it's 0
            max_abs_values = max_abs_values.replace(0, 1)
            df[control_columns] = control_data / max_abs_values
            logger.debug(f"Scaled {len(control_columns)} control columns using maxabs scaling")
        
        logger.info(f"Transformed dataset for PyMC with shape {df.shape}")
        logger.info(f"Scaled {len(control_columns)} control columns: {control_columns}")
        return df.reset_index()
    
    @property
    def raw_data(self) -> pd.DataFrame:
        """Get the raw processed dataset.
        
        Returns:
            The raw dataset without framework-specific transformations
        """
        return self._raw_data.copy()