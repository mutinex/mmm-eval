"""
Base adapter class for MMM frameworks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseAdapter(ABC):
    """
    Abstract base class for MMM framework adapters.

    All framework adapters must inherit from this class and implement
    the required methods to provide a unified interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.model = None
        self.is_fitted = False
        self.config = config or None

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Fit the MMM model to the provided data.

        Args:
            data: Input data containing media channels, KPI, and other variables
            **kwargs: Additional framework-specific parameters
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate predictions using the fitted model.

        Args:
            data: Input data for prediction
            **kwargs: Additional framework-specific parameters

        Returns:
            Predicted values as a pandas Series
        """
        pass

    @abstractmethod
    def get_channel_roi(self) -> Dict[str, float]:
        """
        Return ROI by channel.
        """
        pass
