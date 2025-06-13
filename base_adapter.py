"""
Base adapter class for MMM frameworks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from ..core.results import EvaluationResults


class BaseAdapter(ABC):
    """
    Abstract base class for MMM framework adapters.

    All framework adapters must inherit from this class and implement
    the required methods to provide a unified interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter with framework-specific configuration.

        Args:
            config: Framework-specific configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self.is_fitted = False

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
    def get_framework_name(self) -> str:
        """
        Return the name of the MMM framework.

        Returns:
            Framework name (e.g., 'meridian', 'pymc', 'robyn')
        """
        pass

    def fit_and_evaluate(
        self,
        data: pd.DataFrame,
        target_column: str = "kpi",
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> EvaluationResults:
        """
        Fit the model and evaluate it using specified metrics.

        Args:
            data: Input data
            target_column: Name of the target/KPI column
            metrics: List of metrics to compute
            **kwargs: Additional parameters

        Returns:
            EvaluationResults object containing computed metrics
        """
        from ..metrics.accuracy import AVAILABLE_METRICS
        from ..core.evaluator import _compute_metrics

        if metrics is None:
            metrics = ["mape", "rmse"]

        # Fit the model
        self.fit(data, **kwargs)

        # Generate predictions
        predictions = self.predict(data, **kwargs)

        # Compute metrics
        actual = data[target_column]
        metric_results = _compute_metrics(actual, predictions, metrics)

        return EvaluationResults(
            framework=self.get_framework_name(),
            metrics=metric_results,
            predictions=predictions,
            actual=actual,
        )
