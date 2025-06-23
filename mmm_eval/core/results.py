"""Evaluation results data structures."""

from typing import Any

import numpy as np


class EvaluationResults:
    """Container for MMM evaluation results."""

    def __init__(
        self,
        framework: str,
        metrics: dict[str, float],
        predictions: np.ndarray,
        actual: np.ndarray,
        additional_info: dict[str, Any] | None = None,
    ):
        """Initialize evaluation results.

        Args:
            framework: Name of the evaluated framework
            metrics: Dictionary of evaluation metrics
            predictions: Model predictions
            actual: Actual values
            additional_info: Additional framework-specific information

        """
        self.framework = framework
        self.metrics = metrics
        self.predictions = predictions
        self.actual = actual
        self.additional_info = additional_info or {}

    def get_metric(self, metric_name: str) -> float:
        """Get a specific metric value.

        Args:
            metric_name: Name of the metric

        Returns:
            Metric value

        Raises:
            KeyError: If metric doesn't exist

        """
        if metric_name not in self.metrics:
            raise KeyError(f"Metric '{metric_name}' not found. Available: {list(self.metrics.keys())}")
        return self.metrics[metric_name]

    def get_metric_summary(self) -> dict[str, float]:
        """Get a summary of all metrics.

        Returns
            Dictionary of all metrics

        """
        return self.metrics.copy()

    def add_metric(self, name: str, value: float) -> None:
        """Add a new metric to the results.

        Args:
            name: Metric name
            value: Metric value

        """
        self.metrics[name] = value

    def add_info(self, key: str, value: Any) -> None:
        """Add additional information to the results.

        Args:
            key: Information key
            value: Information value

        """
        self.additional_info[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary format.

        Returns
            Dictionary representation of results

        """
        return {
            "framework": self.framework,
            "metrics": self.metrics,
            "predictions": self.predictions.tolist() if hasattr(self.predictions, "tolist") else self.predictions,
            "actual": self.actual.tolist() if hasattr(self.actual, "tolist") else self.actual,
            "additional_info": self.additional_info,
        }

    def __str__(self) -> str:
        """Return string representation of results."""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in self.metrics.items()])
        return f"EvaluationResults(framework='{self.framework}', metrics=[{metrics_str}])"

    def __repr__(self) -> str:
        """Return detailed string representation of results."""
        return self.__str__()
