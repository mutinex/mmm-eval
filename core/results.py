"""
Result containers for MMM evaluation framework.
"""

from typing import Dict, Any, Optional
import pandas as pd


class EvaluationResults:
    """
    Container for MMM evaluation results.

    This class holds the results of evaluating an MMM framework,
    including computed metrics, predictions, and actual values.
    """

    def __init__(
        self,
        framework: str,
        metrics: Dict[str, float],
        predictions: Optional[pd.Series] = None,
        actual: Optional[pd.Series] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize evaluation results.

        Args:
            framework: Name of the MMM framework evaluated
            metrics: Dictionary of computed metrics (metric_name -> value)
            predictions: Model predictions (optional)
            actual: Actual target values (optional)
            metadata: Additional metadata about the evaluation (optional)
        """
        self.framework = framework
        self.metrics = metrics
        self.predictions = predictions
        self.actual = actual
        self.metadata = metadata or {}

    def get_metric(self, metric_name: str) -> float:
        """
        Get a specific metric value.

        Args:
            metric_name: Name of the metric to retrieve

        Returns:
            Metric value

        Raises:
            KeyError: If metric not found
        """
        if metric_name not in self.metrics:
            raise KeyError(
                f"Metric '{metric_name}' not found. Available metrics: {list(self.metrics.keys())}"
            )
        return self.metrics[metric_name]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to dictionary format.

        Returns:
            Dictionary representation of results
        """
        result = {
            "framework": self.framework,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

        if self.predictions is not None:
            result["predictions"] = self.predictions.to_list()

        if self.actual is not None:
            result["actual"] = self.actual.to_list()

        return result

    def __repr__(self) -> str:
        """String representation of results."""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in self.metrics.items()])
        return (
            f"EvaluationResults(framework={self.framework}, metrics={{{metrics_str}}})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"Framework: {self.framework}"]
        lines.append("Metrics:")
        for metric, value in self.metrics.items():
            lines.append(f"  {metric}: {value:.4f}")
        return "\n".join(lines)
