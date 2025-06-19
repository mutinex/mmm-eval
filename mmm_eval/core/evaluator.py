"""
Main evaluator for MMM frameworks.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path
from .results import EvaluationResults


def _compute_metrics(
    actual: pd.Series, predicted: pd.Series, metrics: List[str]
) -> Dict[str, float]:
    """
    Compute specified metrics for actual vs predicted values.

    Args:
        actual: Actual values
        predicted: Predicted values
        metrics: List of metric names to compute

    Returns:
        Dictionary mapping metric names to values
    """
    from ..metrics.accuracy import AVAILABLE_METRICS

    results = {}

    for metric in metrics:
        if metric not in AVAILABLE_METRICS:
            raise ValueError(
                f"Unknown metric: {metric}. Available metrics: {list(AVAILABLE_METRICS.keys())}"
            )

        metric_func = AVAILABLE_METRICS[metric]
        results[metric] = metric_func(actual, predicted)

    return results


def evaluate_framework(
    framework: str,
    data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    **kwargs,
) -> EvaluationResults:
    """
    Evaluate an MMM framework using the unified API.

    Args:
        framework: Name of the MMM framework to evaluate
        data: Input data containing media channels, KPI, and other variables
        config: Framework-specific configuration
        metrics: List of metrics to compute (defaults to ["mape", "rmse"])
        **kwargs: Additional framework-specific parameters

    Returns:
        EvaluationResults object containing evaluation metrics and predictions

    Example:
        >>> import pandas as pd
        >>> from mmm_eval import evaluate_framework
        >>>
        >>> # Sample data
        >>> data = pd.DataFrame({
        ...     'kpi': [100, 120, 110, 130],
        ...     'tv': [50, 60, 55, 65],
        ...     'digital': [30, 35, 32, 40]
        ... })
        >>>
        >>> # Evaluate framework
        >>> results = evaluate_framework(
        ...     framework="meridian",
        ...     data=data,
        ...     metrics=["mape", "rmse", "r_squared"]
        ... )
        >>> print(results)
    """
    from ..adapters import get_adapter

    if metrics is None:
        metrics = ["mape", "rmse"]

    # Get the appropriate adapter for the framework
    adapter = get_adapter(framework, config)

    return 0

    # TODO: implement fit and evaluate

    # Use the adapter to fit and evaluate
    # results = adapter.fit_and_evaluate(
    #     data=data, target_column=target_column, metrics=metrics, **kwargs
    # )

    # save result to output_path

    # return results
