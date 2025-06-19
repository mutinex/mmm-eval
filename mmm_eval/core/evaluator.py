"""Main evaluator for MMM frameworks."""

from pathlib import Path

import pandas as pd

from mmm_eval.configs import EvalConfig
from .validation_test_results import ValidationResult


def evaluate_framework(
    framework: str,
    data: pd.DataFrame,
    config: BaseConfig,
    metrics: list[str] | None = None,
    output_path: Path | None = None,
    **kwargs,
) -> ValidationResult:
    """
    Evaluate an MMM framework using the unified API.

    Args:
        framework: Name of the MMM framework to evaluate
        data: Input data containing media channels, KPI, and other variables
        config: Framework-specific configuration
        metrics: List of metrics to compute (defaults to ["mape", "rmse"])
        output_path: Optional path to save evaluation results
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
    if metrics is None:
        metrics = ["mape", "rmse"]

    # TODO: implement fit and evaluate
    # For now, return a placeholder result
    return EvaluationResults(
        framework=framework,
        metrics={},
        predictions=pd.Series(),
        actual=pd.Series(),
    )
