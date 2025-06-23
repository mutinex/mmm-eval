"""Accuracy metrics for MMM evaluation.
"""

from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from mmm_eval.metrics.metric_models import (
    AccuracyMetricNames,
    AccuracyMetricResults,
)


def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE).

    This function wraps sklearn's mean_absolute_percentage_error with proper
    type conversion and returns a float value.

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        MAPE as a percentage (0-100)
    
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    return float(mean_absolute_percentage_error(actual, predicted))


def calculate_r_squared(actual: Union[pd.Series, np.ndarray], predicted: Union[pd.Series, np.ndarray]) -> float:
    """Calculate R-squared (coefficient of determination).

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        R-squared value (0-1, where 1 is perfect fit)
    
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    return float(r2_score(actual, predicted))


def calculate_mean_for_singular_values_across_cross_validation_folds(
    fold_metrics: list[AccuracyMetricResults],
    metric_name: Union[AccuracyMetricNames],
) -> float:
    """Calculate the mean of the fold metrics for single values.

    Args:
        fold_metrics: List of metric result objects
        metric_name: Name of the metric attribute

    Returns:
        Mean value as float
    
    """
    metric_name = metric_name.value if hasattr(metric_name, "value") else metric_name
    return np.mean([getattr(fold_metric, metric_name) for fold_metric in fold_metrics])


def calculate_means_for_series_across_cross_validation_folds(
    folds_of_series: list[pd.Series],
) -> pd.Series:
    """Calculate the mean of pandas Series across folds.

    Args:
        folds_of_series: List of pandas Series (e.g., ROI series from different folds)

    Returns:
        Mean Series with same index as input series
    
    """
    return pd.concat(folds_of_series, axis=1).mean(axis=1)


def calculate_std_for_singular_values_across_cross_validation_folds(
    fold_metrics: list[AccuracyMetricResults],
    metric_name: Union[AccuracyMetricNames],
) -> float:
    """Calculate the standard deviation of the fold metrics for single values.

    Args:
        fold_metrics: List of metric result objects
        metric_name: Name of the metric attribute

    Returns:
        Standard deviation value as float
    
    """
    metric_name = metric_name.value if hasattr(metric_name, "value") else metric_name
    return np.std([getattr(fold_metric, metric_name) for fold_metric in fold_metrics])


def calculate_stds_for_series_across_cross_validation_folds(
    folds_of_series: list[pd.Series],
) -> pd.Series:
    """Calculate the standard deviation of pandas Series across folds.

    Args:
        folds_of_series: List of pandas Series (e.g., ROI series from different folds)

    Returns:
        Standard deviation Series with same index as input series
    
    """
    return pd.concat(folds_of_series, axis=1).std(axis=1)


def calculate_absolute_percentage_change(baseline_series: pd.Series, comparison_series: pd.Series) -> pd.Series:
    """Calculate the refresh stability of the MMM framework.
    """
    return np.abs((comparison_series - baseline_series) / baseline_series)
