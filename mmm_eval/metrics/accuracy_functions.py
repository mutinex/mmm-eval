"""Accuracy metrics for MMM evaluation."""

import numpy as np
import pandas as pd

from mmm_eval.metrics.metric_models import AccuracyMetricNames, AccuracyMetricResults


def calculate_smape(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE).

    SMAPE is calculated as: 100 * (2 * |actual - predicted|) / (|actual| + |predicted|)

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        SMAPE value as float (percentage)

    Raises:
        ValueError: If series are empty or have different lengths

    """
    # Validate inputs
    if len(actual) == 0 or len(predicted) == 0:
        raise ValueError("Cannot calculate SMAPE on empty series")

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted series must have the same length")

    # Handle NaN values
    if actual.isna().any() or predicted.isna().any():
        return float(np.nan)

    # Handle division by zero and edge cases
    denominator = np.abs(actual) + np.abs(predicted)
    # Avoid division by zero by setting denominator to 1 where it's 0
    denominator = np.where(denominator == 0, 1, denominator)

    smape = 100 * np.mean(2 * np.abs(actual - predicted) / denominator)
    return float(smape)


def calculate_mean_for_singular_values_across_cross_validation_folds(
    fold_metrics: list[AccuracyMetricResults],
    metric_name: AccuracyMetricNames,
) -> float:
    """Calculate the mean of the fold metrics for single values.

    Args:
        fold_metrics: List of metric result objects
        metric_name: Name of the metric attribute

    Returns:
        Mean value as float

    """
    metric_attr = metric_name.value
    return np.mean([getattr(fold_metric, metric_attr) for fold_metric in fold_metrics])


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
    metric_name: AccuracyMetricNames,
) -> float:
    """Calculate the standard deviation of the fold metrics for single values.

    Args:
        fold_metrics: List of metric result objects
        metric_name: Name of the metric attribute

    Returns:
        Standard deviation value as float

    """
    metric_attr = metric_name.value
    return np.std([getattr(fold_metric, metric_attr) for fold_metric in fold_metrics])


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
    """Calculate the absolute percentage change between two series."""
    return np.abs((comparison_series - baseline_series) / baseline_series)
