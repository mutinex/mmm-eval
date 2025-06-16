"""
Accuracy metrics for MMM evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union
from mmm_eval.metrics.metric_models import AccuracyMetricNames, AccuracyMetricResults

#todo(): Can we use Phils metrics here instead?

def calculate_mape(
    actual: pd.Series, predicted: pd.Series
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        MAPE as a percentage (0-100)
    """

    actual = np.array(actual)
    predicted = np.array(predicted)

    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def calculate_r_squared(
    actual: Union[pd.Series, np.ndarray], predicted: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate R-squared (coefficient of determination).

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        R-squared value (0-1, where 1 is perfect fit)
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1 - (ss_res / ss_tot))

def calculate_mean_for_cross_validation_folds(
    fold_metrics: List[AccuracyMetricResults], 
    metric_name: AccuracyMetricNames,
) -> float:
    """
    Calculate the mean of the fold metrics.
    """
    metric_name = metric_name.value
    return np.mean([getattr(fold_metric, metric_name) for fold_metric in fold_metrics])

def calculate_std_for_cross_validation_folds(
    fold_metrics: List[AccuracyMetricResults], 
    metric_name: AccuracyMetricNames,
) -> float:
    """
    Calculate the standard deviation of the fold metrics.
    """
    metric_name = metric_name.value
    return np.std([getattr(fold_metric, metric_name) for fold_metric in fold_metrics])