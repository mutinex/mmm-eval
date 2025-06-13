"""
Accuracy metrics for MMM evaluation.
"""

import numpy as np
import pandas as pd
from typing import Union


def mape(
    actual: Union[pd.Series, np.ndarray], predicted: Union[pd.Series, np.ndarray]
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

    # Handle zero values to avoid division by zero
    mask = actual != 0
    if not mask.any():
        return 0.0

    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmse(
    actual: Union[pd.Series, np.ndarray], predicted: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate Root Mean Square Error (RMSE).

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        RMSE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(
    actual: Union[pd.Series, np.ndarray], predicted: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        MAE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    return float(np.mean(np.abs(actual - predicted)))


def r_squared(
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


def symmetric_mape(
    actual: Union[pd.Series, np.ndarray], predicted: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        SMAPE as a percentage (0-100)
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denominator != 0

    if not mask.any():
        return 0.0

    return float(
        np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100
    )


# Registry of available metrics
AVAILABLE_METRICS = {
    "mape": mape,
    "rmse": rmse,
    "mae": mae,
    "r_squared": r_squared,
    "symmetric_mape": symmetric_mape,
}
