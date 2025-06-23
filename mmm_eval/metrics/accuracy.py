"""Accuracy metrics for MMM evaluation."""

import numpy as np
import pandas as pd


def mae(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE).

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        MAE value

    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return np.mean(np.abs(actual - predicted))


def mape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        MAPE value as percentage

    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    # Avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        return np.inf

    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def rmse(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """Calculate Root Mean Square Error (RMSE).

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        RMSE value

    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))


def r_squared(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """Calculate R-squared (coefficient of determination).

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        R-squared value

    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def symmetric_mape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        SMAPE value as percentage

    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    # Avoid division by zero
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denominator != 0

    if not np.any(mask):
        return np.inf

    return np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100


# Registry of available metrics
AVAILABLE_METRICS = {
    "mae": mae,
    "mape": mape,
    "rmse": rmse,
    "r_squared": r_squared,
    "smape": symmetric_mape,
}
