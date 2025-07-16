"""Abstract base classes for MMM validation framework."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator

import numpy as np
import pandas as pd
from pydantic import PositiveInt

from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.core.constants import ValidationTestConstants
from mmm_eval.core.exceptions import (
    MetricCalculationError,
    TestExecutionError,
)
from mmm_eval.core.validation_test_results import ValidationTestResult

logger = logging.getLogger(__name__)


class BaseValidationTest(ABC):
    """Abstract base class for validation tests.

    All validation tests must inherit from this class and implement
    the required methods to provide a unified testing interface.
    """

    def __init__(self, date_column: str):
        """Initialize the validation test."""
        self.date_column = date_column
        self.rng = np.random.default_rng(ValidationTestConstants.RANDOM_STATE)

    def run_with_error_handling(self, adapter: BaseAdapter, data: pd.DataFrame) -> "ValidationTestResult":
        """Run the validation test with error handling.

        Args:
            adapter: The adapter to validate
            data: Input data for validation

        Returns:
            TestResult object containing test results

        Raises:
            MetricCalculationError: If metric calculation fails
            TestExecutionError: If test execution fails

        """
        try:
            return self.run(adapter, data)
        except ZeroDivisionError as e:
            # This is clearly a mathematical calculation issue
            raise MetricCalculationError(f"Metric calculation error in {self.test_name} test: {str(e)}") from e
        except Exception as e:
            # All other errors - let individual tests handle specific categorization if needed
            raise TestExecutionError(f"Test execution error in {self.test_name} test: {str(e)}") from e

    @abstractmethod
    def run(self, adapter: BaseAdapter, data: pd.DataFrame) -> "ValidationTestResult":
        """Run the validation test.

        Args:
            adapter: The adapter to validate
            data: Input data for validation

        Returns:
            TestResult object containing test results

        """
        pass

    @property
    @abstractmethod
    def test_name(self) -> str:
        """Return the name of the test.

        Returns
            Test name (e.g., 'accuracy', 'stability')

        """
        pass

    def _split_data_holdout(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data into train and test sets.

        Args:
            data: The data to split

        Returns:
            train: The train data
            test: The test data

        """
        logger.info(f"Splitting data into train and test sets for {self.test_name} test")

        train_idx, test_idx = split_timeseries_data(
            data, test_size=ValidationTestConstants.ACCURACY_TEST_SIZE, date_column=self.date_column
        )
        return data[train_idx], data[test_idx]

    def _split_data_time_series_cv(self, data: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split the data into train and test sets using time series cross-validation.

        Args:
            data: The data to split

        Returns:
            list of tuples, each containing the train and test indices

        """
        logger.info(f"Splitting data into train and test sets for {self.test_name} test")

        return list(
            split_timeseries_cv(
                data,
                ValidationTestConstants.N_SPLITS,
                ValidationTestConstants.TIME_SERIES_CROSS_VALIDATION_TEST_SIZE,
                date_column=self.date_column,
            )
        )


def split_timeseries_data(
    data: pd.DataFrame,
    test_size: PositiveInt,
    date_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Split data globally based on date.

    Given a timeseries, reserve the last `test_size` unique dates for test set whereas
    all other dates are used for training.

    Arguments:
        data: timeseries data to split, possibly with another index like geography
        date_column: name of the date column
        test_size: number of unique dates to reserve for testing.

    Returns:
        boolean masks for training and test data respectively

    Raises:
        ValueError: if `test_size` is invalid.
    
    """
    if test_size <= 0:
        raise ValueError("`test_size` must be greater than 0")

    sorted_dates = sorted(data[date_column].unique())

    # Reserve the last test_size data points for testing
    if test_size >= len(sorted_dates):
        raise ValueError(
            f"`test_size` ({test_size}) must be less than the number of unique dates ({len(sorted_dates)})"
        )

    cutoff = sorted_dates[-test_size]  # Use the date before the test period starts

    train_mask = data[date_column] < cutoff
    test_mask = data[date_column] >= cutoff

    return train_mask, test_mask


def split_timeseries_cv(
    data: pd.DataFrame, n_splits: PositiveInt, test_size: PositiveInt, date_column: str
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Produce train/test masks for rolling CV, split globally based on date.

    This simulates regular refreshes and utilises the last `test_size` data points for
    testing in the first fold, using all prior data for training. For a dataset with
    T dates, the subsequen test folds follow the pattern [T-4, T], [T-8, T-4], ...

    Arguments:
        data: dataframe of MMM data to be split
        n_splits: number of unique folds to generate
        test_size: the number of observations in each testing fold
        date_column: the name of the date column in the dataframe to split by

    Yields:
        integer masks corresponding training and test set indices.

    """
    sorted_dates = sorted(data[date_column].unique())
    n_dates = len(sorted_dates)

    # assuming the minimum training set size allowable is equal to `test_size`, ensure there's
    # enough data temporally to do the splits
    n_required_dates = test_size * (n_splits + 1)
    if n_dates < n_required_dates:
        raise ValueError(
            "Insufficient timeseries data provided for splitting. In order to "
            f"perform {n_splits} splits with test_size={test_size}, at least "
            f"{n_required_dates} unique dates are required, but only {n_dates} "
            f"dates are available."
        )

    for i in range(n_splits):
        test_end = n_dates - i * test_size
        test_start = n_dates - (i + 1) * test_size
        test_dates = sorted_dates[test_start:test_end]
        train_dates = sorted_dates[:test_start]

        train_mask = data[date_column].isin(train_dates)
        test_mask = data[date_column].isin(test_dates)
        yield train_mask, test_mask
