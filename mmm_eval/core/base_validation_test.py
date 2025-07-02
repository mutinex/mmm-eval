"""Abstract base classes for MMM validation framework."""

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split

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
        train_idx, test_idx = split_timeseries_data(data, ValidationTestConstants.TRAIN_TEST_SPLIT_TEST_PROPORTION,
                                                    date_column=self.date_column)
        return data[train_idx], data[test_idx]

    def _split_data_time_series_cv(self, data: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split the data into train and test sets using time series cross-validation.

        Args:
            data: The data to split

        Returns:
            list of tuples, each containing the train and test indices

        """
        logger.info(f"Splitting data into train and test sets for {self.test_name} test")

        return list(split_timeseries_cv(data, ValidationTestConstants.N_SPLITS, ValidationTestConstants.TIME_SERIES_CROSS_VALIDATION_TEST_SIZE,
                                        date_column=self.date_column))


def split_timeseries_data(data: pd.DataFrame, test_proportion: float, date_column: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Split data globally based on date, without grouping.
    """
    sorted_dates = sorted(data[date_column].unique())
    split_idx = int(len(sorted_dates) * (1 - test_proportion))
    cutoff = sorted_dates[split_idx]

    train_mask = data[date_column] < cutoff
    test_mask = data[date_column] >= cutoff

    return train_mask, test_mask


# TODO: add logic to ensure there's actually enough data to satisfy splits
def split_timeseries_cv(data: pd.DataFrame, n_splits: int, test_size: int, date_column: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Generator yielding train/test masks for rolling CV, globally based on date.

    This simulates monthly refreshes in the case where the data is at a weekly frequency.
    """
    sorted_dates = sorted(data[date_column].unique())
    n_dates = len(sorted_dates)

    for i in range(n_splits):
        test_end = n_dates - i * test_size
        test_start = n_dates - (i+1) * test_size
        test_dates = sorted_dates[test_start:test_end]
        train_dates = sorted_dates[:test_start]

        train_mask = data[date_column].isin(train_dates)
        test_mask = data[date_column].isin(test_dates)
        yield train_mask, test_mask