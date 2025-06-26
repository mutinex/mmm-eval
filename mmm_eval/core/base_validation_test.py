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

    def __init__(self):
        """Initialize the validation test."""
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

        train, test = train_test_split(
            data,
            test_size=ValidationTestConstants.TRAIN_TEST_SPLIT_TEST_SIZE,
            random_state=ValidationTestConstants.RANDOM_STATE,
        )

        return train, test

    def _split_data_time_series_cv(self, data: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split the data into train and test sets using time series cross-validation.

        Args:
            data: The data to split

        Returns:
            list of tuples, each containing the train and test indices

        """
        logger.info(f"Splitting data into train and test sets for {self.test_name} test")

        cv = TimeSeriesSplit(
            n_splits=ValidationTestConstants.N_SPLITS,
            test_size=ValidationTestConstants.TIME_SERIES_CROSS_VALIDATION_TEST_SIZE,
        )

        return list(cv.split(data))
