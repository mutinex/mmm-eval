"""
Abstract base classes for MMM validation framework.
"""

from abc import ABC, abstractmethod
from typing import Any

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from mmm_eval.core.constants import ValidationTestConstants
from mmm_eval.core.validation_test_results import TestResult
import pandas as pd
class BaseValidationTest(ABC):
    """
    Abstract base class for validation tests.
    
    All validation tests must inherit from this class and implement
    the required methods to provide a unified testing interface.
    """

    
    @abstractmethod
    def run(self, model: Any, data: pd.DataFrame) -> 'TestResult':
        """
        Run the validation test.
        
        Args:
            model: The model to validate
            data: Input data for validation
            
        Returns:
            TestResult object containing test resultsTestResult
        """
        pass
    
    @abstractmethod
    def get_test_name(self) -> str:
        """
        Return the name of the test.
        
        Returns:
            Test name (e.g., 'accuracy', 'stability')
        """
        pass

    def _split_data_holdout(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

        """
        Split the data into train and test sets.

        Args:
            data: The data to split

        Returns:
            train: The train data
            test: The test data
        """

        train, test = train_test_split(
            data,
            test_size=ValidationTestConstants.TRAIN_TEST_SPLIT_RATIO,
            random_state=ValidationTestConstants.RANDOM_STATE,
        )

        return train, test
    
    def _split_data_time_series_cv(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

        """
        Split the data into train and test sets using time series cross-validation.

        Args:
            data: The data to split

        Returns:
            train: The train data
            test: The test data
        """

        cv = TimeSeriesSplit(
            n_splits=ValidationTestConstants.N_SPLITS,
            test_size=ValidationTestConstants.TIME_SERIES_CROSS_VALIDATION_TEST_SIZE,
        )

        return cv.split(data)
