"""
Abstract base classes for MMM validation framework.
"""

from abc import ABC, abstractmethod
from typing import Any
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
