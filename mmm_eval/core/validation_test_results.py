"""
Result containers for MMM validation framework.
"""

from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import pandas as pd

from mmm_eval.core.validation_tests_models import ValidationResultAttributeNames, ValidationTestAttributeNames, ValidationTestNames
from mmm_eval.metrics.metric_models import (
    AccuracyMetricResults, 
    CrossValidationMetricResults, 
    RefreshStabilityMetricResults, 
    PerturbationMetricResults
)


class TestResult:
    """
    Container for individual test results.
    
    This class holds the results of a single validation test,
    including pass/fail status, metrics, and any error messages.
    """
    
    def __init__(
        self,
        test_name: ValidationTestNames,
        passed: bool,
        metric_names: List[str],
        test_scores: Union[AccuracyMetricResults, CrossValidationMetricResults, RefreshStabilityMetricResults, PerturbationMetricResults],
    ):
        """
        Initialize test results.
        
        Args:
            test_name: Name of the test
            passed: Whether the test passed
            metric_names: List of metric names
            test_scores: Computed metric results
        """
        self.test_name = test_name
        self.passed = passed
        self.metric_names = metric_names
        self.test_scores = test_scores
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[ValidationTestAttributeNames, Any]:
        """Convert results to dictionary format."""
        return {
            ValidationTestAttributeNames.TEST_NAME: self.test_name,
            ValidationTestAttributeNames.PASSED: self.passed,
            ValidationTestAttributeNames.METRIC_NAMES: self.metric_names,
            ValidationTestAttributeNames.TEST_SCORES: self.test_scores,
            ValidationTestAttributeNames.TIMESTAMP: self.timestamp.isoformat(),
        }


class ValidationResult:
    """
    Container for complete validation results.
    
    This class holds the results of all validation tests run,
    including individual test results and overall summary.
    """
    
    def __init__(self, test_results: Dict[ValidationTestNames, TestResult]):
        """
        Initialize validation results.
        
        Args:
            test_results: Dictionary mapping test names to their results
        """
        self.test_results = test_results
        self.timestamp = datetime.now()
    
    def get_test_result(self, test_name: ValidationTestNames) -> TestResult:
        """Get results for a specific test."""
        return self.test_results[test_name]
    
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return all(result.passed for result in self.test_results.values())
    
    def to_dict(self) -> Dict[ValidationResultAttributeNames, Any]:
        """Convert results to dictionary format."""
        return {
            ValidationResultAttributeNames.TIMESTAMP: self.timestamp.isoformat(),
            ValidationResultAttributeNames.ALL_PASSED: self.all_passed(),
            ValidationResultAttributeNames.RESULTS: {
                name: result.to_dict() 
                for name, result in self.test_results.items()
            }
        }
