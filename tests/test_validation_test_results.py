"""
Unit tests for validation test result classes.
"""

import pytest
import pandas as pd
from datetime import datetime
from mmm_eval.core.validation_test_results import TestResult, ValidationResult
from mmm_eval.core.validation_tests_models import (
    ValidationTestNames,
    ValidationTestAttributeNames,
    ValidationResultAttributeNames,
)
from mmm_eval.metrics.metric_models import (
    AccuracyMetricResults,
    RefreshStabilityMetricResults,
    AccuracyMetricNames,
    RefreshStabilityMetricNames,
)


class TestTestResult:
    """Test cases for TestResult class."""

    def test_test_result_creation_and_to_dict(self):
        """Test TestResult creation and to_dict conversion."""
        test_scores = AccuracyMetricResults(mape=0.1, r_squared=0.8)
        metric_names = AccuracyMetricNames.metrics_to_list()
        
        result = TestResult(
            test_name=ValidationTestNames.ACCURACY,
            passed=True,
            metric_names=metric_names,
            test_scores=test_scores,
        )
        
        # Test basic properties
        assert result.test_name == ValidationTestNames.ACCURACY
        assert result.passed is True
        assert isinstance(result.timestamp, datetime)
        
        # Test to_dict conversion
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict[ValidationTestAttributeNames.TEST_NAME] == ValidationTestNames.ACCURACY
        assert result_dict[ValidationTestAttributeNames.PASSED] is True
        assert ValidationTestAttributeNames.TIMESTAMP in result_dict


class TestValidationResult:
    """Test cases for ValidationResult class."""

    def test_validation_result_creation_and_basic_operations(self):
        """Test ValidationResult creation and basic operations."""
        # Create test results
        accuracy_result = TestResult(
            test_name=ValidationTestNames.ACCURACY,
            passed=True,
            metric_names=AccuracyMetricNames.metrics_to_list(),
            test_scores=AccuracyMetricResults(mape=0.1, r_squared=0.8),
        )
        
        # Create stability result with new field names
        mean_series = pd.Series({'channel_1': 0.1, 'channel_2': 0.05})
        std_series = pd.Series({'channel_1': 0.02, 'channel_2': 0.01})
        stability_result = TestResult(
            test_name=ValidationTestNames.REFRESH_STABILITY,
            passed=False,
            metric_names=RefreshStabilityMetricNames.metrics_to_list(),
            test_scores=RefreshStabilityMetricResults(
                mean_percentage_change_for_each_channel=mean_series,
                std_percentage_change_for_each_channel=std_series
            ),
        )
        
        test_results = {
            ValidationTestNames.ACCURACY: accuracy_result,
            ValidationTestNames.REFRESH_STABILITY: stability_result,
        }
        
        validation_result = ValidationResult(test_results)
        
        # Test basic properties
        assert validation_result.test_results == test_results
        assert isinstance(validation_result.timestamp, datetime)
        
        # Test get_test_result
        retrieved_result = validation_result.get_test_result(ValidationTestNames.ACCURACY)
        assert retrieved_result == accuracy_result
        
        # Test all_passed
        assert validation_result.all_passed() is False

    def test_all_passed_empty(self):
        """Test all_passed with no test results."""
        test_results = {}
        validation_result = ValidationResult(test_results)
        assert validation_result.all_passed() is True

    def test_validation_result_to_dict(self):
        """Test ValidationResult to_dict conversion."""
        accuracy_result = TestResult(
            test_name=ValidationTestNames.ACCURACY,
            passed=True,
            metric_names=AccuracyMetricNames.metrics_to_list(),
            test_scores=AccuracyMetricResults(mape=0.1, r_squared=0.8),
        )
        
        test_results = {ValidationTestNames.ACCURACY: accuracy_result}
        validation_result = ValidationResult(test_results)
        
        result_dict = validation_result.to_dict()
        
        assert ValidationResultAttributeNames.TIMESTAMP in result_dict
        assert ValidationResultAttributeNames.ALL_PASSED in result_dict
        assert ValidationResultAttributeNames.RESULTS in result_dict
        assert result_dict[ValidationResultAttributeNames.ALL_PASSED] is True
        assert ValidationTestNames.ACCURACY in result_dict[ValidationResultAttributeNames.RESULTS] 