"""Unit tests for validation test result classes."""

from datetime import datetime

import pandas as pd

from mmm_eval.core.validation_test_results import ValidationResults, ValidationTestResult
from mmm_eval.core.validation_tests_models import (
    ValidationTestAttributeNames,
    ValidationTestNames,
)
from mmm_eval.metrics.metric_models import (
    AccuracyMetricNames,
    AccuracyMetricResults,
    PerturbationMetricResults,
    RefreshStabilityMetricNames,
    RefreshStabilityMetricResults,
    TestResultDFAttributes,
)


class TestValidationTestResult:
    """Test cases for TestResult class."""

    def test_test_result_creation_and_to_df(self):
        """Test TestResult creation and to_df conversion."""
        test_scores = AccuracyMetricResults(mape=0.1, r_squared=0.8)
        metric_names = AccuracyMetricNames.to_list()

        result = ValidationTestResult(
            test_name=ValidationTestNames.ACCURACY,
            metric_names=metric_names,
            test_scores=test_scores,
        )

        # Test basic properties
        assert result.test_name == ValidationTestNames.ACCURACY
        assert isinstance(result.timestamp, datetime)

        # Test to_df conversion
        result_df = result.to_df()
        assert isinstance(result_df, pd.DataFrame)
        assert result_df[ValidationTestAttributeNames.TEST_NAME.value].iloc[0] == ValidationTestNames.ACCURACY.value
        assert ValidationTestAttributeNames.TIMESTAMP.value in result_df.columns

        # Check that we have the expected long-format structure
        expected_columns = [
            TestResultDFAttributes.GENERAL_METRIC_NAME.value,
            TestResultDFAttributes.SPECIFIC_METRIC_NAME.value,
            TestResultDFAttributes.METRIC_VALUE.value,
            TestResultDFAttributes.METRIC_PASS.value,
            ValidationTestAttributeNames.TEST_NAME.value,
            ValidationTestAttributeNames.TIMESTAMP.value,
        ]
        for col in expected_columns:
            assert col in result_df.columns

    def test_test_result_to_df_with_series_metrics(self):
        """Test TestResult to_df conversion with Series-based metrics."""
        # Test with RefreshStabilityMetricResults
        mean_series = pd.Series({"channel_1": 0.1, "channel_2": 0.05})
        std_series = pd.Series({"channel_1": 0.02, "channel_2": 0.01})
        test_scores = RefreshStabilityMetricResults(
            mean_percentage_change_for_each_channel=mean_series,
            std_percentage_change_for_each_channel=std_series,
        )
        metric_names = RefreshStabilityMetricNames.to_list()

        result = ValidationTestResult(
            test_name=ValidationTestNames.REFRESH_STABILITY,
            metric_names=metric_names,
            test_scores=test_scores,
        )

        # Test to_df conversion
        result_df = result.to_df()
        assert isinstance(result_df, pd.DataFrame)
        assert (
            result_df[ValidationTestAttributeNames.TEST_NAME.value].iloc[0]
            == ValidationTestNames.REFRESH_STABILITY.value
        )

        # Check that test_scores are properly included in DataFrame (long format)
        # Should have 4 rows: 2 channels × 2 metrics (mean and std)
        assert len(result_df) == 4

        # Check that we have the expected metric names in the general_metric_name column
        metric_names_in_df = result_df[TestResultDFAttributes.GENERAL_METRIC_NAME.value].unique()
        expected_metric_names = [
            RefreshStabilityMetricNames.MEAN_PERCENTAGE_CHANGE.value,
            RefreshStabilityMetricNames.STD_PERCENTAGE_CHANGE.value,
        ]
        for metric_name in expected_metric_names:
            assert metric_name in metric_names_in_df


class TestValidationResults:
    """Test cases for ValidationResults class."""

    def test_validation_results_creation_and_basic_operations(self):
        """Test ValidationResults creation and basic operations."""
        # Create test results
        accuracy_result = ValidationTestResult(
            test_name=ValidationTestNames.ACCURACY,
            metric_names=AccuracyMetricNames.to_list(),
            test_scores=AccuracyMetricResults(mape=0.1, r_squared=0.8),
        )

        # Create stability result with new field names
        mean_series = pd.Series({"channel_1": 0.1, "channel_2": 0.05})
        std_series = pd.Series({"channel_1": 0.02, "channel_2": 0.01})
        stability_result = ValidationTestResult(
            test_name=ValidationTestNames.REFRESH_STABILITY,
            metric_names=RefreshStabilityMetricNames.to_list(),
            test_scores=RefreshStabilityMetricResults(
                mean_percentage_change_for_each_channel=mean_series,
                std_percentage_change_for_each_channel=std_series,
            ),
        )

        test_results = {
            ValidationTestNames.ACCURACY: accuracy_result,
            ValidationTestNames.REFRESH_STABILITY: stability_result,
        }

        validation_result = ValidationResults(test_results)

        # Test basic properties
        assert validation_result.test_results == test_results

        # Test get_test_result
        retrieved_result = validation_result.get_test_result(ValidationTestNames.ACCURACY)
        assert retrieved_result == accuracy_result

    def test_validation_result_to_df(self):
        """Test ValidationResults to_df conversion."""
        accuracy_result = ValidationTestResult(
            test_name=ValidationTestNames.ACCURACY,
            metric_names=AccuracyMetricNames.to_list(),
            test_scores=AccuracyMetricResults(mape=0.1, r_squared=0.8),
        )

        test_results = {ValidationTestNames.ACCURACY: accuracy_result}
        validation_result = ValidationResults(test_results)

        result_df = validation_result.to_df()

        assert isinstance(result_df, pd.DataFrame)
        assert ValidationTestAttributeNames.TEST_NAME.value in result_df.columns
        assert ValidationTestAttributeNames.TIMESTAMP.value in result_df.columns
        # Should have 2 rows: one for each accuracy metric (mape and r_squared)
        assert len(result_df) == 2

    def test_validation_result_to_df_with_series_metrics(self):
        """Test ValidationResults to_df conversion with Series-based metrics."""
        # Create test result with Series-based metrics
        percentage_change_series = pd.Series({"TV": 3.0, "Radio": 7.0})
        perturbation_result = ValidationTestResult(
            test_name=ValidationTestNames.PERTURBATION,
            metric_names=["percentage_change_for_each_channel"],
            test_scores=PerturbationMetricResults(percentage_change_for_each_channel=percentage_change_series),
        )

        test_results = {ValidationTestNames.PERTURBATION: perturbation_result}
        validation_result = ValidationResults(test_results)

        result_df = validation_result.to_df()

        # Check that the result is properly converted to DataFrame
        assert isinstance(result_df, pd.DataFrame)
        assert ValidationTestAttributeNames.TEST_NAME.value in result_df.columns
        assert result_df[ValidationTestAttributeNames.TEST_NAME.value].iloc[0] == ValidationTestNames.PERTURBATION.value

        # Check that the test scores are properly included in DataFrame (long format)
        # Should have 2 rows: one for each channel (TV and Radio)
        assert len(result_df) == 2

        # Check that we have the expected metric name in the general_metric_name column
        metric_names_in_df = result_df[TestResultDFAttributes.GENERAL_METRIC_NAME.value].unique()
        assert "percentage_change" in metric_names_in_df
