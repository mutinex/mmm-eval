"""
Unit tests for metric models.
"""

import pandas as pd
from mmm_eval.metrics.metric_models import (
    AccuracyMetricResults,
    CrossValidationMetricResults,
    RefreshStabilityMetricResults,
    PerturbationMetricResults,
)


class TestAccuracyMetricResults:
    """Test cases for accuracy metric results."""

    def test_accuracy_test_passed_good_metrics(self):
        """Test accuracy test passed with good metrics."""
        results = AccuracyMetricResults(mape=0.1, r_squared=0.85)
        assert results.check_test_passed() is True

    def test_accuracy_test_failed_high_mape(self):
        """Test accuracy test failed with high MAPE."""
        results = AccuracyMetricResults(mape=0.2, r_squared=0.85)
        assert results.check_test_passed() is False

    def test_accuracy_test_failed_low_r_squared(self):
        """Test accuracy test failed with low R-squared."""
        results = AccuracyMetricResults(mape=0.1, r_squared=0.75)
        assert results.check_test_passed() is False


class TestCrossValidationMetricResults:
    """Test cases for cross-validation metric results."""

    def test_cross_validation_test_passed_good_metrics(self):
        """Test cross-validation test passed with good metrics."""
        results = CrossValidationMetricResults(
            mean_mape=0.12, std_mape=0.02, mean_r_squared=0.85, std_r_squared=0.03
        )
        assert results.check_test_passed() is True

    def test_cross_validation_test_failed_high_mean_mape(self):
        """Test cross-validation test failed with high mean MAPE."""
        results = CrossValidationMetricResults(
            mean_mape=0.16, std_mape=0.02, mean_r_squared=0.85, std_r_squared=0.03
        )
        assert results.check_test_passed() is False

    def test_cross_validation_test_failed_low_mean_r_squared(self):
        """Test cross-validation test failed with low mean R-squared."""
        results = CrossValidationMetricResults(
            mean_mape=0.12, std_mape=0.02, mean_r_squared=0.75, std_r_squared=0.03
        )
        assert results.check_test_passed() is False


class TestRefreshStabilityMetricResults:
    """Test cases for refresh stability metric results."""

    def test_stability_test_passed_good_metrics(self):
        """Test stability test passed with good metrics."""
        mean_series = pd.Series({"channel_1": 0.1, "channel_2": 0.05})
        std_series = pd.Series({"channel_1": 0.02, "channel_2": 0.01})
        results = RefreshStabilityMetricResults(
            mean_percentage_change_for_each_channel=mean_series,
            std_percentage_change_for_each_channel=std_series,
        )
        assert results.check_test_passed() is True

    def test_stability_test_failed_high_mean_percentage_change(self):
        """Test stability test failed with high mean percentage change."""
        mean_series = pd.Series(
            {"channel_1": 0.16, "channel_2": 0.05}
        )  # channel_1 > threshold
        std_series = pd.Series({"channel_1": 0.02, "channel_2": 0.01})
        results = RefreshStabilityMetricResults(
            mean_percentage_change_for_each_channel=mean_series,
            std_percentage_change_for_each_channel=std_series,
        )
        assert results.check_test_passed() is False


class TestPerturbationMetricResults:
    """Test cases for perturbation metric results."""

    def test_perturbation_test_passed_good_metrics(self):
        """Test perturbation test passed with good metrics."""
        percentage_change_series = pd.Series({"TV": 0.03, "Radio": 0.07})
        results = PerturbationMetricResults(
            percentage_change_for_each_channel=percentage_change_series
        )
        assert results.check_test_passed() is True

    def test_perturbation_test_failed_high_individual_channel(self):
        """Test perturbation test failed with high individual channel change."""
        percentage_change_series = pd.Series(
            {"TV": 0.03, "Radio": 0.11}
        )  # Radio > threshold
        results = PerturbationMetricResults(
            percentage_change_for_each_channel=percentage_change_series
        )
        assert results.check_test_passed() is False
