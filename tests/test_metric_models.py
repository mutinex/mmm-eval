"""Unit tests for metric models."""

import pandas as pd
import pytest

from mmm_eval.metrics.exceptions import InvalidMetricNameException
from mmm_eval.metrics.metric_models import (
    AccuracyMetricNames,
    AccuracyMetricResults,
    CrossValidationMetricNames,
    CrossValidationMetricResults,
    PerturbationMetricNames,
    PerturbationMetricResults,
    RefreshStabilityMetricNames,
    RefreshStabilityMetricResults,
    TestResultDFAttributes,
)


class TestAccuracyMetricResults:
    """Test cases for accuracy metric results."""

    def test_accuracy_metric_threshold_checking_good_metrics(self):
        """Test accuracy metric threshold checking with good metrics."""
        results = AccuracyMetricResults(mape=0.1, r_squared=0.85)

        # Test individual metric threshold checking
        assert results._check_metric_threshold(AccuracyMetricNames.MAPE.value, 0.1) is True
        assert results._check_metric_threshold(AccuracyMetricNames.R_SQUARED.value, 0.85) is True

    def test_accuracy_metric_threshold_checking_bad_metrics(self):
        """Test accuracy metric threshold checking with bad metrics."""
        results = AccuracyMetricResults(mape=0.1, r_squared=0.85)

        # Test individual metric threshold checking
        assert results._check_metric_threshold(AccuracyMetricNames.MAPE.value, 0.2) is False
        assert results._check_metric_threshold(AccuracyMetricNames.R_SQUARED.value, 0.75) is False

    def test_accuracy_metric_dataframe_output(self):
        """Test accuracy metric results DataFrame output."""
        results = AccuracyMetricResults(mape=0.1, r_squared=0.85)
        df = results.to_df()

        # Check DataFrame structure
        expected_columns = TestResultDFAttributes.to_list()
        assert list(df.columns) == expected_columns
        assert len(df) == 2

        # Check metric values
        mape_row = df[df[TestResultDFAttributes.GENERAL_METRIC_NAME.value] == AccuracyMetricNames.MAPE.value].iloc[0]
        r_squared_row = df[
            df[TestResultDFAttributes.GENERAL_METRIC_NAME.value] == AccuracyMetricNames.R_SQUARED.value
        ].iloc[0]

        assert mape_row[TestResultDFAttributes.METRIC_VALUE.value] == 0.1
        assert r_squared_row[TestResultDFAttributes.METRIC_VALUE.value] == 0.85
        assert mape_row[TestResultDFAttributes.METRIC_PASS.value] == True  # noqa: E712
        assert r_squared_row[TestResultDFAttributes.METRIC_PASS.value] == True  # noqa: E712

    def test_accuracy_metric_invalid_metric_name(self):
        """Test accuracy metric with invalid metric name."""
        results = AccuracyMetricResults(mape=0.1, r_squared=0.85)

        with pytest.raises(InvalidMetricNameException):
            results._check_metric_threshold("invalid_metric", 0.1)


class TestCrossValidationMetricResults:
    """Test cases for cross-validation metric results."""

    def test_cross_validation_metric_threshold_checking_good_metrics(self):
        """Test cross-validation metric threshold checking with good metrics."""
        results = CrossValidationMetricResults(mean_mape=0.12, std_mape=0.02, mean_r_squared=0.85)

        # Test individual metric threshold checking
        assert results._check_metric_threshold(CrossValidationMetricNames.MEAN_MAPE.value, 0.12) is True
        assert results._check_metric_threshold(CrossValidationMetricNames.STD_MAPE.value, 0.02) is True
        assert results._check_metric_threshold(CrossValidationMetricNames.MEAN_R_SQUARED.value, 0.85) is True

    def test_cross_validation_metric_threshold_checking_bad_metrics(self):
        """Test cross-validation metric threshold checking with bad metrics."""
        results = CrossValidationMetricResults(mean_mape=0.12, std_mape=0.02, mean_r_squared=0.85)

        # Test individual metric threshold checking
        assert results._check_metric_threshold(CrossValidationMetricNames.MEAN_MAPE.value, 0.16) is False
        assert results._check_metric_threshold(CrossValidationMetricNames.MEAN_R_SQUARED.value, 0.75) is False

    def test_cross_validation_metric_dataframe_output(self):
        """Test cross-validation metric results DataFrame output."""
        results = CrossValidationMetricResults(mean_mape=0.12, std_mape=0.02, mean_r_squared=0.85)
        df = results.to_df()

        # Check DataFrame structure
        expected_columns = TestResultDFAttributes.to_list()
        assert list(df.columns) == expected_columns
        assert len(df) == 3

        # Check metric values
        mean_mape_row = df[df[TestResultDFAttributes.GENERAL_METRIC_NAME.value] == "mean_mape"].iloc[0]
        std_mape_row = df[df[TestResultDFAttributes.GENERAL_METRIC_NAME.value] == "std_mape"].iloc[0]
        mean_r_squared_row = df[df[TestResultDFAttributes.GENERAL_METRIC_NAME.value] == "mean_r_squared"].iloc[0]

        assert mean_mape_row[TestResultDFAttributes.METRIC_VALUE.value] == 0.12
        assert std_mape_row[TestResultDFAttributes.METRIC_VALUE.value] == 0.02
        assert mean_r_squared_row[TestResultDFAttributes.METRIC_VALUE.value] == 0.85
        assert mean_mape_row[TestResultDFAttributes.METRIC_PASS.value] == True  # noqa: E712
        assert std_mape_row[TestResultDFAttributes.METRIC_PASS.value] == True  # noqa: E712
        assert mean_r_squared_row[TestResultDFAttributes.METRIC_PASS.value] == True  # noqa: E712


class TestRefreshStabilityMetricResults:
    """Test cases for refresh stability metric results."""

    def test_stability_metric_threshold_checking_good_metrics(self):
        """Test stability metric threshold checking with good metrics."""
        mean_series = pd.Series({"channel_1": 0.1, "channel_2": 0.05})
        std_series = pd.Series({"channel_1": 0.02, "channel_2": 0.01})
        results = RefreshStabilityMetricResults(
            mean_percentage_change_for_each_channel=mean_series,
            std_percentage_change_for_each_channel=std_series,
        )

        # Test individual metric threshold checking
        assert results._check_metric_threshold(RefreshStabilityMetricNames.MEAN_PERCENTAGE_CHANGE.value, 0.1) is True
        assert results._check_metric_threshold(RefreshStabilityMetricNames.STD_PERCENTAGE_CHANGE.value, 0.02) is True

    def test_stability_metric_threshold_checking_bad_metrics(self):
        """Test stability metric threshold checking with bad metrics."""
        mean_series = pd.Series({"channel_1": 0.1, "channel_2": 0.05})
        std_series = pd.Series({"channel_1": 0.02, "channel_2": 0.01})
        results = RefreshStabilityMetricResults(
            mean_percentage_change_for_each_channel=mean_series,
            std_percentage_change_for_each_channel=std_series,
        )

        # Test individual metric threshold checking
        assert results._check_metric_threshold(RefreshStabilityMetricNames.MEAN_PERCENTAGE_CHANGE.value, 0.16) is False
        assert results._check_metric_threshold(RefreshStabilityMetricNames.STD_PERCENTAGE_CHANGE.value, 0.06) is False

    def test_stability_metric_dataframe_output(self):
        """Test stability metric results DataFrame output."""
        mean_series = pd.Series({"channel_1": 0.1, "channel_2": 0.05})
        std_series = pd.Series({"channel_1": 0.02, "channel_2": 0.01})
        results = RefreshStabilityMetricResults(
            mean_percentage_change_for_each_channel=mean_series,
            std_percentage_change_for_each_channel=std_series,
        )
        df = results.to_df()

        # Check DataFrame structure
        expected_columns = TestResultDFAttributes.to_list()
        assert list(df.columns) == expected_columns
        assert len(df) == 4  # 2 channels × 2 metrics

        # Check channel-specific metric values
        channel1_mean_row = df[
            df[TestResultDFAttributes.SPECIFIC_METRIC_NAME.value] == "mean_percentage_change_channel_1"
        ].iloc[0]
        channel2_std_row = df[
            df[TestResultDFAttributes.SPECIFIC_METRIC_NAME.value] == "std_percentage_change_channel_2"
        ].iloc[0]

        assert channel1_mean_row[TestResultDFAttributes.METRIC_VALUE.value] == 0.1
        assert channel2_std_row[TestResultDFAttributes.METRIC_VALUE.value] == 0.01
        assert channel1_mean_row[TestResultDFAttributes.METRIC_PASS.value] == True  # noqa: E712
        assert channel2_std_row[TestResultDFAttributes.METRIC_PASS.value] == True  # noqa: E712


class TestPerturbationMetricResults:
    """Test cases for perturbation metric results."""

    def test_perturbation_metric_threshold_checking_good_metrics(self):
        """Test perturbation metric threshold checking with good metrics."""
        percentage_change_series = pd.Series({"TV": 0.03, "Radio": 0.07})
        results = PerturbationMetricResults(percentage_change_for_each_channel=percentage_change_series)

        # Test individual metric threshold checking
        assert results._check_metric_threshold(PerturbationMetricNames.PERCENTAGE_CHANGE.value, 0.03) is True
        assert results._check_metric_threshold(PerturbationMetricNames.PERCENTAGE_CHANGE.value, 0.07) is True

    def test_perturbation_metric_threshold_checking_bad_metrics(self):
        """Test perturbation metric threshold checking with bad metrics."""
        percentage_change_series = pd.Series({"TV": 0.03, "Radio": 0.07})
        results = PerturbationMetricResults(percentage_change_for_each_channel=percentage_change_series)

        # Test individual metric threshold checking
        assert results._check_metric_threshold(PerturbationMetricNames.PERCENTAGE_CHANGE.value, 0.11) is False

    def test_perturbation_metric_dataframe_output(self):
        """Test perturbation metric results DataFrame output."""
        percentage_change_series = pd.Series({"TV": 0.03, "Radio": 0.07})
        results = PerturbationMetricResults(percentage_change_for_each_channel=percentage_change_series)
        df = results.to_df()

        # Check DataFrame structure
        expected_columns = TestResultDFAttributes.to_list()
        assert list(df.columns) == expected_columns
        assert len(df) == 2  # 2 channels

        # Check channel-specific metric values
        tv_row = df[df[TestResultDFAttributes.SPECIFIC_METRIC_NAME.value] == "percentage_change_TV"].iloc[0]
        radio_row = df[df[TestResultDFAttributes.SPECIFIC_METRIC_NAME.value] == "percentage_change_Radio"].iloc[0]

        assert tv_row[TestResultDFAttributes.METRIC_VALUE.value] == 0.03
        assert radio_row[TestResultDFAttributes.METRIC_VALUE.value] == 0.07
        assert tv_row[TestResultDFAttributes.METRIC_PASS.value] == True  # noqa: E712
        assert radio_row[TestResultDFAttributes.METRIC_PASS.value] == True  # noqa: E712
