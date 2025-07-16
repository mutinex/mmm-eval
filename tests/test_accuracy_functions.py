"""Unit tests for accuracy functions."""

import numpy as np
import pandas as pd
import pytest

from mmm_eval.metrics.accuracy_functions import (
    calculate_absolute_percentage_change,
    calculate_mean_for_singular_values_across_cross_validation_folds,
    calculate_means_for_series_across_cross_validation_folds,
    calculate_std_for_singular_values_across_cross_validation_folds,
    calculate_stds_for_series_across_cross_validation_folds,
)
from mmm_eval.metrics.metric_models import (
    AccuracyMetricNames,
    AccuracyMetricResults,
    calculate_smape,
)


class TestCalculateAbsolutePercentageChange:
    """Test cases for absolute percentage change calculation."""

    def test_no_change(self):
        """Test with no change (should return 0)."""
        baseline = pd.Series([100, 200, 300])
        comparison = pd.Series([100, 200, 300])

        result = calculate_absolute_percentage_change(baseline, comparison)

        assert all(result == 0.0)
        assert isinstance(result, pd.Series)

    def test_constant_percentage_increase(self):
        """Test with constant percentage increase."""
        baseline = pd.Series([100, 200, 300])
        comparison = pd.Series([110, 220, 330])  # 10% increase

        result = calculate_absolute_percentage_change(baseline, comparison)

        assert all(result == 0.1)  # 10% absolute change
        assert isinstance(result, pd.Series)

    def test_zero_baseline_handling(self):
        """Test handling of zero values in baseline."""
        baseline = pd.Series([0, 100, 200])
        comparison = pd.Series([10, 110, 210])

        result = calculate_absolute_percentage_change(baseline, comparison)

        # Check that the first element is inf (division by zero)
        assert np.isinf(result.iloc[0])
        # Check that other elements are calculated correctly
        assert result.iloc[1] == pytest.approx(0.1)  # (110-100)/100 = 0.1
        assert result.iloc[2] == pytest.approx(0.05)  # (210-200)/200 = 0.05


class TestCrossValidationFoldCalculations:
    """Test cases for cross-validation fold calculations."""

    def test_calculate_mean_for_singular_values_across_cross_validation_folds(self):
        """Test mean calculation across folds for single values."""
        fold_metrics = [
            AccuracyMetricResults(mape=0.1, smape=0.095, r_squared=0.8),
            AccuracyMetricResults(mape=0.2, smape=0.105, r_squared=0.7),
            AccuracyMetricResults(mape=0.3, smape=0.115, r_squared=0.9),
        ]

        result = calculate_mean_for_singular_values_across_cross_validation_folds(
            fold_metrics, AccuracyMetricNames.MAPE
        )

        expected = (0.1 + 0.2 + 0.3) / 3
        assert result == expected
        assert isinstance(result, float)

    def test_calculate_std_for_singular_values_across_cross_validation_folds(self):
        """Test standard deviation calculation across folds for single values."""
        fold_metrics = [
            AccuracyMetricResults(mape=0.1, smape=0.095, r_squared=0.8),
            AccuracyMetricResults(mape=0.2, smape=0.105, r_squared=0.7),
            AccuracyMetricResults(mape=0.3, smape=0.115, r_squared=0.9),
        ]

        result = calculate_std_for_singular_values_across_cross_validation_folds(fold_metrics, AccuracyMetricNames.MAPE)

        # Expected std of [0.1, 0.2, 0.3]
        expected = np.std([0.1, 0.2, 0.3])
        assert abs(result - expected) < 1e-10
        assert isinstance(result, float)

    def test_calculate_means_for_series_across_cross_validation_folds(self):
        """Test mean calculation across folds for pandas Series."""
        fold_series = [
            pd.Series({"channel_1": 0.1, "channel_2": 0.2}),
            pd.Series({"channel_1": 0.2, "channel_2": 0.3}),
            pd.Series({"channel_1": 0.3, "channel_2": 0.4}),
        ]

        result = calculate_means_for_series_across_cross_validation_folds(fold_series)

        # Expected means: channel_1 = (0.1+0.2+0.3)/3 = 0.2, channel_2 = (0.2+0.3+0.4)/3 = 0.3
        assert result["channel_1"] == pytest.approx(0.2)
        assert result["channel_2"] == pytest.approx(0.3)
        assert isinstance(result, pd.Series)

    def test_calculate_stds_for_series_across_cross_validation_folds(self):
        """Test standard deviation calculation across folds for pandas Series."""
        fold_series = [
            pd.Series({"channel_1": 0.1, "channel_2": 0.2}),
            pd.Series({"channel_1": 0.2, "channel_2": 0.3}),
            pd.Series({"channel_1": 0.3, "channel_2": 0.4}),
        ]

        result = calculate_stds_for_series_across_cross_validation_folds(fold_series)

        # Expected stds: channel_1 = std([0.1,0.2,0.3]), channel_2 = std([0.2,0.3,0.4])
        # pandas uses ddof=1 by default (sample std), numpy uses ddof=0 by default (population std)
        expected_channel_1 = np.std([0.1, 0.2, 0.3], ddof=1)  # Use sample std to match pandas
        expected_channel_2 = np.std([0.2, 0.3, 0.4], ddof=1)  # Use sample std to match pandas
        assert result["channel_1"] == pytest.approx(expected_channel_1)
        assert result["channel_2"] == pytest.approx(expected_channel_2)
        assert isinstance(result, pd.Series)


class TestCalculateSMAPE:
    """Test cases for Symmetric Mean Absolute Percentage Error (SMAPE) calculation."""

    def test_perfect_predictions(self):
        """Test SMAPE with perfect predictions (should be 0)."""
        actual = pd.Series([100, 200, 300])
        predicted = pd.Series([100, 200, 300])

        result = calculate_smape(actual, predicted)

        assert result == 0.0
        assert isinstance(result, float)

    def test_constant_percentage_error_overestimation(self):
        """Test SMAPE with constant percentage overestimation."""
        actual = pd.Series([100, 200, 300])
        predicted = pd.Series([110, 220, 330])  # 10% overestimation

        result = calculate_smape(actual, predicted)

        # Expected SMAPE for 10% overestimation
        # SMAPE = 100 * (2 * |actual - predicted|) / (|actual| + |predicted|)
        # For 10% overestimation: (2 * 10) / (100 + 110) = 20/210 ≈ 9.52%
        expected = 100 * (2 * 10) / (100 + 110)
        assert result == pytest.approx(expected, rel=1e-10)
        assert isinstance(result, float)

    def test_constant_percentage_error_underestimation(self):
        """Test SMAPE with constant percentage underestimation."""
        actual = pd.Series([100, 200, 300])
        predicted = pd.Series([90, 180, 270])  # 10% underestimation

        result = calculate_smape(actual, predicted)

        # Expected SMAPE for 10% underestimation
        # SMAPE = 100 * (2 * |actual - predicted|) / (|actual| + |predicted|)
        # For 10% underestimation: (2 * 10) / (100 + 90) = 20/190 ≈ 10.53%
        expected = 100 * (2 * 10) / (100 + 90)
        assert result == pytest.approx(expected, rel=1e-10)
        assert isinstance(result, float)

    def test_mixed_errors(self):
        """Test SMAPE with mixed over and underestimation errors."""
        actual = pd.Series([100, 200, 300])
        predicted = pd.Series([110, 180, 330])  # Mixed errors

        result = calculate_smape(actual, predicted)

        # Calculate expected manually
        # Error 1: (2 * 10) / (100 + 110) = 20/210 ≈ 0.0952
        # Error 2: (2 * 20) / (200 + 180) = 40/380 ≈ 0.1053
        # Error 3: (2 * 30) / (300 + 330) = 60/630 ≈ 0.0952
        # Average: (0.0952 + 0.1053 + 0.0952) / 3 ≈ 0.0986
        expected = 100 * ((2 * 10) / (100 + 110) + (2 * 20) / (200 + 180) + (2 * 30) / (300 + 330)) / 3
        assert result == pytest.approx(expected, rel=1e-10)
        assert isinstance(result, float)

    def test_zero_values_handling(self):
        """Test SMAPE handling of zero values."""
        actual = pd.Series([0, 100, 200])
        predicted = pd.Series([10, 110, 210])

        result = calculate_smape(actual, predicted)

        # For zero actual value, denominator becomes |predicted| to avoid division by zero
        # Error 1: (2 * 10) / (0 + 10) = 20/10 = 2.0
        # Error 2: (2 * 10) / (100 + 110) = 20/210 ≈ 0.0952
        # Error 3: (2 * 10) / (200 + 210) = 20/410 ≈ 0.0488
        # Average: (2.0 + 0.0952 + 0.0488) / 3 ≈ 0.7147
        expected = 100 * ((2 * 10) / (0 + 10) + (2 * 10) / (100 + 110) + (2 * 10) / (200 + 210)) / 3
        assert result == pytest.approx(expected, rel=1e-10)
        assert isinstance(result, float)

    def test_both_zero_values(self):
        """Test SMAPE when both actual and predicted are zero."""
        actual = pd.Series([0, 100, 0])
        predicted = pd.Series([0, 110, 0])

        result = calculate_smape(actual, predicted)

        # When both actual and predicted are zero, denominator becomes 1 to avoid division by zero
        # Error 1: (2 * 0) / 1 = 0
        # Error 2: (2 * 10) / (100 + 110) = 20/210 ≈ 0.0952
        # Error 3: (2 * 0) / 1 = 0
        # Average: (0 + 0.0952 + 0) / 3 ≈ 0.0317
        expected = 100 * (0 + (2 * 10) / (100 + 110) + 0) / 3
        assert result == pytest.approx(expected, rel=1e-10)
        assert isinstance(result, float)

    def test_single_value(self):
        """Test SMAPE with single value."""
        actual = pd.Series([100])
        predicted = pd.Series([110])

        result = calculate_smape(actual, predicted)

        expected = 100 * (2 * 10) / (100 + 110)
        assert result == pytest.approx(expected, rel=1e-10)
        assert isinstance(result, float)

    def test_large_numbers(self):
        """Test SMAPE with large numbers."""
        actual = pd.Series([1000000, 2000000, 3000000])
        predicted = pd.Series([1100000, 2200000, 3300000])  # 10% overestimation

        result = calculate_smape(actual, predicted)

        # Should give same percentage as smaller numbers
        expected = 100 * (2 * 100000) / (1000000 + 1100000)
        assert result == pytest.approx(expected, rel=1e-10)
        assert isinstance(result, float)

    def test_negative_values(self):
        """Test SMAPE with negative values."""
        actual = pd.Series([-100, -200, -300])
        predicted = pd.Series([-110, -220, -330])  # 10% overestimation

        result = calculate_smape(actual, predicted)

        # SMAPE should work with absolute values
        expected = 100 * (2 * 10) / (100 + 110)
        assert result == pytest.approx(expected, rel=1e-10)
        assert isinstance(result, float)

    def test_empty_series(self):
        """Test SMAPE with empty series (should raise error)."""
        actual = pd.Series([])
        predicted = pd.Series([])

        with pytest.raises(ValueError):
            calculate_smape(actual, predicted)

    def test_different_lengths(self):
        """Test SMAPE with different length series (should raise error)."""
        actual = pd.Series([100, 200])
        predicted = pd.Series([110, 220, 330])

        with pytest.raises(ValueError):
            calculate_smape(actual, predicted)

    def test_nan_values(self):
        """Test SMAPE with NaN values."""
        actual = pd.Series([100, np.nan, 300])
        predicted = pd.Series([110, 220, np.nan])

        result = calculate_smape(actual, predicted)

        # Should handle NaN values gracefully
        assert np.isnan(result)
        assert isinstance(result, float)
