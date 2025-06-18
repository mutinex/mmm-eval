"""
Unit tests for accuracy functions.
"""

import pytest
import numpy as np
import pandas as pd
from mmm_eval.metrics.accuracy_functions import (
    calculate_mean_for_cross_validation_folds,
    calculate_std_for_cross_validation_folds,
    calculate_absolute_percentage_change,
)
from mmm_eval.metrics.metric_models import (
    AccuracyMetricResults,
    RefreshStabilityMetricResults,
    AccuracyMetricNames,
    RefreshStabilityMetricNames,
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

    def test_calculate_mean_for_cross_validation_folds(self):
        """Test mean calculation across folds."""
        fold_metrics = [
            AccuracyMetricResults(mape=0.1, r_squared=0.8),
            AccuracyMetricResults(mape=0.2, r_squared=0.7),
            AccuracyMetricResults(mape=0.3, r_squared=0.9),
        ]
        
        result = calculate_mean_for_cross_validation_folds(
            fold_metrics, AccuracyMetricNames.MAPE
        )
        
        expected = (0.1 + 0.2 + 0.3) / 3
        assert result == expected
        assert isinstance(result, float)

    def test_calculate_std_for_cross_validation_folds(self):
        """Test standard deviation calculation across folds."""
        fold_metrics = [
            RefreshStabilityMetricResults(mean_percentage_change=0.1, std_percentage_change=0.05),
            RefreshStabilityMetricResults(mean_percentage_change=0.2, std_percentage_change=0.05),
            RefreshStabilityMetricResults(mean_percentage_change=0.3, std_percentage_change=0.05),
        ]
        
        result = calculate_std_for_cross_validation_folds(
            fold_metrics, RefreshStabilityMetricNames.MEAN_PERCENTAGE_CHANGE
        )
        
        # Expected std of [0.1, 0.2, 0.3]
        expected = np.std([0.1, 0.2, 0.3])
        assert abs(result - expected) < 1e-10
        assert isinstance(result, float) 