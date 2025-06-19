"""
Unit tests for base validation test utilities.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from mmm_eval.core.base_validation_test import BaseValidationTest
from mmm_eval.core.constants import ValidationDataframeConstants
from mmm_eval.data.input_dataframe_constants import InputDataframeConstants
from mmm_eval.core.exceptions import MetricCalculationError
from mmm_eval.core.exceptions import DataValidationError


class ConcreteTestClass(BaseValidationTest):
    """Concrete implementation for testing abstract base class."""
    
    @property
    def test_name(self) -> str:
        return "concrete_test"
    
    def run(self, model, data):
        return Mock()  # Mock result for testing


class TestBaseValidationTest:
    """Test cases for base validation test utilities."""

    def setup_method(self):
        """Set up test data."""
        self.test_instance = ConcreteTestClass()
        
        # Create comprehensive test data (21 points - minimum for time series CV)
        # This serves as the lowest common denominator for all tests
        self.test_data = pd.DataFrame({
            InputDataframeConstants.DATE_COL: pd.date_range('2023-01-01', periods=21),
            InputDataframeConstants.MEDIA_CHANNEL_COL: ['TV', 'Radio', 'Digital'] * 7,
            InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL: [110, 220, 330] * 7,
            InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL: [160, 260, 360] * 7,
            InputDataframeConstants.MEDIA_CHANNEL_VOLUME_CONTRIBUTION_COL: [55, 55, 55] * 7,
            ValidationDataframeConstants.PERCENTAGE_CHANGE_CHANNEL_CONTRIBUTION_COL: [0.1, 0.2, 0.3] * 7,
        })
        
        # Create insufficient data for time series CV error testing (10 points)
        self.insufficient_data = pd.DataFrame({
            InputDataframeConstants.DATE_COL: pd.date_range('2023-01-01', periods=10),
            InputDataframeConstants.MEDIA_CHANNEL_COL: ['TV', 'Radio'] * 5,
            InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL: [100, 200] * 5,
            InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL: [150, 250] * 5,
            InputDataframeConstants.MEDIA_CHANNEL_VOLUME_CONTRIBUTION_COL: [50, 50] * 5,
        })

    def test_split_data_holdout(self):
        """Test holdout data splitting."""
        train, test = self.test_instance._split_data_holdout(self.test_data)
        
        # Check that data is split
        assert len(train) + len(test) == len(self.test_data)
        assert len(train) > 0
        assert len(test) > 0
        
        # Check that train and test are disjoint
        train_indices = set(train.index)
        test_indices = set(test.index)
        assert train_indices.isdisjoint(test_indices)

    def test_split_data_time_series_cv(self):
        """Test time series cross-validation splitting."""
        cv_splits = self.test_instance._split_data_time_series_cv(self.test_data)
        splits_list = list(cv_splits)
        
        # Check that we get the expected number of splits
        assert len(splits_list) == 5  # Default N_SPLITS
        
        # Check that each split has train and test indices
        for train_idx, test_idx in splits_list:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert max(train_idx) < min(test_idx)  # Time series order

    def test_split_data_time_series_cv_insufficient_data(self):
        """Test time series cross-validation with insufficient data."""
        # Should raise sklearn's ValueError with informative message
        # The error occurs when we try to iterate over the generator
        cv_splits = self.test_instance._split_data_time_series_cv(self.insufficient_data)
        with pytest.raises(ValueError, match="Too many splits"):
            list(cv_splits)  # This triggers the error

    def test_split_data_time_series_cv_minimum_data(self):
        """Test time series cross-validation with exactly minimum required data."""
        # Should not raise an error
        cv_splits = self.test_instance._split_data_time_series_cv(self.test_data)
        splits_list = list(cv_splits)
        
        # Should get exactly 5 splits
        assert len(splits_list) == 5

    def test_add_calculated_roi_column(self):
        """Test ROI calculation."""
        result = self.test_instance._add_calculated_roi_column(self.test_data)
        
        # Check that ROI column was added
        assert ValidationDataframeConstants.CALCULATED_ROI_COL in result.columns
        
        # Check ROI calculations: (revenue - spend) / spend
        # First 3 rows: TV(160-110)/110=0.455, Radio(260-220)/220=0.182, Digital(360-330)/330=0.091
        expected_roi = [(160 - 110) / 110, (260 - 220) / 220, (360 - 330) / 330]  # [0.455, 0.182, 0.091]
        assert list(result[ValidationDataframeConstants.CALCULATED_ROI_COL][:3]) == expected_roi

    def test_aggregate_by_channel_and_sum(self):
        """Test channel aggregation."""
        result = self.test_instance._aggregate_by_channel_and_sum(self.test_data)
        
        # Check aggregation
        assert len(result) == 3  # 3 unique channels
        
        # Check TV aggregation (7 rows of TV with spend=110 each = 770 total)
        tv_row = result[result[InputDataframeConstants.MEDIA_CHANNEL_COL] == 'TV'].iloc[0]
        assert tv_row[InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL] == 770  # 110 * 7
        assert tv_row[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL] == 1120  # 160 * 7

    def test_combine_dataframes_by_channel(self):
        """Test dataframe combination by channel."""
        # Create a simpler comparison dataset with just the essential columns
        comparison_df = pd.DataFrame({
            InputDataframeConstants.MEDIA_CHANNEL_COL: ['TV', 'Radio', 'Digital'],
            'spend': [110, 220, 330],
            'revenue': [160, 260, 360],
        })
        
        # Create a baseline dataset with just the essential columns
        baseline_df = pd.DataFrame({
            InputDataframeConstants.MEDIA_CHANNEL_COL: ['TV', 'Radio', 'Digital'],
            'spend': [100, 200, 300],
            'revenue': [150, 250, 350],
        })
        
        result = self.test_instance._combine_dataframes_by_channel(
            baseline_df, comparison_df, suffixes=("_original", "_perturbed")
        )
        
        # Check that dataframes are combined
        assert len(result) == 3  # 3 channels
        assert 'spend_original' in result.columns
        assert 'spend_perturbed' in result.columns
        
        # Check TV values
        tv_row = result[result[InputDataframeConstants.MEDIA_CHANNEL_COL] == 'TV'].iloc[0]
        assert tv_row['spend_original'] == 100
        assert tv_row['spend_perturbed'] == 110

    def test_get_mean_aggregate_channel_roi_pct_change(self):
        """Test mean ROI percentage change calculation."""
        result = self.test_instance._get_mean_aggregate_channel_roi_pct_change(self.test_data)
        
        # Expected mean of [0.1, 0.2, 0.3] repeated 7 times = 0.2
        expected_mean = 0.2
        assert result == expected_mean
        assert isinstance(result, float)

    def test_run_with_error_handling_success(self):
        """Test successful error handling."""
        mock_model = Mock()
        mock_data = pd.DataFrame({'test': [1, 2, 3]})
        
        # Mock the run method to return a successful result
        self.test_instance.run = Mock(return_value=Mock())
        
        result = self.test_instance.run_with_error_handling(mock_model, mock_data)
        
        assert result is not None
        self.test_instance.run.assert_called_once_with(mock_model, mock_data)

    def test_run_with_error_handling_key_error(self):
        """Test error handling for KeyError."""
        mock_model = Mock()
        mock_data = pd.DataFrame({'test': [1, 2, 3]})
        
        # Mock the run method to raise KeyError
        self.test_instance.run = Mock(side_effect=KeyError("Missing column"))
        
        with pytest.raises(DataValidationError, match="Missing required columns"):
            self.test_instance.run_with_error_handling(mock_model, mock_data)

    def test_run_with_error_handling_value_error(self):
        """Test error handling for ValueError."""
        mock_model = Mock()
        mock_data = pd.DataFrame({'test': [1, 2, 3]})
        
        # Mock the run method to raise ValueError
        self.test_instance.run = Mock(side_effect=ValueError("Invalid input"))
        
        with pytest.raises(MetricCalculationError, match="Invalid metric input"):
            self.test_instance.run_with_error_handling(mock_model, mock_data) 