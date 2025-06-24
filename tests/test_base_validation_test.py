"""Unit tests for base validation test utilities."""

from unittest.mock import Mock

import pandas as pd
import pytest

from mmm_eval.core.base_validation_test import BaseValidationTest
from mmm_eval.core.constants import ValidationDataframeConstants
from mmm_eval.core.exceptions import DataValidationError, MetricCalculationError
from mmm_eval.data.constants import InputDataframeConstants


class ConcreteTestClass(BaseValidationTest):
    """Concrete implementation for testing abstract base class."""

    @property
    def test_name(self) -> str:
        """Return the name of the test."""
        return "concrete_test"

    def run(self, adapter, data):
        """Run the test."""
        return Mock()  # Mock result for testing


class TestBaseValidationTest:
    """Test cases for base validation test utilities."""

    def setup_method(self):
        """Set up test data."""
        self.test_instance = ConcreteTestClass()

        # Create comprehensive test data (21 points - minimum for time series CV)
        # This serves as the lowest common denominator for all tests
        self.test_data = pd.DataFrame(
            {
                InputDataframeConstants.DATE_COL: pd.date_range("2023-01-01", periods=21),
                "media_channel": ["TV", "Radio", "Digital"] * 7,
                InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL: [110, 220, 330] * 7,
                InputDataframeConstants.RESPONSE_COL: [160, 260, 360] * 7,
                "volume_contribution": [55, 55, 55] * 7,
                ValidationDataframeConstants.PERCENTAGE_CHANGE_CHANNEL_CONTRIBUTION_COL: [
                    0.1,
                    0.2,
                    0.3,
                ]
                * 7,
            }
        )

        # Create insufficient data for time series CV error testing (10 points)
        self.insufficient_data = pd.DataFrame(
            {
                InputDataframeConstants.DATE_COL: pd.date_range("2023-01-01", periods=10),
                "media_channel": ["TV", "Radio"] * 5,
                InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL: [100, 200] * 5,
                InputDataframeConstants.RESPONSE_COL: [150, 250] * 5,
                "volume_contribution": [50, 50] * 5,
            }
        )

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
        # The error occurs when we try to call the method
        with pytest.raises(ValueError, match="Too many splits"):
            self.test_instance._split_data_time_series_cv(self.insufficient_data)

    def test_split_data_time_series_cv_minimum_data(self):
        """Test time series cross-validation with exactly minimum required data."""
        # Should not raise an error
        cv_splits = self.test_instance._split_data_time_series_cv(self.test_data)
        splits_list = list(cv_splits)

        # Should get exactly 5 splits
        assert len(splits_list) == 5

    def test_run_with_error_handling_success(self):
        """Test successful error handling."""
        mock_model = Mock()
        mock_data = pd.DataFrame({"test": [1, 2, 3]})

        # Mock the run method to return a successful result
        self.test_instance.run = Mock(return_value=Mock())

        result = self.test_instance.run_with_error_handling(mock_model, mock_data)

        assert result is not None
        self.test_instance.run.assert_called_once_with(mock_model, mock_data)

    def test_run_with_error_handling_key_error(self):
        """Test error handling for KeyError."""
        mock_model = Mock()
        mock_data = pd.DataFrame({"test": [1, 2, 3]})

        # Mock the run method to raise KeyError
        self.test_instance.run = Mock(side_effect=KeyError("Missing column"))

        with pytest.raises(DataValidationError, match="Data validation error"):
            self.test_instance.run_with_error_handling(mock_model, mock_data)

    def test_run_with_error_handling_value_error(self):
        """Test error handling for ValueError."""
        mock_model = Mock()
        mock_data = pd.DataFrame({"test": [1, 2, 3]})

        # Mock the run method to raise ValueError
        self.test_instance.run = Mock(side_effect=ValueError("Invalid input"))

        with pytest.raises(DataValidationError, match="Data validation error"):
            self.test_instance.run_with_error_handling(mock_model, mock_data)

    def test_run_with_error_handling_type_error(self):
        """Test error handling for TypeError."""
        mock_model = Mock()
        mock_data = pd.DataFrame({"test": [1, 2, 3]})

        # Mock the run method to raise TypeError
        self.test_instance.run = Mock(side_effect=TypeError("Invalid type"))

        with pytest.raises(MetricCalculationError, match="Metric calculation error"):
            self.test_instance.run_with_error_handling(mock_model, mock_data)
