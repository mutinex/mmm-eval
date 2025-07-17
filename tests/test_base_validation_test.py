"""Unit tests for base validation test utilities."""

from unittest.mock import Mock

import pandas as pd
import pytest

from mmm_eval.core.base_validation_test import (
    BaseValidationTest,
    split_timeseries_cv,
    split_timeseries_data,
)
from mmm_eval.core.constants import ValidationTestConstants
from mmm_eval.core.exceptions import MetricCalculationError, TestExecutionError
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
        self.test_instance = ConcreteTestClass(InputDataframeConstants.DATE_COL)

        # Calculate minimum required data size based on constants
        # Required: test_size * (n_splits + 1) = TIME_SERIES_CROSS_VALIDATION_TEST_SIZE * (N_SPLITS + 1)
        min_required_size = ValidationTestConstants.TIME_SERIES_CROSS_VALIDATION_TEST_SIZE * (
            ValidationTestConstants.N_SPLITS + 1
        )

        # Create comprehensive test data (minimum for data validation)
        # This serves as the lowest common denominator for all tests
        self.test_data = pd.DataFrame(
            {
                InputDataframeConstants.DATE_COL: pd.date_range("2023-01-01", periods=min_required_size),
                "media_channel": ["TV", "Radio"] * (min_required_size // 2),
                InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL: [110, 220] * (min_required_size // 2),
                InputDataframeConstants.RESPONSE_COL: [160, 260] * (min_required_size // 2),
            }
        )
        # Trim to exactly the required size
        self.test_data = self.test_data.head(min_required_size)

        # Create insufficient data for time series CV error testing (10 points)
        self.insufficient_data = pd.DataFrame(
            {
                InputDataframeConstants.DATE_COL: pd.date_range("2023-01-01", periods=10),
                "media_channel": ["TV", "Radio"] * 5,
                InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL: [100, 200] * 5,
                InputDataframeConstants.RESPONSE_COL: [150, 250] * 5,
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
        assert len(splits_list) == ValidationTestConstants.N_SPLITS

        # Check that each split has train and test masks
        for i, (train_mask, test_mask) in enumerate(splits_list):
            assert isinstance(train_mask, pd.Series)
            assert isinstance(test_mask, pd.Series)
            assert train_mask.dtype == bool
            assert test_mask.dtype == bool
            assert len(train_mask) > 0
            assert len(test_mask) > 0

            # Check that train and test are disjoint
            assert not (train_mask & test_mask).any()

            # Check that test size is correct
            assert test_mask.sum() == ValidationTestConstants.TIME_SERIES_CROSS_VALIDATION_TEST_SIZE

            # Check that train size decreases with each split (rolling window behavior)
            total_data_size = len(self.test_data)
            test_size = ValidationTestConstants.TIME_SERIES_CROSS_VALIDATION_TEST_SIZE
            expected_train_size = total_data_size - test_size * (i + 1)
            assert train_mask.sum() == expected_train_size

    def test_split_data_time_series_cv_insufficient_data(self):
        """Test time series cross-validation with insufficient data."""
        # Should raise ValueError with informative message
        with pytest.raises(ValueError, match="Insufficient timeseries data provided for splitting"):
            self.test_instance._split_data_time_series_cv(self.insufficient_data)

    def test_split_data_time_series_cv_minimum_data(self):
        """Test time series cross-validation with exactly minimum required data."""
        # Should not raise an error
        cv_splits = self.test_instance._split_data_time_series_cv(self.test_data)
        splits_list = list(cv_splits)

        # Should get exactly the expected number of splits
        assert len(splits_list) == ValidationTestConstants.N_SPLITS

    def test_split_timeseries_data_basic(self):
        """Test basic functionality of split_timeseries_data."""
        # Create test data with 10 unique dates
        dates = pd.date_range("2023-01-01", periods=10)
        data = pd.DataFrame({InputDataframeConstants.DATE_COL: dates, "value": range(10)})

        train_mask, test_mask = split_timeseries_data(data, 3, InputDataframeConstants.DATE_COL)

        # Check that masks are boolean arrays
        assert isinstance(train_mask, pd.Series)
        assert isinstance(test_mask, pd.Series)
        assert train_mask.dtype == bool
        assert test_mask.dtype == bool

        # Check that train and test are disjoint
        assert not (train_mask & test_mask).any()

        # Check that all data is covered
        assert (train_mask | test_mask).all()

        # Check proportions (7 train, 3 test for test_size=3)
        assert train_mask.sum() == 7
        assert test_mask.sum() == 3

    def test_split_timeseries_data_exact_test_size(self):
        """Test that split_timeseries_data reserves exactly test_size dates for testing."""
        # Create test data with 10 unique dates
        dates = pd.date_range("2023-01-01", periods=10)
        data = pd.DataFrame({InputDataframeConstants.DATE_COL: dates, "value": range(10)})

        # Test with different test sizes
        for test_size in [1, 3, 5, 7]:
            train_mask, test_mask = split_timeseries_data(data, test_size, InputDataframeConstants.DATE_COL)

            # Count unique dates in test set
            test_dates = data[test_mask][InputDataframeConstants.DATE_COL].unique()
            assert len(test_dates) == test_size, f"Expected {test_size} test dates, got {len(test_dates)}"

            # Verify train and test are disjoint
            train_dates = data[train_mask][InputDataframeConstants.DATE_COL].unique()
            assert len(set(train_dates) & set(test_dates)) == 0

            # Verify all data is covered
            assert len(train_dates) + len(test_dates) == len(dates)

    def test_split_timeseries_data_edge_cases(self):
        """Test edge cases for split_timeseries_data."""
        dates = pd.date_range("2023-01-01", periods=5)
        data = pd.DataFrame({InputDataframeConstants.DATE_COL: dates, "value": range(5)})

        # Test with very small test size
        train_mask, test_mask = split_timeseries_data(data, 1, InputDataframeConstants.DATE_COL)
        assert train_mask.sum() == 4
        assert test_mask.sum() == 1

        # Test with large test size
        train_mask, test_mask = split_timeseries_data(data, 4, InputDataframeConstants.DATE_COL)
        # With test_size=4 and 5 dates: cutoff = dates[-4] = '2023-01-02'
        # So 1 date < '2023-01-02' and 4 dates >= '2023-01-02'
        assert train_mask.sum() == 1  # 1 date < '2023-01-02'
        assert test_mask.sum() == 4  # 4 dates >= '2023-01-02'

    def test_split_timeseries_data_duplicate_dates(self):
        """Test split_timeseries_data with duplicate dates."""
        # Create data with duplicate dates
        dates = pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03"])
        data = pd.DataFrame({InputDataframeConstants.DATE_COL: dates, "value": range(5)})

        train_mask, test_mask = split_timeseries_data(data, 1, InputDataframeConstants.DATE_COL)

        # Should split based on unique dates (3 unique dates, test_size=1)
        unique_dates = data[InputDataframeConstants.DATE_COL].unique()
        assert len(unique_dates) == 3

        # Check that all rows with the same date go to the same split
        train_dates = data[train_mask][InputDataframeConstants.DATE_COL].unique()
        test_dates = data[test_mask][InputDataframeConstants.DATE_COL].unique()
        assert len(set(train_dates) & set(test_dates)) == 0

    def test_split_timeseries_cv_basic(self):
        """Test basic functionality of split_timeseries_cv."""
        # Create test data with 20 unique dates (enough for 3 splits with test_size=4)
        dates = pd.date_range("2023-01-01", periods=20)
        data = pd.DataFrame({InputDataframeConstants.DATE_COL: dates, "value": range(20)})

        splits = list(split_timeseries_cv(data, n_splits=3, test_size=4, date_column=InputDataframeConstants.DATE_COL))

        # Should get 3 splits
        assert len(splits) == 3

        # Check each split
        for i, (train_mask, test_mask) in enumerate(splits):
            assert isinstance(train_mask, pd.Series)
            assert isinstance(test_mask, pd.Series)
            assert train_mask.dtype == bool
            assert test_mask.dtype == bool

            # Train and test should be disjoint
            assert not (train_mask & test_mask).any()

            # Test size should be 4
            assert test_mask.sum() == 4

            # Train size should decrease with each split
            if i == 0:
                expected_train_size = 16  # 20 - 4
            elif i == 1:
                expected_train_size = 12  # 20 - 8
            elif i == 2:
                expected_train_size = 8  # 20 - 12
            else:
                raise AssertionError(f"Unexpected split index: {i}")

            assert train_mask.sum() == expected_train_size

    def test_split_timeseries_cv_insufficient_data(self):
        """Test split_timeseries_cv with insufficient data."""
        # Create data with only 8 dates (not enough for 3 splits with test_size=4)
        # Required: test_size * (n_splits + 1) = 4 * (3 + 1) = 16 dates
        dates = pd.date_range("2023-01-01", periods=8)
        data = pd.DataFrame({InputDataframeConstants.DATE_COL: dates, "value": range(8)})

        with pytest.raises(ValueError, match="Insufficient timeseries data provided for splitting"):
            list(split_timeseries_cv(data, n_splits=3, test_size=4, date_column=InputDataframeConstants.DATE_COL))

    def test_split_timeseries_cv_exact_minimum_data(self):
        """Test split_timeseries_cv with exactly minimum required data."""
        # Create data with exactly 16 dates (minimum for 3 splits with test_size=4)
        dates = pd.date_range("2023-01-01", periods=16)
        data = pd.DataFrame({InputDataframeConstants.DATE_COL: dates, "value": range(16)})

        splits = list(split_timeseries_cv(data, n_splits=3, test_size=4, date_column=InputDataframeConstants.DATE_COL))

        # Should get exactly 3 splits
        assert len(splits) == 3

        # Check the splits are correct
        expected_train_sizes = [12, 8, 4]  # 16-4, 16-8, 16-12
        for i, (train_mask, test_mask) in enumerate(splits):
            assert train_mask.sum() == expected_train_sizes[i]
            assert test_mask.sum() == 4

    def test_split_timeseries_cv_duplicate_dates(self):
        """Test split_timeseries_cv with duplicate dates."""
        # Create data with duplicate dates but enough unique dates for splits
        base_dates = pd.date_range("2023-01-01", periods=8)
        dates = []
        for date in base_dates:
            dates.extend([date, date])  # Duplicate each date

        data = pd.DataFrame({InputDataframeConstants.DATE_COL: dates, "value": range(len(dates))})

        splits = list(split_timeseries_cv(data, n_splits=2, test_size=2, date_column=InputDataframeConstants.DATE_COL))

        # Should get 2 splits
        assert len(splits) == 2

        # Check that all rows with the same date go to the same split
        for train_mask, test_mask in splits:
            train_dates = data[train_mask][InputDataframeConstants.DATE_COL].unique()
            test_dates = data[test_mask][InputDataframeConstants.DATE_COL].unique()
            assert len(set(train_dates) & set(test_dates)) == 0

    def test_split_timeseries_cv_time_order(self):
        """Test that split_timeseries_cv maintains temporal order."""
        dates = pd.date_range("2023-01-01", periods=20)
        data = pd.DataFrame({InputDataframeConstants.DATE_COL: dates, "value": range(20)})

        splits = list(split_timeseries_cv(data, n_splits=3, test_size=4, date_column=InputDataframeConstants.DATE_COL))

        # Check that train dates come before test dates in each split
        for train_mask, test_mask in splits:
            train_dates = data[train_mask][InputDataframeConstants.DATE_COL]
            test_dates = data[test_mask][InputDataframeConstants.DATE_COL]

            if len(train_dates) > 0 and len(test_dates) > 0:
                assert train_dates.max() < test_dates.min()

    def test_split_timeseries_cv_zero_splits(self):
        """Test split_timeseries_cv with zero splits."""
        dates = pd.date_range("2023-01-01", periods=10)
        data = pd.DataFrame({InputDataframeConstants.DATE_COL: dates, "value": range(10)})

        splits = list(split_timeseries_cv(data, n_splits=0, test_size=4, date_column=InputDataframeConstants.DATE_COL))

        # Should get no splits
        assert len(splits) == 0

    def test_split_timeseries_cv_single_split(self):
        """Test split_timeseries_cv with single split."""
        dates = pd.date_range("2023-01-01", periods=10)
        data = pd.DataFrame({InputDataframeConstants.DATE_COL: dates, "value": range(10)})

        splits = list(split_timeseries_cv(data, n_splits=1, test_size=3, date_column=InputDataframeConstants.DATE_COL))

        # Should get 1 split
        assert len(splits) == 1

        train_mask, test_mask = splits[0]
        assert train_mask.sum() == 7  # 10 - 3
        assert test_mask.sum() == 3

    def test_run_with_error_handling_success(self):
        """Test successful error handling."""
        mock_model = Mock()
        mock_data = pd.DataFrame({"test": [1, 2, 3]})

        # Mock the run method to return a successful result
        self.test_instance.run = Mock(return_value=Mock())

        result = self.test_instance.run_with_error_handling(mock_model, mock_data)

        assert result is not None
        self.test_instance.run.assert_called_once_with(mock_model, mock_data)

    def test_run_with_error_handling_metric_calculation_error(self):
        """Test error handling for MetricCalculationError."""
        mock_model = Mock()
        mock_data = pd.DataFrame({"test": [1, 2, 3]})

        # Mock the run method to raise ZeroDivisionError
        self.test_instance.run = Mock(side_effect=ZeroDivisionError("Division by zero"))

        with pytest.raises(MetricCalculationError):
            self.test_instance.run_with_error_handling(mock_model, mock_data)

    def test_run_with_error_handling_test_execution_error(self):
        """Test error handling for TestExecutionError."""
        mock_model = Mock()
        mock_data = pd.DataFrame({"test": [1, 2, 3]})

        # Mock the run method to raise a generic exception
        self.test_instance.run = Mock(side_effect=ValueError("Some test error"))

        with pytest.raises(TestExecutionError):
            self.test_instance.run_with_error_handling(mock_model, mock_data)
