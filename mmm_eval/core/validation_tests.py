# This file defines the validation tests for the MMM framework.

import logging
import numpy as np
import pandas as pd
from mmm_eval.core.base_validation_test import BaseValidationTest
from mmm_eval.core.constants import PerturbationConstants
from mmm_eval.core.validation_test_results import TestResult
from mmm_eval.core.validation_tests_models import ValidationTestNames
from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.data.input_dataframe_constants import InputDataframeConstants
from mmm_eval.metrics.accuracy_functions import (
    calculate_absolute_percentage_change,
    calculate_mape,
    calculate_r_squared,
    calculate_mean_for_singular_values_across_cross_validation_folds,
    calculate_std_for_singular_values_across_cross_validation_folds,
    calculate_stds_for_series_across_cross_validation_folds,
    calculate_means_for_series_across_cross_validation_folds,
)
from mmm_eval.metrics.metric_models import (
    AccuracyMetricNames,
    AccuracyMetricResults,
    CrossValidationMetricNames,
    CrossValidationMetricResults,
    PerturbationMetricNames,
    PerturbationMetricResults,
    RefreshStabilityMetricNames,
    RefreshStabilityMetricResults,
)

logger = logging.getLogger(__name__)


class AccuracyTest(BaseValidationTest):
    """
    Validation test for model accuracy using holdout validation.

    This test evaluates model performance by splitting data into train/test sets
    and calculating MAPE and R-squared metrics on the test set.
    """

    @property
    def test_name(self) -> str:
        return ValidationTestNames.ACCURACY

    def run(self, adapter: BaseAdapter, data: pd.DataFrame) -> TestResult:
        train, test = self._split_data_holdout(data)
        adapter.fit(train)  # fit() modifies model in-place, returns None
        predictions = adapter.predict(test)  # predict() on same model instance

        # Calculate metrics and convert to expected format
        test_scores = AccuracyMetricResults(
            mape=calculate_mape(
                actual=test[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL],
                predicted=predictions,
            ),
            r_squared=calculate_r_squared(
                actual=test[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL],
                predicted=predictions,
            ),
        )

        logger.info(f"Saving the test results for {self.test_name} test")

        return TestResult(
            test_name=ValidationTestNames.ACCURACY,
            passed=test_scores.check_test_passed(),
            metric_names=AccuracyMetricNames.metrics_to_list(),
            test_scores=test_scores,
        )


class CrossValidationTest(BaseValidationTest):
    """
    Validation test for the cross-validation of the MMM framework.
    """

    @property
    def test_name(self) -> str:
        return ValidationTestNames.CROSS_VALIDATION

    def run(self, adapter: BaseAdapter, data: pd.DataFrame) -> TestResult:
        """
        Run the cross-validation test using time-series splits.

        Args:
            model: Model to validate
            data: Input data

        Returns:
            TestResult containing cross-validation metrics
        """
        # Initialize cross-validation splitter
        cv_splits = self._split_data_time_series_cv(data)

        # Store metrics for each fold
        fold_metrics = []

        # Run cross-validation
        for i, (train_idx, test_idx) in enumerate(cv_splits):

            logger.info(f"Running cross-validation fold {i+1} of {len(cv_splits)}")

            # Get train/test data
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]

            # Get predictions
            adapter.fit(train)
            predictions = adapter.predict(test)

            # Add in fold results
            fold_metrics.append(
                AccuracyMetricResults(
                    mape=calculate_mape(
                        actual=test[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL],
                        predicted=predictions,
                    ),
                    r_squared=calculate_r_squared(
                        actual=test[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL],
                        predicted=predictions,
                    ),
                )
            )

        # Calculate mean and std of metrics across folds and create metric results
        test_scores = CrossValidationMetricResults(
            mean_mape=calculate_mean_for_singular_values_across_cross_validation_folds(
                fold_metrics, AccuracyMetricNames.MAPE
            ),
            std_mape=calculate_std_for_singular_values_across_cross_validation_folds(
                fold_metrics, AccuracyMetricNames.MAPE
            ),
            mean_r_squared=calculate_mean_for_singular_values_across_cross_validation_folds(
                fold_metrics, AccuracyMetricNames.R_SQUARED
            ),
            std_r_squared=calculate_std_for_singular_values_across_cross_validation_folds(
                fold_metrics, AccuracyMetricNames.R_SQUARED
            ),
        )

        logger.info(f"Saving the test results for {self.test_name} test")

        return TestResult(
            test_name=ValidationTestNames.CROSS_VALIDATION,
            passed=test_scores.check_test_passed(),
            metric_names=CrossValidationMetricNames.metrics_to_list(),
            test_scores=test_scores,
        )


class RefreshStabilityTest(BaseValidationTest):
    """
    Validation test for the stability of the MMM framework.
    """

    @property
    def test_name(self) -> str:
        return ValidationTestNames.REFRESH_STABILITY

    def _get_common_dates(
        self, baseline_data: pd.DataFrame, comparison_data: pd.DataFrame
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Filter the data to the common dates for stability comparison."""

        common_start_date = max(
            baseline_data[InputDataframeConstants.DATE_COL].min(),
            comparison_data[InputDataframeConstants.DATE_COL].min(),
        )
        common_end_date = min(
            baseline_data[InputDataframeConstants.DATE_COL].max(),
            comparison_data[InputDataframeConstants.DATE_COL].max(),
        )

        return common_start_date, common_end_date

    def run(self, adapter: BaseAdapter, data: pd.DataFrame) -> TestResult:
        """
        Run the stability test.
        """

        # Initialize cross-validation splitter
        cv_splits = self._split_data_time_series_cv(data)

        # Store metrics for each fold
        fold_metrics = []

        # Run cross-validation
        for i, (train_idx, refresh_idx) in enumerate(cv_splits):

            logger.info(
                f"Running refresh stability test fold {i+1} of {len(cv_splits)}"
            )

            # Get train/test data
            current_data = data.iloc[train_idx]
            # Combine current data with refresh data for retraining
            refresh_data = pd.concat(
                [current_data, data.iloc[refresh_idx]], ignore_index=True
            )
            # Get common dates for roi stability comparison
            common_start_date, common_end_date = self._get_common_dates(
                baseline_data=current_data,
                comparison_data=refresh_data,
            )

            # Train model and get coefficients
            adapter.fit(current_data)
            current_model_rois = adapter.get_channel_roi(
                start_date=common_start_date,
                end_date=common_end_date,
            )
            adapter.fit(refresh_data)
            refreshed_model_rois = adapter.get_channel_roi(
                start_date=common_start_date,
                end_date=common_end_date,
            )

            # calculate the pct change in volume
            percentage_change = calculate_absolute_percentage_change(
                baseline_series=current_model_rois,
                comparison_series=refreshed_model_rois,
            )

            fold_metrics.append(percentage_change)

        # Calculate mean and std of percentage change for each channel across cross validation folds
        test_scores = RefreshStabilityMetricResults(
            mean_percentage_change_for_each_channel=calculate_means_for_series_across_cross_validation_folds(
                fold_metrics
            ),
            std_percentage_change_for_each_channel=calculate_stds_for_series_across_cross_validation_folds(
                fold_metrics
            ),
        )

        logger.info(f"Saving the test results for {self.test_name} test")

        return TestResult(
            test_name=ValidationTestNames.REFRESH_STABILITY,
            passed=test_scores.check_test_passed(),
            metric_names=RefreshStabilityMetricNames.metrics_to_list(),
            test_scores=test_scores,
        )


class PerturbationTest(BaseValidationTest):
    """
    Validation test for the perturbation of the MMM framework.
    """

    @property
    def test_name(self) -> str:
        return ValidationTestNames.PERTURBATION

    def _get_percent_gaussian_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Gaussian noise for perturbation testing.

        Args:
            df: Input dataframe to determine noise size

        Returns:
            Array of Gaussian noise values
        """
        return np.random.normal(
            PerturbationConstants.GAUSSIAN_NOISE_LOC,
            PerturbationConstants.NOISE_PERCENTAGE,
            size=len(df),
        )

    def _add_gaussian_noise_to_spend(
        self,
        df: pd.DataFrame,
        spend_cols: list[str],
    ) -> pd.DataFrame:
        """
        Add Gaussian noise to spend data for perturbation testing.

        Args:
            df: Input dataframe
            spend_col: Column name for spend data

        Returns:
            Dataframe with noise added to spend column
        """
        df_copy = df.copy()
        noise = self._get_percent_gaussian_noise(df)
        for spend_col in spend_cols:
            df_copy[spend_col] = df[spend_col] * (1 + noise)
        return df_copy

    def run(self, adapter: BaseAdapter, data: pd.DataFrame) -> TestResult:
        """
        Run the perturbation test.
        """

        # Train model on original data
        adapter.fit(data)
        original_model = adapter.get_channel_roi()

        # Add noise to spend data and retrain
        noisy_data = self._add_gaussian_noise_to_spend(
            df=data,
            spend_cols=adapter.channel_spend_cols,
        )
        adapter.fit(noisy_data)
        noisy_model = adapter.get_channel_roi()

        # calculate the pct change in roi
        percentage_change = calculate_absolute_percentage_change(
            baseline_series=original_model,
            comparison_series=noisy_model,
        )

        # Create metric results - roi % change for each channel
        test_scores = PerturbationMetricResults(
            percentage_change_for_each_channel=percentage_change,
        )

        logger.info(f"Saving the test results for {self.test_name} test")

        return TestResult(
            test_name=ValidationTestNames.PERTURBATION,
            passed=test_scores.check_test_passed(),
            metric_names=PerturbationMetricNames.metrics_to_list(),
            test_scores=test_scores,
        )
