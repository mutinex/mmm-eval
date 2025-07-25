# This file defines the validation tests for the MMM framework.

import logging

import numpy as np
import pandas as pd

from mmm_eval.adapters.base import BaseAdapter, PrimaryMediaRegressor
from mmm_eval.core.base_validation_test import BaseValidationTest
from mmm_eval.core.constants import ValidationTestConstants
from mmm_eval.core.validation_test_results import ValidationTestResult
from mmm_eval.core.validation_tests_models import ValidationTestNames
from mmm_eval.data.constants import InputDataframeConstants
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
    CrossValidationMetricNames,
    CrossValidationMetricResults,
    PerturbationMetricNames,
    PerturbationMetricResults,
    PlaceboMetricNames,
    PlaceboMetricResults,
    RefreshStabilityMetricNames,
    RefreshStabilityMetricResults,
)

logger = logging.getLogger(__name__)


class HoldoutAccuracyTest(BaseValidationTest):
    """Validation test for model accuracy using holdout validation.

    This test evaluates model performance by splitting data into train/test sets
    and calculating MAPE and R-squared metrics on the test set.
    """

    @property
    def test_name(self) -> ValidationTestNames:
        """Return the name of the test."""
        return ValidationTestNames.HOLDOUT_ACCURACY

    def run(self, adapter: BaseAdapter, data: pd.DataFrame) -> ValidationTestResult:
        """Run the accuracy test."""
        # Split data into train/test sets
        train, test = self._split_data_holdout(data)
        predictions = adapter.fit_and_predict(train, test)
        actual = test.groupby(self.date_column)[InputDataframeConstants.RESPONSE_COL].sum()
        assert len(actual) == len(predictions), "Actual and predicted lengths must match"

        # Calculate metrics
        test_scores = AccuracyMetricResults.populate_object_with_metrics(
            actual=pd.Series(actual),  # Ensure it's a Series
            predicted=pd.Series(predictions, index=actual.index),
        )

        logger.info(f"Saving the test results for {self.test_name} test")

        return ValidationTestResult(
            test_name=ValidationTestNames.HOLDOUT_ACCURACY,
            metric_names=AccuracyMetricNames.to_list(),
            test_scores=test_scores,
        )


class InSampleAccuracyTest(BaseValidationTest):
    """Validation test for model accuracy using in-sample validation.

    This test evaluates model performance by fitting the model on the full dataset
    and calculating MAPE and R-squared metrics on the training data.
    """

    @property
    def test_name(self) -> ValidationTestNames:
        """Return the name of the test."""
        return ValidationTestNames.IN_SAMPLE_ACCURACY

    def run(self, adapter: BaseAdapter, data: pd.DataFrame) -> ValidationTestResult:
        """Run the in-sample accuracy test."""
        # Fit model on full dataset and get predictions
        predictions = adapter.fit_and_predict_in_sample(data)
        actual = data.groupby(self.date_column)[InputDataframeConstants.RESPONSE_COL].sum()
        assert len(actual) == len(predictions), "Actual and predicted lengths must match"

        # Calculate metrics
        test_scores = AccuracyMetricResults.populate_object_with_metrics(
            actual=pd.Series(actual),  # Ensure it's a Series
            predicted=pd.Series(predictions, index=actual.index),
        )

        logger.info(f"Saving the test results for {self.test_name} test")

        return ValidationTestResult(
            test_name=ValidationTestNames.IN_SAMPLE_ACCURACY,
            metric_names=AccuracyMetricNames.to_list(),
            test_scores=test_scores,
        )


class CrossValidationTest(BaseValidationTest):
    """Validation test for the cross-validation of the MMM framework."""

    @property
    def test_name(self) -> ValidationTestNames:
        """Return the name of the test."""
        return ValidationTestNames.CROSS_VALIDATION

    def run(self, adapter: BaseAdapter, data: pd.DataFrame) -> ValidationTestResult:
        """Run the cross-validation test using time-series splits.

        Args:
            model: Model to validate
            adapter: Adapter to use for the test
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
            train = data.loc[train_idx]
            test = data.loc[test_idx]

            # Get predictions
            predictions = adapter.fit_and_predict(train, test)
            actual = test.groupby(self.date_column)[InputDataframeConstants.RESPONSE_COL].sum()
            assert len(actual) == len(predictions), "Actual and predicted lengths must match"

            # Add in fold results
            fold_metrics.append(
                AccuracyMetricResults.populate_object_with_metrics(
                    actual=pd.Series(actual),  # Ensure it's a Series
                    predicted=pd.Series(predictions, index=actual.index),
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
            mean_smape=calculate_mean_for_singular_values_across_cross_validation_folds(
                fold_metrics, AccuracyMetricNames.SMAPE
            ),
            std_smape=calculate_std_for_singular_values_across_cross_validation_folds(
                fold_metrics, AccuracyMetricNames.SMAPE
            ),
            mean_r_squared=calculate_mean_for_singular_values_across_cross_validation_folds(
                fold_metrics, AccuracyMetricNames.R_SQUARED
            ),
        )

        logger.info(f"Saving the test results for {self.test_name} test")

        return ValidationTestResult(
            test_name=ValidationTestNames.CROSS_VALIDATION,
            metric_names=CrossValidationMetricNames.to_list(),
            test_scores=test_scores,
        )


class RefreshStabilityTest(BaseValidationTest):
    """Validation test for the stability of the MMM framework."""

    @property
    def test_name(self) -> ValidationTestNames:
        """Return the name of the test."""
        return ValidationTestNames.REFRESH_STABILITY

    def _get_common_dates(
        self,
        baseline_data: pd.DataFrame,
        comparison_data: pd.DataFrame,
        date_column: str,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Filter the data to the common dates for stability comparison."""
        common_start_date = max(
            baseline_data[date_column].min(),
            comparison_data[date_column].min(),
        )
        common_end_date = min(
            baseline_data[date_column].max(),
            comparison_data[date_column].max(),
        )

        # Convert to Timestamps if they're Series
        if isinstance(common_start_date, pd.Series):
            common_start_date = common_start_date.iloc[0]
        if isinstance(common_end_date, pd.Series):
            common_end_date = common_end_date.iloc[0]

        return common_start_date, common_end_date

    def run(self, adapter: BaseAdapter, data: pd.DataFrame) -> ValidationTestResult:
        """Run the stability test."""
        # Initialize cross-validation splitter
        cv_splits = self._split_data_time_series_cv(data)

        # Store metrics for each fold
        fold_metrics = []

        # Run cross-validation
        for i, (train_idx, refresh_idx) in enumerate(cv_splits):

            logger.info(f"Running refresh stability test fold {i+1} of {len(cv_splits)}")

            # Get train/test data
            # todo(): Can we somehow store these training changes in the adapter for use in time series holdout test
            current_data = data.loc[train_idx]
            # Combine current data with refresh data for retraining
            refresh_data = pd.concat([current_data, data.loc[refresh_idx]], ignore_index=True)
            # Get common dates for roi stability comparison
            common_start_date, common_end_date = self._get_common_dates(
                baseline_data=current_data,
                comparison_data=refresh_data,
                date_column=adapter.date_column,
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

        return ValidationTestResult(
            test_name=ValidationTestNames.REFRESH_STABILITY,
            metric_names=RefreshStabilityMetricNames.to_list(),
            test_scores=test_scores,
        )


class PerturbationTest(BaseValidationTest):
    """Validation test for the perturbation of the MMM framework."""

    @property
    def test_name(self) -> ValidationTestNames:
        """Return the name of the test."""
        return ValidationTestNames.PERTURBATION

    def _get_percent_gaussian_noise(self, df: pd.DataFrame) -> np.ndarray:
        """Generate Gaussian noise for perturbation testing.

        Args:
            df: Input dataframe to determine noise size

        Returns:
            Array of Gaussian noise values

        """
        return self.rng.normal(
            ValidationTestConstants.PerturbationConstants.GAUSSIAN_NOISE_LOC,
            ValidationTestConstants.PerturbationConstants.GAUSSIAN_NOISE_SCALE,
            size=len(df),
        )

    def _add_gaussian_noise_to_primary_regressors(
        self,
        df: pd.DataFrame,
        regressor_cols: list[str],
    ) -> pd.DataFrame:
        """Add Gaussian noise to primary media regressors for perturbation testing.

        Args:
            df: Input dataframe
            regressor_cols: Column names of primary media regressors

        Returns:
            Dataframe with noise added to primary regressor columns

        """
        df_copy = df.copy()
        noise = self._get_percent_gaussian_noise(df)
        for regressor_col in regressor_cols:
            df_copy[regressor_col] = df[regressor_col] * (1 + noise)
        return df_copy

    def run(self, adapter: BaseAdapter, data: pd.DataFrame) -> ValidationTestResult:
        """Run the perturbation test."""
        # Train model on original data
        adapter.fit(data)
        original_rois = adapter.get_channel_roi()

        # TODO: support perturbation of reach and frequency regressors
        if adapter.primary_media_regressor_type == PrimaryMediaRegressor.REACH_AND_FREQUENCY:
            logger.warning(
                "Perturbation test skipped: Reach and frequency regressor type not supported for perturbation."
            )
            # Return NaN results for each channel indicating the test was not run
            channel_names = adapter.get_channel_names()
            test_scores = PerturbationMetricResults(
                percentage_change_for_each_channel=pd.Series(np.nan, index=channel_names),
            )
            return ValidationTestResult(
                test_name=ValidationTestNames.PERTURBATION,
                metric_names=PerturbationMetricNames.to_list(),
                test_scores=test_scores,
            )

        # Add noise to primary regressor data and retrain
        noisy_data = self._add_gaussian_noise_to_primary_regressors(
            df=data,
            regressor_cols=adapter.primary_media_regressor_columns,
        )
        adapter.fit(noisy_data)
        noise_rois = adapter.get_channel_roi()

        # calculate the pct change in roi
        percentage_change = calculate_absolute_percentage_change(
            baseline_series=original_rois,
            comparison_series=noise_rois,
        )

        # Create metric results - roi % change for each channel
        test_scores = PerturbationMetricResults(
            percentage_change_for_each_channel=percentage_change,
        )

        logger.info(f"Saving the test results for {self.test_name} test")

        return ValidationTestResult(
            test_name=ValidationTestNames.PERTURBATION,
            metric_names=PerturbationMetricNames.to_list(),
            test_scores=test_scores,
        )


class PlaceboTest(BaseValidationTest):
    """Validation test for detecting spurious correlations in the MMM framework.

    This test creates a shuffled version of an existing media channel and tests whether
    the model assigns a low ROI to this spurious feature.
    """

    @property
    def test_name(self) -> ValidationTestNames:
        """Return the name of the test."""
        return ValidationTestNames.PLACEBO

    def run(self, adapter: BaseAdapter, data: pd.DataFrame) -> ValidationTestResult:
        """Run the placebo test."""
        if adapter.primary_media_regressor_type == PrimaryMediaRegressor.REACH_AND_FREQUENCY:
            logger.warning(
                "Placebo test skipped: Reach and frequency regressor type not supported for placebo testing."
            )
            # Return NaN results indicating the test was not run
            test_scores = PlaceboMetricResults(
                shuffled_channel_roi=np.nan,
                shuffled_channel_name="test_skipped",
            )
            return ValidationTestResult(
                test_name=ValidationTestNames.PLACEBO,
                metric_names=PlaceboMetricNames.to_list(),
                test_scores=test_scores,
            )

        # Select a random channel to shuffle
        original_channel = str(self.rng.choice(adapter.media_channels))

        # Create shuffled indices for consistent shuffling across related columns
        shuffled_indices = np.arange(len(data))
        self.rng.shuffle(shuffled_indices)

        # Use the new template method to add placebo channel
        adapter_copy, shuffled_data = adapter.add_placebo_channel(
            original_channel_name=original_channel, data_to_shuffle=data, shuffled_indices=shuffled_indices
        )

        # Fit the copied adapter and check ROI
        adapter_copy.fit(shuffled_data)
        rois = adapter_copy.get_channel_roi()
        shuffled_channel_name = f"{original_channel}_shuffled"
        shuffled_roi = rois[shuffled_channel_name]

        # Create metric results
        test_scores = PlaceboMetricResults(
            shuffled_channel_roi=shuffled_roi,
            shuffled_channel_name=shuffled_channel_name,
        )

        logger.info(f"Saving the test results for {self.test_name} test")

        return ValidationTestResult(
            test_name=ValidationTestNames.PLACEBO,
            metric_names=PlaceboMetricNames.to_list(),
            test_scores=test_scores,
        )
