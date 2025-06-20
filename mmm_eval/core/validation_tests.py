# This file defines the validation tests for the MMM framework.

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Union
from mmm_eval.core.base_validation_test import BaseValidationTest
from mmm_eval.core.constants import PerturbationConstants, ValidationDataframeConstants
from mmm_eval.core.constants import ValidationTestConstants
from mmm_eval.core.validation_test_results import TestResult
from mmm_eval.core.validation_tests_models import ValidationTestNames
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.data.input_dataframe_constants import InputDataframeConstants
from mmm_eval.metrics.accuracy_functions import (
    calculate_absolute_percentage_change,
    calculate_mape,
    calculate_mean_for_cross_validation_folds,
    calculate_r_squared,
    calculate_std_for_cross_validation_folds,
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
            mean_mape=calculate_mean_for_cross_validation_folds(
                fold_metrics, AccuracyMetricNames.MAPE
            ),
            std_mape=calculate_std_for_cross_validation_folds(
                fold_metrics, AccuracyMetricNames.MAPE
            ),
            mean_r_squared=calculate_mean_for_cross_validation_folds(
                fold_metrics, AccuracyMetricNames.R_SQUARED
            ),
            std_r_squared=calculate_std_for_cross_validation_folds(
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

    def _filter_to_common_dates(
        self, baseline_data: pd.DataFrame, comparison_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter the data to the common dates for stability comparison."""

        common_start_date = max(
            baseline_data[InputDataframeConstants.DATE_COL].min(),
            comparison_data[InputDataframeConstants.DATE_COL].min(),
        )
        common_end_date = min(
            baseline_data[InputDataframeConstants.DATE_COL].max(),
            comparison_data[InputDataframeConstants.DATE_COL].max(),
        )

        baseline_data_fil = baseline_data[
            baseline_data[InputDataframeConstants.DATE_COL].between(
                common_start_date, common_end_date
            )
        ]
        comparison_data_fil = comparison_data[
            comparison_data[InputDataframeConstants.DATE_COL].between(
                common_start_date, common_end_date
            )
        ]

        return baseline_data_fil, comparison_data_fil

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

            logger.info(f"Running refresh stability test fold {i+1} of {len(cv_splits)}")

            # Get train/test data
            current_data = data.iloc[train_idx]
            # Combine current data with refresh data for retraining
            refresh_data = pd.concat([current_data, data.iloc[refresh_idx]], ignore_index=True)

            # Train model and get coefficients
            adapter.fit(current_data)
            current_model_rois = adapter.get_channel_roi()  # todo(): Update these names when Sam finishes the adapter
            adapter.fit(refresh_data)
            refreshed_model_rois = adapter.get_channel_roi()  # todo(): Update these names when Sam finishes the adapter

            # We test stability on how similar the retrained models coefficents are to the original model coefficents for the same time period
            # todo(): Sam is going to build a function into get roi to do it by time period
            # current_model, refresh_model = self._filter_to_common_dates(
            #     baseline_data=current_model,
            #     comparison_data=refreshed_model,
            # )

            # Get sum spend and return by channel
            # current_model_grpd = self._aggregate_by_channel_and_sum(current_model)
            # refresh_model_grpd = self._aggregate_by_channel_and_sum(refresh_model)

            # # Add calculated ROI column
            # current_model_grpd[ValidationDataframeConstants.CALCULATED_ROI_COL] = (
            #     self._add_calculated_roi_column(current_model_grpd)
            # )
            # refresh_model_grpd[ValidationDataframeConstants.CALCULATED_ROI_COL] = (
            #     self._add_calculated_roi_column(refresh_model_grpd)
            # )

            # merge the composition dfs by channel
            # merged = self._combine_dataframes_by_channel(
            #     baseline_df=current_model_rois,
            #     comparison_df=refreshed_model_rois,
            #     suffixes=("_current", "_refresh"),
            # )

            # calculate the pct change in volume
            percentage_change = calculate_absolute_percentage_change(
                baseline_series=current_model_rois,
                comparison_series=refreshed_model_rois,
            )
            # merged[
            #     ValidationDataframeConstants.PERCENTAGE_CHANGE_CHANNEL_CONTRIBUTION_COL
            # ] = calculate_absolute_percentage_change(
            #     baseline_series=merged[
            #         ValidationDataframeConstants.CALCULATED_ROI_COL + "_current"
            #     ],
            #     comparison_series=merged[
            #         ValidationDataframeConstants.CALCULATED_ROI_COL + "_refresh"
            #     ],
            # )

            fold_metrics.append(percentage_change)

        # Question: Does it make sense to calculate the mean of the mean percentage change?
        test_scores = RefreshStabilityMetricResults(
            mean_percentage_change=calculate_mean_for_cross_validation_folds(
                fold_metrics, RefreshStabilityMetricNames.MEAN_PERCENTAGE_CHANGE
            ),
            std_percentage_change=calculate_std_for_cross_validation_folds(
                fold_metrics, RefreshStabilityMetricNames.STD_PERCENTAGE_CHANGE
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
        return ValidationTestNames.PERTUBATION

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
        spend_col: InputDataframeConstants = InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL,
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
        df_copy[spend_col] = df[spend_col] * (1 + noise)
        return df_copy

    def _collate_channel_and_corresponding_roi_pct_change(
        self, df: pd.DataFrame
    ) -> dict[str, float]:
        """Collate the channel and corresponding ROI percentage change."""
        return dict(
            zip(
                df[InputDataframeConstants.MEDIA_CHANNEL_COL],
                df[
                    ValidationDataframeConstants.PERCENTAGE_CHANGE_CHANNEL_CONTRIBUTION_COL
                ],
            )
        )

    def run(self, adapter: BaseAdapter, data: pd.DataFrame) -> TestResult:
        """
        Run the perturbation test.
        """

        # Train model on original data
        adapter.fit(data)
        original_model = adapter.df
        original_contributions = self._aggregate_by_channel_and_sum(original_model)

        # Add noise to spend data and retrain
        noisy_data = self._add_gaussian_noise_to_spend(data)
        adapter.fit(noisy_data)
        noisy_model = adapter.df
        noisy_contributions = self._aggregate_by_channel_and_sum(noisy_model)

        # Add calculated ROI column
        original_contributions[ValidationDataframeConstants.CALCULATED_ROI_COL] = (
            self._add_calculated_roi_column(original_contributions)
        )
        noisy_contributions[ValidationDataframeConstants.CALCULATED_ROI_COL] = (
            self._add_calculated_roi_column(noisy_contributions)
        )

        # merge the composition dfs by channel
        merged = self._combine_dataframes_by_channel(
            baseline_df=original_contributions,
            comparison_df=noisy_contributions,
            suffixes=("_original", "_perturbed"),
        )

        # calculate the pct change in roi
        merged[
            ValidationDataframeConstants.PERCENTAGE_CHANGE_CHANNEL_CONTRIBUTION_COL
        ] = calculate_absolute_percentage_change(
            baseline_series=merged[
                ValidationDataframeConstants.CALCULATED_ROI_COL + "_original"
            ],
            comparison_series=merged[
                ValidationDataframeConstants.CALCULATED_ROI_COL + "_perturbed"
            ],
        )

        # Create metric results
        test_scores = PerturbationMetricResults(
            mean_aggregate_channel_roi_pct_change=self._get_mean_aggregate_channel_roi_pct_change(
                merged
            ),
            individual_channel_roi_pct_change=self._collate_channel_and_corresponding_roi_pct_change(
                merged
            ),
        )

        logger.info(f"Saving the test results for {self.test_name} test")

        return TestResult(
            test_name=ValidationTestNames.PERTUBATION,
            passed=test_scores.check_test_passed(),
            metric_names=PerturbationMetricNames.metrics_to_list(),
            test_scores=test_scores,
        )
