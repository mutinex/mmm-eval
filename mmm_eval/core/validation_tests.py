# This file defines the validation tests for the MMM framework.

import numpy as np
import pandas as pd
from typing import Any, Dict, Union
from mmm_eval.core.base_validation_test import BaseValidationTest
from mmm_eval.core.constants import ValidationDataframeConstants
from mmm_eval.core.constants import ValidationTestConstants
from mmm_eval.core.validation_test_results import TestResult
from mmm_eval.core.validation_tests_models import ValidationTestNames
from sklearn.model_selection import TimeSeriesSplit, train_test_split

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


class AccuracyTest(BaseValidationTest):

    def run(self, model: Any, data: pd.DataFrame) -> TestResult:
        train, test = self._split_data_holdout(data)
        trained_model = model.fit(train)
        predictions = trained_model.predict(test)

        # Calculate metrics and convert to expected format
        test_scores = AccuracyMetricResults(
            mape=calculate_mape(
                actual=test[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL], predicted=predictions
            ),  # todo(): Use some constant revenue column, perhaps from loaders.py or a constants file
            r_squared=calculate_r_squared(
                actual=test[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL], predicted=predictions
            ),
        )

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

    def run(self, model: Any, data: pd.DataFrame) -> TestResult:
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
        for train_idx, test_idx in cv_splits:
            # Get train/test data
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]

            # Get predictions
            trained_model = model.fit(train)
            predictions = trained_model.predict(test)

            # Add in fold results
            fold_metrics.append(
                AccuracyMetricResults(
                    mape=calculate_mape(
                        actual=test[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL],
                        predicted=predictions,
                    ),  # todo(): Use some constant revenue column, perhaps from loaders.py or a constants file
                    r_squared=calculate_r_squared(
                        actual=test[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL],
                        predicted=predictions,
                    ),  # todo(): Use some constant revenue column, perhaps from loaders.py or a constants file
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

        return TestResult(
            test_name=ValidationTestNames.CROSS_VALIDATION,
            passed=test_scores.check_test_passed(),
            metric_names=CrossValidationMetricNames.metrics_to_list(),
            test_scores=test_scores,
        )

class StabilityTest(BaseValidationTest):
    """
    Validation test for the stability of the MMM framework.
    """

    def _filter_to_common_dates(
        baseline_data: pd.DataFrame, comparison_data: pd.DataFrame
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


    def run(self, model: Any, data: pd.DataFrame) -> TestResult:
        """
        Run the stability test.
        """

        # Initialize cross-validation splitter
        cv_splits = self._split_data_time_series_cv(data)

        # Store metrics for each fold
        fold_metrics = []

        # Run cross-validation
        for train_idx, refresh_idx in cv_splits:
            # Get train/test data
            current_data = data.iloc[train_idx]
            refresh_data = data.iloc[refresh_idx] + current_data

            # Train model and get coefficients
            current_model = model.fit(
                current_data
            ).df  # todo(): Update these names when Sam finishes the adapter
            refreshed_model = model.fit(
                refresh_data
            ).df  # todo(): Update these names when Sam finishes the adapter

            # We test stability on how similar the retrained models coefficents are to the original model coefficents for the same time period
            current_model, refresh_model = self._filter_to_common_dates(
                baseline_data=current_model,
                comparison_data=refreshed_model,
            )

            # Get sum spend and return by channel
            current_model_grpd = self._aggregate_by_channel_and_sum(current_model)
            refresh_model_grpd = self._aggregate_by_channel_and_sum(refresh_model)

            # Add calculated ROI column
            current_model_grpd[ValidationDataframeConstants.CALCULATED_ROI_COL] = self._add_calculated_roi_column(current_model_grpd)
            refresh_model_grpd[ValidationDataframeConstants.CALCULATED_ROI_COL] = self._add_calculated_roi_column(refresh_model_grpd)

            # merge the composition dfs by channel
            merged = self._combine_dataframes_by_channel(
                baseline_df=current_model_grpd,
                comparison_df=refresh_model_grpd,
                suffixes=("_current", "_refresh"),
            )

            # calculate the pct change in volume
            merged[
                ValidationDataframeConstants.PERCENTAGE_CHANGE_CHANNEL_CONTRIBUTION_COL
            ] = calculate_absolute_percentage_change(
                baseline_series=merged[
                    ValidationDataframeConstants.CALCULATED_ROI_COL + "_current"
                ],
                comparison_series=merged[
                    ValidationDataframeConstants.CALCULATED_ROI_COL + "_refresh"
                ],
            )

            fold_metrics.append(
                RefreshStabilityMetricResults(
                    mean_percentage_change=merged[
                        ValidationDataframeConstants.PERCENTAGE_CHANGE_CHANNEL_CONTRIBUTION_COL
                    ].mean(),
                    std_percentage_change=merged[
                        ValidationDataframeConstants.PERCENTAGE_CHANGE_CHANNEL_CONTRIBUTION_COL
                    ].std(),
                )
            )

        # Question: Does it make sense to calculate the mean of the mean percentage change?
        test_scores = RefreshStabilityMetricResults(
            mean_percentage_change=calculate_mean_for_cross_validation_folds(
                fold_metrics, RefreshStabilityMetricNames.MEAN_PERCENTAGE_CHANGE
            ),
            std_percentage_change=calculate_std_for_cross_validation_folds(
                fold_metrics, RefreshStabilityMetricNames.STD_PERCENTAGE_CHANGE
            ),
        )

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

        def _get_5_percent_gaussian_noise(self, df: pd.DataFrame) -> pd.DataFrame:
            """Get the gaussian noise for the perturbation test."""
            return np.random.normal(0, 0.05, size=len(df))
        
        def _add_gaussian_noise_to_spend(self, df: pd.DataFrame, spend_col: InputDataframeConstants = InputDataframeConstants.MEDIA_CHANNEL_SPEND_COL) -> pd.DataFrame:
            """Add the gaussian noise to the spend."""
            df_copy = df.copy()
            noise = self._get_5_percent_gaussian_noise(df_copy[spend_col])
            df_copy[spend_col] = df_copy[spend_col] * (1 + noise) #todo(): Does this mean its only positive noise?
            return df
        
        def _collate_channel_and_corresponding_roi_pct_change(self, df: pd.DataFrame) -> dict[str, float]:
            """Collate the channel and corresponding ROI pct change."""
            return df.groupby(InputDataframeConstants.MEDIA_CHANNEL_COL)[ValidationDataframeConstants.CALCULATED_ROI_COL].mean().to_dict()

        def run(self, model: Any, data: pd.DataFrame) -> TestResult:
            """
            Run the perturbation test.
            """

            df_original = data.copy()
            df_with_noise = self._add_gaussian_noise_to_spend(df_original)

            # Train model and get coefficients
            trained_model_original_spends = model.fit(df_original).df # todo(): Update these names when Sam finishes the adapter
            trained_model_perturbed_spends = model.fit(df_with_noise).df # todo(): Update these names when Sam finishes the adapter

            # Get sum spend and return by channel
            original_spends_grpd = self._aggregate_by_channel_and_sum(trained_model_original_spends)
            perturbed_spends_grpd = self._aggregate_by_channel_and_sum(trained_model_perturbed_spends)

            # Add calculated ROI column
            original_spends_grpd[ValidationDataframeConstants.CALCULATED_ROI_COL] = self._add_calculated_roi_column(original_spends_grpd)
            perturbed_spends_grpd[ValidationDataframeConstants.CALCULATED_ROI_COL] = self._add_calculated_roi_column(perturbed_spends_grpd)

            original_and_perturbed_spends = self._combine_dataframes_by_channel(
                baseline_df=original_spends_grpd,
                comparison_df=perturbed_spends_grpd,
                suffixes=("_original", "_perturbed"),
            )

            # calculate the pct change in volume
            original_and_perturbed_spends[
                ValidationDataframeConstants.PERCENTAGE_CHANGE_CHANNEL_CONTRIBUTION_COL
            ] = calculate_absolute_percentage_change(
                baseline_series=original_and_perturbed_spends[
                    InputDataframeConstants.MEDIA_CHANNEL_VOLUME_CONTRIBUTION_COL + "_original"
                ],
                comparison_series=original_and_perturbed_spends[
                    InputDataframeConstants.MEDIA_CHANNEL_VOLUME_CONTRIBUTION_COL + "_perturbed"
                ],
            )

            # Question: Does it make sense to calculate the mean of the mean percentage change?
            test_scores = PerturbationMetricResults(
                mean_aggregate_channel_roi_pct_change=self._get_mean_aggregate_channel_roi_pct_change(original_and_perturbed_spends),
                individual_channel_roi_pct_change=self._collate_channel_and_corresponding_roi_pct_change(original_and_perturbed_spends),
            )

            return TestResult(
                test_name=ValidationTestNames.PERTUBATION,
                passed=test_scores.check_test_passed(),
                metric_names=PerturbationMetricNames.metrics_to_list(),
                test_scores=test_scores,
            )
            
