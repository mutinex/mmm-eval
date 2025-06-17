# This file defines the validation tests for the MMM framework.

import numpy as np
import pandas as pd
from typing import Any, Dict, Union
from mmm_eval.core.base_validation_test import BaseValidationTest
from mmm_eval.core.dataframe_constants import ValidationDataframeConstants
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
    RefreshStabilityMetricNames,
    RefreshStabilityMetricResults,
)
from mmm_eval.metrics.refresh_stability_functions import (
    aggregate_via_media_channel,
    calculate_absolute_percentage_change_between_series,
    filter_to_common_dates,
)
from mmm_eval.metrics.threshold_constants import (
    AccuracyThresholdConstants,
    CrossValidationThresholdConstants,
)


class AccuracyTest(BaseValidationTest):

    def run(self, model: Any, data: pd.DataFrame) -> TestResult:
        train, test = self._split_data_holdout(data)
        trained_model = model.fit(train)
        predictions = trained_model.predict(test)

        # Calculate metrics and convert to expected format
        test_scores = AccuracyMetricResults(
            mape=calculate_mape(
                actual=test[InputDataframeConstants.REVENUE_COL], predicted=predictions
            ),  # todo(): Use some constant revenue column, perhaps from loaders.py or a constants file
            r_squared=calculate_r_squared(
                actual=test[InputDataframeConstants.REVENUE_COL], predicted=predictions
            ),
        )

        return TestResult(
            test_name=ValidationTestNames.ACCURACY,
            passed=test_scores.check_test_passed(),
            metric_names=AccuracyMetricNames.metrics_to_list(),
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

    def _aggregate_via_media_channel(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate the data to the media spend per channel."""
        return (
            data.groupby(InputDataframeConstants.MEDIA_CHANNEL_COL, dropna=False)
            .sum(numeric_only=True)
            .reset_index()
        )

    def _combine_current_and_refresh_data(
        self, current_data: pd.DataFrame, refresh_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine the current and refresh data."""
        return current_data.merge(
            refresh_data,
            on=[InputDataframeConstants.MEDIA_CHANNEL_COL],
            suffixes=("_current", "_refresh"),
            how="inner",
        )

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

            current_model_grpd = self._aggregate_via_media_channel(current_model)
            refresh_model_grpd = self._aggregate_via_media_channel(refresh_model)

            # merge the composition dfs
            merged = self._combine_current_and_refresh_data(
                current_data=current_model_grpd,
                refresh_data=refresh_model_grpd,
            )

            # calculate the pct change in volume
            merged[
                ValidationDataframeConstants.PERCENTAGE_CHANGE_CHANNEL_CONTRIBUTION_COL
            ] = calculate_absolute_percentage_change(
                baseline_series=merged[
                    InputDataframeConstants.MEDIA_CHANNEL_CONTRIBUTION_COL + "_current"
                ],
                comparison_series=merged[
                    InputDataframeConstants.MEDIA_CHANNEL_CONTRIBUTION_COL + "_refresh"
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
                        actual=test[InputDataframeConstants.REVENUE_COL],
                        predicted=predictions,
                    ),  # todo(): Use some constant revenue column, perhaps from loaders.py or a constants file
                    r_squared=calculate_r_squared(
                        actual=test[InputDataframeConstants.REVENUE_COL],
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

    class PerturbationTest(BaseValidationTest):
        """
        Validation test for the perturbation of the MMM framework.
        """

        def run(self, model: Any, data: pd.DataFrame) -> TestResult:
            """
            Run the perturbation test.
            """

            pass
