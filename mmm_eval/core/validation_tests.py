# This file defines the validation tests for the MMM framework.

import numpy as np
import pandas as pd
from typing import Any, Dict, Union
from mmm_eval.core.base_validation_test import BaseValidationTest
from mmm_eval.core.validation_test_constants import ValidationTestConstants
from mmm_eval.core.validation_test_results import TestResult
from mmm_eval.core.validation_tests_models import ValidationTestNames
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from mmm_eval.metrics.accuracy_functions import (
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
from mmm_eval.metrics.threshold_constants import (
    AccuracyThresholdConstants,
    CrossValidationThresholdConstants,
)


class AccuracyTest(BaseValidationTest):

    def run(self, model: Any, data: pd.DataFrame) -> TestResult:
        train, test = train_test_split(
            data,
            test_size=ValidationTestConstants.TRAIN_TEST_SPLIT_RATIO,
            random_state=ValidationTestConstants.RANDOM_STATE,
        )
        trained_model = model.fit(train)
        predictions = trained_model.predict(test)

        # Calculate metrics and convert to expected format
        test_scores = AccuracyMetricResults(
            mape=calculate_mape(
                actual=test["revenue"], predicted=predictions
            ),  # todo(): Use some constant revenue column, perhaps from loaders.py or a constants file
            r_squared=calculate_r_squared(
                actual=test["revenue"], predicted=predictions
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

    def run(self, model: Any, data: pd.DataFrame) -> TestResult:
        """
        Run the stability test.
        """

        # Initialize cross-validation splitter
        cv = TimeSeriesSplit(
            n_splits=ValidationTestConstants.N_SPLITS,
            test_size=ValidationTestConstants.TIME_SERIES_CROSS_VALIDATION_TEST_SIZE,
        )

        # Store metrics for each fold
        fold_metrics = []

        # Run cross-validation
        for train_idx, refresh_idx in cv.split(data):
            # Get train/test data
            train_data = data.iloc[train_idx]
            refresh_data = data.iloc[refresh_idx] + train_data

            # Train model and get coefficients
            current_model = model.fit(train_data)
            refreshed_model = model.fit(refresh_data)

            train_data, refresh_data = self.filter_to_common_dates(current_model, refreshed_model)

            train_data_grpd = (
                train_data.groupby(MEDIA_COL, dropna=False)
                .sum(numeric_only=True)
                .reset_index()
            )
            refresh_data_grpd = (
                refresh_data.groupby(MEDIA_COL, dropna=False)
                .sum(numeric_only=True)
                .reset_index()
            )

            # merge the composition dfs
            merged = train_data_grpd.merge(
                refresh_data_grpd,
                on=[DATE_COL, MEDIA_COL],
                suffixes=("_train", "_refresh"),
                how="inner",
            )

            # calculate the pct change in volume
            merged["mean_percentage_change"] = (
                (merged[MEDIA_COL + "_refresh"] - merged[MEDIA_COL + "_train"])
                / merged[MEDIA_COL + "_train"]
            ).abs()

            fold_metrics.append(
                RefreshStabilityMetricResults(
                    mean_percentage_change=merged["mean_percentage_change"].mean(),
                    std_percentage_change=merged["mean_percentage_change"].std(),
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
        cv = TimeSeriesSplit(n_splits=ValidationTestConstants.N_SPLITS)

        # Store metrics for each fold
        fold_metrics = []

        # Run cross-validation
        for train_idx, test_idx in cv.split(data):
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
                        actual=test["revenue"], predicted=predictions
                    ),  # todo(): Use some constant revenue column, perhaps from loaders.py or a constants file
                    r_squared=calculate_r_squared(
                        actual=test["revenue"], predicted=predictions
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
