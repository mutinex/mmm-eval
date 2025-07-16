from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from mmm_eval.metrics.exceptions import InvalidMetricNameException
from mmm_eval.metrics.threshold_constants import (
    AccuracyThresholdConstants,
    CrossValidationThresholdConstants,
    PerturbationThresholdConstants,
    RefreshStabilityThresholdConstants,
)


def calculate_smape(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE).

    SMAPE is calculated as: 100 * (2 * |actual - predicted|) / (|actual| + |predicted|)

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        SMAPE value as float (percentage)

    Raises:
        ValueError: If series are empty or have different lengths

    """
    # Validate inputs
    if len(actual) == 0 or len(predicted) == 0:
        raise ValueError("Cannot calculate SMAPE on empty series")

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted series must have the same length")

    # Handle NaN values
    if actual.isna().any() or predicted.isna().any():
        raise ValueError("Actual and predicted series must be free of NaN values")

    # Handle division by zero and edge cases
    denominator = np.abs(actual) + np.abs(predicted)
    # Avoid division by zero by setting denominator to 1 where it's 0
    denominator = np.where(denominator == 0, 1, denominator)

    smape = 100 * np.mean(2 * np.abs(predicted - actual) / denominator)
    return float(smape)


class MetricNamesBase(Enum):
    """Base class for metric name enums."""

    @classmethod
    def to_list(cls) -> list[str]:
        """Convert the enum to a list of strings."""
        return [member.value for member in cls]


class AccuracyMetricNames(MetricNamesBase):
    """Define the names of the accuracy metrics."""

    MAPE = "mape"
    SMAPE = "smape"
    R_SQUARED = "r_squared"


class CrossValidationMetricNames(MetricNamesBase):
    """Define the names of the cross-validation metrics."""

    MEAN_MAPE = "mean_mape"
    STD_MAPE = "std_mape"
    MEAN_SMAPE = "mean_smape"
    STD_SMAPE = "std_smape"
    MEAN_R_SQUARED = "mean_r_squared"


class RefreshStabilityMetricNames(MetricNamesBase):
    """Define the names of the stability metrics."""

    MEAN_PERCENTAGE_CHANGE = "mean_percentage_change"
    STD_PERCENTAGE_CHANGE = "std_percentage_change"


# todo(): standardise to specify we are using decimal percents everywhere
class PerturbationMetricNames(MetricNamesBase):
    """Define the names of the perturbation metrics."""

    PERCENTAGE_CHANGE = "percentage_change"


class TestResultDFAttributes(MetricNamesBase):
    """Define the attributes of the test result DataFrame."""

    GENERAL_METRIC_NAME = "general_metric_name"
    SPECIFIC_METRIC_NAME = "specific_metric_name"
    METRIC_VALUE = "metric_value"
    METRIC_PASS = "metric_pass"


class MetricResults(BaseModel):
    """Define the results of the metrics."""

    def to_df(self) -> pd.DataFrame:
        """Convert the class of test results to a flat DataFrame format."""
        raise NotImplementedError("This method should be implemented by the subclass.")

    def _check_metric_threshold(self, metric_name: str, metric_value: float) -> bool:
        """Check if a specific metric passes its threshold.

        Args:
            metric_name: String name of the metric to check
            metric_value: Value of the metric

        Returns:
            True if metric passes threshold, False otherwise

        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def to_dict(self) -> dict[str, Any]:
        """Convert the class of test results to dictionary format."""
        return self.model_dump()

    def _create_single_metric_dataframe_row(
        self, general_metric_name: str, specific_metric_name: str, metric_value: float
    ) -> dict[str, Any]:
        """Create a standardized row dictionary for a single metric value in DataFrame format."""
        return {
            TestResultDFAttributes.GENERAL_METRIC_NAME.value: general_metric_name,
            TestResultDFAttributes.SPECIFIC_METRIC_NAME.value: specific_metric_name,
            TestResultDFAttributes.METRIC_VALUE.value: metric_value,
        }

    def _create_channel_based_metric_dataframe_rows(
        self, channel_series: pd.Series, metric_name: MetricNamesBase
    ) -> list[dict[str, float]]:
        """Create multiple DataFrame rows for channel-based metrics (e.g., per-channel percentages)."""
        return [
            self._create_single_metric_dataframe_row(
                general_metric_name=metric_name.value,
                specific_metric_name=f"{metric_name.value}_{channel}",
                metric_value=value,
            )
            for channel, value in channel_series.items()
        ]

    def add_pass_fail_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a pass/fail column to the DataFrame based on metric thresholds.

        Args:
            df: DataFrame with general_metric_name and metric_value columns

        Returns:
            DataFrame with additional metric_pass column

        """
        df_copy = df.copy()
        df_copy[TestResultDFAttributes.METRIC_PASS.value] = df_copy.apply(
            lambda row: self._check_metric_threshold(
                row[TestResultDFAttributes.GENERAL_METRIC_NAME.value], row[TestResultDFAttributes.METRIC_VALUE.value]
            ),
            axis=1,
        )
        return df_copy


class AccuracyMetricResults(MetricResults):
    """Define the results of the accuracy metrics."""

    mape: float
    smape: float
    r_squared: float

    def _check_metric_threshold(self, metric_name: str, metric_value: float) -> bool:
        """Check if a specific accuracy metric passes its threshold."""
        if metric_name == AccuracyMetricNames.MAPE.value:
            return bool(metric_value <= AccuracyThresholdConstants.MAPE)
        elif metric_name == AccuracyMetricNames.SMAPE.value:
            return bool(metric_value <= AccuracyThresholdConstants.SMAPE)
        elif metric_name == AccuracyMetricNames.R_SQUARED.value:
            return bool(metric_value >= AccuracyThresholdConstants.R_SQUARED)
        else:
            valid_metric_names = AccuracyMetricNames.to_list()
            raise InvalidMetricNameException(
                f"Invalid metric name: {metric_name}. Valid metric names are: {valid_metric_names}"
            )

    def to_df(self) -> pd.DataFrame:
        """Convert the accuracy metric results to a long DataFrame format."""
        df = pd.DataFrame(
            [
                self._create_single_metric_dataframe_row(
                    general_metric_name=AccuracyMetricNames.MAPE.value,
                    specific_metric_name=AccuracyMetricNames.MAPE.value,
                    metric_value=self.mape,
                ),
                self._create_single_metric_dataframe_row(
                    general_metric_name=AccuracyMetricNames.SMAPE.value,
                    specific_metric_name=AccuracyMetricNames.SMAPE.value,
                    metric_value=self.smape,
                ),
                self._create_single_metric_dataframe_row(
                    general_metric_name=AccuracyMetricNames.R_SQUARED.value,
                    specific_metric_name=AccuracyMetricNames.R_SQUARED.value,
                    metric_value=self.r_squared,
                ),
            ]
        )
        return self.add_pass_fail_column(df)

    @classmethod
    def populate_object_with_metrics(cls, actual: pd.Series, predicted: pd.Series) -> "AccuracyMetricResults":
        """Populate the object with the calculated metrics.

        Args:
            actual: The actual values
            predicted: The predicted values

        Returns:
            AccuracyMetricResults object with the metrics

        """
        return cls(
            mape=mean_absolute_percentage_error(actual, predicted) * 100,
            smape=calculate_smape(actual, predicted),
            r_squared=r2_score(actual, predicted),
        )


class CrossValidationMetricResults(MetricResults):
    """Define the results of the cross-validation metrics."""

    mean_mape: float
    std_mape: float
    mean_smape: float
    std_smape: float
    mean_r_squared: float

    def _check_metric_threshold(self, metric_name: str, metric_value: float) -> bool:
        """Check if a specific cross-validation metric passes its threshold."""
        if metric_name == CrossValidationMetricNames.MEAN_MAPE.value:
            return bool(metric_value <= CrossValidationThresholdConstants.MEAN_MAPE)
        elif metric_name == CrossValidationMetricNames.STD_MAPE.value:
            return bool(metric_value <= CrossValidationThresholdConstants.STD_MAPE)
        elif metric_name == CrossValidationMetricNames.MEAN_SMAPE.value:
            return bool(metric_value <= CrossValidationThresholdConstants.MEAN_SMAPE)
        elif metric_name == CrossValidationMetricNames.STD_SMAPE.value:
            return bool(metric_value <= CrossValidationThresholdConstants.STD_SMAPE)
        elif metric_name == CrossValidationMetricNames.MEAN_R_SQUARED.value:
            return bool(metric_value >= CrossValidationThresholdConstants.MEAN_R_SQUARED)
        else:
            valid_metric_names = CrossValidationMetricNames.to_list()
            raise InvalidMetricNameException(
                f"Invalid metric name: {metric_name}. Valid metric names are: {valid_metric_names}"
            )

    def to_df(self) -> pd.DataFrame:
        """Convert the cross-validation metric results to a long DataFrame format."""
        df = pd.DataFrame(
            [
                self._create_single_metric_dataframe_row(
                    general_metric_name=CrossValidationMetricNames.MEAN_MAPE.value,
                    specific_metric_name=CrossValidationMetricNames.MEAN_MAPE.value,
                    metric_value=self.mean_mape,
                ),
                self._create_single_metric_dataframe_row(
                    general_metric_name=CrossValidationMetricNames.STD_MAPE.value,
                    specific_metric_name=CrossValidationMetricNames.STD_MAPE.value,
                    metric_value=self.std_mape,
                ),
                self._create_single_metric_dataframe_row(
                    general_metric_name=CrossValidationMetricNames.MEAN_SMAPE.value,
                    specific_metric_name=CrossValidationMetricNames.MEAN_SMAPE.value,
                    metric_value=self.mean_smape,
                ),
                self._create_single_metric_dataframe_row(
                    general_metric_name=CrossValidationMetricNames.STD_SMAPE.value,
                    specific_metric_name=CrossValidationMetricNames.STD_SMAPE.value,
                    metric_value=self.std_smape,
                ),
                self._create_single_metric_dataframe_row(
                    general_metric_name=CrossValidationMetricNames.MEAN_R_SQUARED.value,
                    specific_metric_name=CrossValidationMetricNames.MEAN_R_SQUARED.value,
                    metric_value=self.mean_r_squared,
                ),
            ]
        )
        return self.add_pass_fail_column(df)


class RefreshStabilityMetricResults(MetricResults):
    """Define the results of the refresh stability metrics."""

    mean_percentage_change_for_each_channel: pd.Series
    std_percentage_change_for_each_channel: pd.Series

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _check_metric_threshold(self, metric_name: str, metric_value: float) -> bool:
        """Check if a specific refresh stability metric passes its threshold."""
        if metric_name == RefreshStabilityMetricNames.MEAN_PERCENTAGE_CHANGE.value:
            return bool(metric_value <= RefreshStabilityThresholdConstants.MEAN_PERCENTAGE_CHANGE)
        elif metric_name == RefreshStabilityMetricNames.STD_PERCENTAGE_CHANGE.value:
            return bool(metric_value <= RefreshStabilityThresholdConstants.STD_PERCENTAGE_CHANGE)
        else:
            valid_metric_names = RefreshStabilityMetricNames.to_list()
            raise InvalidMetricNameException(
                f"Invalid metric name: {metric_name}. Valid metric names are: {valid_metric_names}"
            )

    def to_df(self) -> pd.DataFrame:
        """Convert the refresh stability metric results to a long DataFrame format."""
        rows = []

        # Add mean and std percentage change for each channel
        rows.extend(
            self._create_channel_based_metric_dataframe_rows(
                channel_series=self.mean_percentage_change_for_each_channel,
                metric_name=RefreshStabilityMetricNames.MEAN_PERCENTAGE_CHANGE,
            )
        )
        rows.extend(
            self._create_channel_based_metric_dataframe_rows(
                channel_series=self.std_percentage_change_for_each_channel,
                metric_name=RefreshStabilityMetricNames.STD_PERCENTAGE_CHANGE,
            )
        )

        df = pd.DataFrame(rows)
        return self.add_pass_fail_column(df)


class PerturbationMetricResults(MetricResults):
    """Define the results of the perturbation metrics."""

    percentage_change_for_each_channel: pd.Series

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _check_metric_threshold(self, metric_name: str, metric_value: float) -> bool:
        """Check if a specific perturbation metric passes its threshold."""
        if metric_name == PerturbationMetricNames.PERCENTAGE_CHANGE.value:
            return bool(metric_value <= PerturbationThresholdConstants.PERCENTAGE_CHANGE)
        else:
            valid_metric_names = PerturbationMetricNames.to_list()
            raise InvalidMetricNameException(
                f"Invalid metric name: {metric_name}. Valid metric names are: {valid_metric_names}"
            )

    def to_df(self) -> pd.DataFrame:
        """Convert the perturbation metric results to a long DataFrame format."""
        df = pd.DataFrame(
            self._create_channel_based_metric_dataframe_rows(
                channel_series=self.percentage_change_for_each_channel,
                metric_name=PerturbationMetricNames.PERCENTAGE_CHANGE,
            )
        )
        return self.add_pass_fail_column(df)
