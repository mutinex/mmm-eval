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
    PlaceboThresholdConstants,
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

    numerator = 2 * np.abs(predicted - actual)
    denominator = np.abs(actual) + np.abs(predicted)
    # avoid division by zero edge case if both numerator and denominator are zero
    mask = denominator != 0
    smape_terms = np.where(mask, numerator / denominator, 0)
    smape = 100 * np.mean(smape_terms)
    return float(smape)


def crps_one_date(df: pd.DataFrame) -> float:
    """Calculate CRPS for a single date.

    CRPS is a probabilistic generalisation of the mean absolute error,
    i.e, if the distribution is a repeat of a single value, CRPS is equal to the mean absolute error
    We scale it by 100 / response to make it in the same units as MAPE for ease of comparison and thresholding.

    Args:
        df: DataFrame containing the response and predicted distribution

    Returns:
        Scaled CRPS value as float

    """
    response = df["response"].values[0]
    n_samples = df.shape[0]
    if n_samples == 1:
        # CRPS reduces to absolute error for one sample
        crps = np.abs(df["pred_distribution"].values[0] - response)
    else:
        pred_samples = df["pred_distribution"].values
        pred_sample_one = pred_samples[: n_samples // 2]
        pred_sample_two = pred_samples[n_samples // 2 :]
        if len(pred_sample_two) != len(pred_sample_one):
            # If an odd number of samples, drop the last one
            pred_sample_two = pred_sample_two[:-1]

        crps = np.mean(np.abs(pred_samples - response)) - 0.5 * np.mean(np.abs(pred_sample_one - pred_sample_two))

    response_bounded = max(response, 1e-5)
    return 100 * crps / response_bounded


def calculate_crps(response_series: pd.Series, predicted_distribution: pd.DataFrame, date_column: str) -> float:
    """Calculate Mean Continuous Ranked Probability Score (CRPS).

    Args:
        response_series: A series of the actual sales values
        predicted_distribution: A dataframe containing a sample of the prediction distribution for each date.
            If there is only a single sample, the CRPS reduces to the mean absolute (percentage) error.
        date_column: Name of the date index colume to merge inputs together.

    Returns:
        Scaled CRPS value as float

    """
    response_series.name = "response"
    predicted_distribution = predicted_distribution.merge(response_series.reset_index(), on=date_column)

    return predicted_distribution.groupby(date_column).apply(crps_one_date).mean()


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
    CRPS = "crps"


class CrossValidationMetricNames(MetricNamesBase):
    """Define the names of the cross-validation metrics."""

    MEAN_MAPE = "mean_mape"
    STD_MAPE = "std_mape"
    MEAN_SMAPE = "mean_smape"
    STD_SMAPE = "std_smape"
    MEAN_R_SQUARED = "mean_r_squared"
    MEAN_CRPS = "mean_crps"


class RefreshStabilityMetricNames(MetricNamesBase):
    """Define the names of the stability metrics."""

    MEAN_PERCENTAGE_CHANGE = "mean_percentage_change"
    STD_PERCENTAGE_CHANGE = "std_percentage_change"


# todo(): standardise to specify we are using decimal percents everywhere
class PerturbationMetricNames(MetricNamesBase):
    """Define the names of the perturbation metrics."""

    PERCENTAGE_CHANGE = "percentage_change"


class PlaceboMetricNames(MetricNamesBase):
    """Define the names of the placebo test metrics."""

    SHUFFLED_CHANNEL_ROI = "shuffled_channel_roi"


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
    crps: float

    def _check_metric_threshold(self, metric_name: str, metric_value: float) -> bool:
        """Check if a specific accuracy metric passes its threshold."""
        if metric_name == AccuracyMetricNames.MAPE.value:
            return bool(metric_value <= AccuracyThresholdConstants.MAPE)
        elif metric_name == AccuracyMetricNames.SMAPE.value:
            return bool(metric_value <= AccuracyThresholdConstants.SMAPE)
        elif metric_name == AccuracyMetricNames.R_SQUARED.value:
            return bool(metric_value >= AccuracyThresholdConstants.R_SQUARED)
        elif metric_name == AccuracyMetricNames.CRPS.value:
            return bool(metric_value <= AccuracyThresholdConstants.CRPS)
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
                self._create_single_metric_dataframe_row(
                    general_metric_name=AccuracyMetricNames.CRPS.value,
                    specific_metric_name=AccuracyMetricNames.CRPS.value,
                    metric_value=self.crps,
                ),
            ]
        )
        return self.add_pass_fail_column(df)

    @classmethod
    def populate_object_with_metrics(
        cls, actual: pd.Series, predicted: pd.Series, distribution: pd.DataFrame, date_column: str
    ) -> "AccuracyMetricResults":
        """Populate the object with the calculated metrics.

        Args:
            actual: The actual values
            predicted: The predicted values
            distribution: The predicted distribution
            date_column: The name of the date column

        Returns:
            AccuracyMetricResults object with the metrics

        """
        return cls(
            mape=mean_absolute_percentage_error(actual, predicted) * 100,
            smape=calculate_smape(actual, predicted),
            r_squared=r2_score(actual, predicted),
            crps=calculate_crps(actual, distribution, date_column),
        )


class CrossValidationMetricResults(MetricResults):
    """Define the results of the cross-validation metrics."""

    mean_mape: float
    std_mape: float
    mean_smape: float
    std_smape: float
    mean_r_squared: float
    mean_crps: float

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
        elif metric_name == CrossValidationMetricNames.MEAN_CRPS.value:
            return bool(metric_value <= CrossValidationThresholdConstants.MEAN_CRPS)
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
                self._create_single_metric_dataframe_row(
                    general_metric_name=CrossValidationMetricNames.MEAN_CRPS.value,
                    specific_metric_name=CrossValidationMetricNames.MEAN_CRPS.value,
                    metric_value=self.mean_crps,
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


class PlaceboMetricResults(MetricResults):
    """Define the results of the placebo test metrics."""

    shuffled_channel_roi: float
    shuffled_channel_name: str

    def _check_metric_threshold(self, metric_name: str, metric_value: float) -> bool:
        """Check if a specific placebo test metric passes its threshold."""
        if metric_name == PlaceboMetricNames.SHUFFLED_CHANNEL_ROI.value:
            return bool(metric_value <= PlaceboThresholdConstants.ROI_THRESHOLD)
        else:
            valid_metric_names = PlaceboMetricNames.to_list()
            raise InvalidMetricNameException(
                f"Invalid metric name: {metric_name}. Valid metric names are: {valid_metric_names}"
            )

    def to_df(self) -> pd.DataFrame:
        """Convert the placebo test metric results to a long DataFrame format."""
        df = pd.DataFrame(
            [
                self._create_single_metric_dataframe_row(
                    general_metric_name=PlaceboMetricNames.SHUFFLED_CHANNEL_ROI.value,
                    specific_metric_name=f"{PlaceboMetricNames.SHUFFLED_CHANNEL_ROI.value}_{self.shuffled_channel_name}",
                    metric_value=self.shuffled_channel_roi,
                ),
            ]
        )
        return self.add_pass_fail_column(df)
