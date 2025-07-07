"""Google Meridian adapter for MMM evaluation."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from meridian.analysis.analyzer import Analyzer
from meridian.data import data_frame_input_data_builder as data_builder
from meridian.model.model import Meridian
from meridian.model.spec import ModelSpec

from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.configs import MeridianConfig
from mmm_eval.data.constants import InputDataframeConstants

logger = logging.getLogger(__name__)

# Meridian requires revenue to be specified in this form, but we accept raw revenue as
# input for continuity between frameworks
REVENUE_PER_KPI_COL = "revenue_per_kpi"


def _validate_media_channels(df: pd.DataFrame, config: MeridianConfig) -> None:
    """Validate that media channels have sufficient variation for Meridian modeling.

    Meridian requires media channels to have non-zero spend across time periods and geos
    to estimate meaningful parameters. Channels with zero spend everywhere will cause
    the model to fail or produce unreliable results.

    Args:
        df: Input DataFrame containing media spend data
        config: MeridianConfig object containing media channel specifications

    Raises:
        ValueError: If any media channel has zero spend across all time periods and geos

    """
    input_config = config.input_data_builder_config

    # Check spend columns for each media channel
    for channel, spend_col in zip(input_config.media_channels, input_config.channel_spend_columns, strict=False):
        if spend_col not in df.columns:
            raise ValueError(f"Spend column '{spend_col}' for channel '{channel}' not found in DataFrame")

        # Check if the channel has any non-zero spend
        if df[spend_col].sum() == 0:
            raise ValueError(
                f"Media channel '{channel}' (column '{spend_col}') has zero spend across "
                f"all time periods and geos. This will likely cause Meridian to fail or produce "
                f"unreliable results. Consider removing this channel from the model or "
                f"ensuring it has sufficient spend variation."
            )


def construct_meridian_data_object(df: pd.DataFrame, config: MeridianConfig) -> pd.DataFrame:
    """Construct a Meridian data object from a pandas DataFrame.

    This function transforms a standard DataFrame into the format required by the Meridian
    framework. It handles the conversion of revenue to revenue_per_kpi, and configures
    the data builder with various components including KPI, population, controls,
    media channels (with different types: reach/frequency, impressions, or spend-only),
    organic media, and non-media treatments.

    Args:
        df: Input DataFrame containing the raw data with columns for dates, response,
            revenue, media spend, and other variables as specified in the config.
        config: MeridianConfig object containing the configuration for data transformation
            including column mappings and feature specifications.

    Returns:
        A Meridian data object built from the input DataFrame according to the config.

    Note:
        The function modifies the input DataFrame by:
        - Converting revenue to revenue_per_kpi (revenue / response)
        - Dropping the original revenue column
        - Adding various media and control components based on config
        - Validating that media channels have sufficient variation (non-zero spend)

    Raises:
        ValueError: If a media channel has zero spend across all time periods and geos,
                   as this would cause Meridian to fail or produce unreliable results.

    """
    df = df.copy()

    # Validate media channels have sufficient variation
    _validate_media_channels(df, config)

    # convert from "revenue" to "revenue_per_kpi"
    df[REVENUE_PER_KPI_COL] = (
        df[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL] / df[InputDataframeConstants.RESPONSE_COL]
    )
    df = df.drop(columns=InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL)

    input_data_builder_schema = config.input_data_builder_config

    # KPI, population, and control variables
    builder = (
        data_builder.DataFrameInputDataBuilder(kpi_type="non_revenue")
        .with_kpi(df, time_col=config.date_column, kpi_col=InputDataframeConstants.RESPONSE_COL)
        .with_revenue_per_kpi(df, time_col=config.date_column, revenue_per_kpi_col=REVENUE_PER_KPI_COL)
    )
    # population, if provided, needs to be called "population" in the DF
    if "population" in df.columns:
        builder = builder.with_population(df)

    # controls (non-intervenable, e.g. macroeconomics)
    if input_data_builder_schema.control_columns:
        builder = builder.with_controls(
            df, time_col=config.date_column, control_cols=input_data_builder_schema.control_columns
        )

    # add paid media
    # without impressions/reach/frequency: media_cols = media_spend_cols
    # with impressions: media_cols = impressions cols
    # with reach/frequency: use .with_reach() instead
    if input_data_builder_schema.channel_reach_columns:
        builder = builder.with_reach(
            df,
            reach_cols=input_data_builder_schema.channel_reach_columns,
            frequency_cols=input_data_builder_schema.channel_frequency_columns,
            rf_spend_cols=input_data_builder_schema.channel_spend_columns,
            rf_channels=input_data_builder_schema.media_channels,
            time_col=config.date_column,
        )
    else:
        media_cols = (
            input_data_builder_schema.channel_impressions_columns or input_data_builder_schema.channel_spend_columns
        )
        builder = builder.with_media(
            df,
            media_cols=media_cols,
            media_spend_cols=input_data_builder_schema.channel_spend_columns,
            media_channels=input_data_builder_schema.media_channels,
            time_col=config.date_column,
        )

    # organic media
    if input_data_builder_schema.organic_media_columns:
        builder = builder.with_organic_media(
            df,
            organic_media_cols=input_data_builder_schema.organic_media_columns,
            organic_media_channels=input_data_builder_schema.organic_media_channels,
            media_time_col=config.date_column,
        )

    # non-media treatments (anything that is "intervenable", e.g. pricing/promotions)
    if input_data_builder_schema.non_media_treatment_columns:
        builder = builder.with_non_media_treatments(
            df,
            non_media_treatment_cols=input_data_builder_schema.non_media_treatment_columns,
            time_col=config.date_column,
        )

    return builder.build()


def construct_holdout_mask(max_train_date: pd.Timestamp, time_index: np.ndarray) -> np.ndarray:
    """Construct a boolean mask for holdout period identification.

    This function creates a boolean mask that identifies which time periods fall into
    the holdout/test period (after the maximum training date). The mask can be used
    to separate training and test data or to filter predictions to only the holdout period.

    Args:
        max_train_date: The maximum date to be considered part of the training period.
            All dates after this will be marked as holdout/test data.
        time_index: Array-like object containing the time index to be masked.
            Can be any format that pandas.to_datetime can convert.

    Returns:
        A boolean array of the same length as time_index, where True indicates
        the time period is in the holdout/test set (after max_train_date).

    Example:
        >>> time_index = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        >>> max_train_date = pd.Timestamp('2023-06-30')
        >>> mask = construct_holdout_mask(max_train_date, time_index)
        >>> # mask will be True for dates from 2023-07-01 onwards

    """
    full_index = pd.to_datetime(time_index)
    test_index = full_index[full_index > max_train_date]

    return full_index.isin(test_index)


class MeridianAdapter(BaseAdapter):
    """Adapter for Google Meridian MMM framework."""

    def __init__(self, config: MeridianConfig):
        """Initialize the Meridian adapter.

        Args:
            config: MeridianConfig object

        """
        self.config = config
        self.input_data_builder_schema = config.input_data_builder_config

        self.date_column = config.date_column
        self.channel_spend_columns = self.input_data_builder_schema.channel_spend_columns
        self.media_channels = self.input_data_builder_schema.media_channels

        # Initialize stateful attributes to None/False
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all stateful attributes to their initial values.

        This method ensures that the adapter starts with a clean state each time
        fit() is called, which is critical for the validation suite to work correctly.
        Only attributes that are set during fit() should be reset here.
        """
        self.training_data = None
        self.max_train_date = None
        self.holdout_mask = None
        self.model = None
        self.trace = None
        self.analyzer = None
        self.is_fitted = False

    def fit(self, data: pd.DataFrame, max_train_date: pd.Timestamp | None = None) -> None:
        """Fit the Meridian model to data.

        Args:
            data: Training data
            max_train_date: Optional maximum training date for holdout validation

        """
        # Reset state to ensure clean start when new data is provided
        self._reset_state()

        # build Meridian data object
        self.training_data = construct_meridian_data_object(data, self.config)
        self.max_train_date = max_train_date

        model_spec_kwargs = dict(self.config.model_spec_config)

        # if max train date is provided, construct a mask that is True for all dates before max_train_date
        if self.max_train_date:
            self.holdout_mask = construct_holdout_mask(self.max_train_date, self.training_data.kpi.time)
            # model expects a 2D array of shape (n_geos, n_times) so have to duplicate the values across each geo
            model_spec_kwargs["holdout_id"] = np.repeat(
                self.holdout_mask[None, :], repeats=len(self.training_data.kpi.geo), axis=0
            )

        # Create and fit the Meridian model
        model_spec = ModelSpec(**model_spec_kwargs)
        self.model = Meridian(
            input_data=self.training_data,
            model_spec=model_spec,
        )
        self.trace = self.model.sample_posterior(**dict(self.config.sample_posterior_config))

        # used to compute channel contributions for ROI calculations
        self.analyzer = Analyzer(self.model)
        self.is_fitted = True

    def predict(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Make predictions using the fitted model.

        This returns predictions for the entirety of the dataset passed to fit() unless
        `max_train_date` is specified when calling fit(); in that case it only returns
        predictions for the time periods indicated by the holdout mask.

        Note: Meridian doesn't require input data for prediction - it uses the fitted
        model state, so the `data` argument will be ignored if passed.

        Args:
            data: not used, see above

        Returns:
            Predicted values

        """
        if not self.is_fitted or self.analyzer is None:
            raise RuntimeError("Model must be fit before prediction")

        # shape (n_chains, n_draws, n_times)
        preds_tensor = self.analyzer.expected_outcome(aggregate_geos=True, aggregate_times=False)
        posterior_mean = np.mean(preds_tensor, axis=(0, 1))

        # if holdout mask is provided, use it to mask the predictions to restrict only to the
        # holdout period
        if self.holdout_mask is not None:
            posterior_mean = posterior_mean[self.holdout_mask]

        return posterior_mean

    def fit_and_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """Fit the Meridian model and make predictions given new input data.

        The full dataset must be passed to `fit()`, since making out-of-sample predictions
        is only possible by way of specifying a holdout mask when sampling from the
        posterior.

        Args:
            train: Training data
            test: Test data

        Returns:
            Predicted values for the test period

        """
        train_and_test = pd.concat([train, test])
        self.fit(train_and_test, max_train_date=train[self.date_column].max())
        return self.predict()

    def get_channel_roi(
        self,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.Series:
        """Get channel ROI estimates.

        Args:
            start_date: Optional start date for ROI calculation
            end_date: Optional end date for ROI calculation

        Returns:
            Series containing ROI estimates for each channel

        """
        if not self.is_fitted or self.training_data is None or self.analyzer is None:
            raise RuntimeError("Model must be fit before computing ROI")

        # restrict ROI calculation to a particular period if start/end date args are
        # passed
        training_date_index = pd.to_datetime(self.training_data.kpi.time)
        roi_date_index = training_date_index.copy()
        if start_date:
            roi_date_index = roi_date_index[roi_date_index >= start_date]
        if end_date:
            roi_date_index = roi_date_index[roi_date_index < end_date]

        selected_times = [bool(date) for date in training_date_index.isin(roi_date_index)]

        # analyzer.roi() returns a tensor of shape (n_chains, n_draws, n_channels)
        rois_per_channel = np.mean(self.analyzer.roi(selected_times=selected_times), axis=(0, 1))

        rois = {}
        for channel, roi in zip(self.media_channels, rois_per_channel, strict=False):
            rois[channel] = float(roi)
        return pd.Series(rois)
