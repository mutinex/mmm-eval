"""WIP: Meridian adapter for MMM evaluation.
"""

import logging

import numpy as np
import pandas as pd
import tensorflow_probability as tfp
from meridian import constants
from meridian.analysis.analyzer import Analyzer
from meridian.data import data_frame_input_data_builder as data_builder
from meridian.model import prior_distribution
from meridian.model.model import Meridian
from meridian.model.spec import ModelSpec

from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.configs import MeridianConfig
from mmm_eval.data.constants import InputDataframeConstants

logger = logging.getLogger(__name__)


REVENUE_PER_KPI_COL = "revenue_per_kpi"


def construct_meridian_data_object(df: pd.DataFrame, config: MeridianConfig) -> pd.DataFrame:
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
    if "population" in df.columns:
        builder = builder.with_population(df)

    # controls (non-intervenable, e.g. macroeconomics)
    builder = builder.with_controls(
        df, time_col=config.date_column, control_cols=input_data_builder_schema.control_columns
    )

    # add paid media
    # without impressions/reach/frequency: media_cols = media_spend_cols
    # with impressions: media_cols = impressions cols
    # with reach/frequency: media_cols = use .with_reach() instead
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


def construct_holdout_mask(max_train_date: pd.Timestamp, time_index, geo_index):
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

        # response and revenue columns are constants so don't need to be set here
        # population, if provided, needs to be called "population" in the DF
        self.date_column = config.date_column
        self.channel_spend_columns = self.input_data_builder_schema.channel_spend_columns
        self.media_channels = self.input_data_builder_schema.media_channels

        self.model = None
        self.trace = None
        self.is_fitted = False

    def fit(self, data: pd.DataFrame, max_train_date: pd.Timestamp | None = None) -> None:
        """Fit the Meridian model to data.

        Args:
            data: Training data

        """
        # build Meridian data object
        self.training_data = construct_meridian_data_object(data, self.config)
        self.max_train_date = max_train_date

        # parse Prior object if provided
        model_spec_kwargs = dict(self.config.model_spec_config)
        if prior_spec := self.config.model_spec_config.prior:
            # FIXME: make the functional form configurable
            prior_object = prior_distribution.PriorDistribution(
                roi_m=tfp.distributions.LogNormal(prior_spec.roi_mu, prior_spec.roi_sigma, name=constants.ROI_M)
            )
            model_spec_kwargs["prior"] = prior_object

        # if max train date is provided, construct a mask that is True for all dates before max_train_date
        self.holdout_mask = None
        if self.max_train_date:
            self.holdout_mask = construct_holdout_mask(
                self.max_train_date, self.training_data.kpi.time, self.training_data.kpi.geo
            )
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

        # Compute channel contributions for ROI calculations
        self.analyzer = Analyzer(self.model)
        self.is_fitted = True

    def predict(self) -> np.ndarray:
        """Make predictions using the fitted model.

        Args:
            data: Input data for prediction

        Returns:
            Predicted values

        """
        if not self.is_fitted:
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
        """Fit the Meridian model and make predictions.

        Args:
            train: Training data
            test: Test data
        
        """
        # FIXME: ensure the adapter is reset to a fresh state after predict is called
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
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before computing ROI")

        training_date_index = pd.to_datetime(self.training_data.kpi.time)
        roi_date_index = training_date_index.copy()
        if start_date:
            roi_date_index = roi_date_index[roi_date_index >= start_date]
        if end_date:
            roi_date_index = roi_date_index[roi_date_index < end_date]

        selected_times = [bool(e) for e in training_date_index.isin(roi_date_index)]

        # (n_chains, n_draws, n_channels)
        rois_per_channel = np.mean(self.analyzer.roi(selected_times=selected_times), axis=(0, 1))

        rois = {}
        for channel, roi in zip(self.media_channels, rois_per_channel, strict=False):
            rois[channel] = float(roi)
        return pd.Series(rois)
