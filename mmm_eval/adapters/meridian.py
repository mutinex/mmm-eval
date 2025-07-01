"""WIP: Meridian adapter for MMM evaluation.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import tensorflow_probability as tfp
from meridian.model.spec import ModelSpec
from meridian.model.model import Meridian
from meridian.data import data_frame_input_data_builder as data_builder
from meridian.model import prior_distribution
from meridian import constants
from meridian.analysis.analyzer import Analyzer

from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.configs import MeridianConfig
from mmm_eval.data.constants import InputDataframeConstants

logger = logging.getLogger(__name__)


def construct_meridian_data_object(df: pd.DataFrame, config: MeridianConfig) -> pd.DataFrame:
    model_schema = config.meridian_model_config

    # KPI, population, and control variables
    builder = (
        data_builder.DataFrameInputDataBuilder(kpi_type='non_revenue')
            .with_kpi(df, time_col=config.date_column, kpi_col=InputDataframeConstants.RESPONSE_COL)
            .with_revenue_per_kpi(df, time_col=config.date_column, revenue_per_kpi_col=config.revenue_column)
    )
    if "population" in df.columns:
        builder = builder.with_population(df)
    
    # controls (non-intervenable, e.g. macroeconomics)
    builder = builder.with_controls(df, time_col=config.date_column,
                                    control_cols=model_schema.control_columns)

    # add paid media
    # without impressions/reach/frequency: media_cols = media_spend_cols
    # with impressions: media_cols = impressions cols
    # with reach/frequency: media_cols = use .with_reach() instead
    if model_schema.channel_reach_columns:
        builder = builder.with_reach(
            df,
            reach_cols=model_schema.channel_reach_columns,
            frequency_cols=model_schema.channel_frequency_columns,
            rf_spend_cols=model_schema.channel_spend_columns,
            rf_channels=model_schema.media_channels,
            time_col=config.date_column,
        )
    else:
        media_cols = model_schema.channel_impressions_columns or model_schema.channel_spend_columns
        builder = builder.with_media(
            df,
            media_cols=media_cols,
            media_spend_cols=model_schema.channel_spend_columns,
            media_channels=model_schema.media_channels,
            time_col=config.date_column,
        )

    # organic media
    if model_schema.organic_media_columns:
        builder = builder.with_organic_media(
            df,
            organic_media_cols=model_schema.organic_media_columns,
            organic_media_channels=model_schema.organic_media_channels,
            media_time_col=config.date_column,
        )

    # non-media treatments (anything that is "intervenable", e.g. pricing/promotions)
    if model_schema.non_media_treatment_columns:
        builder = builder.with_non_media_treatments(
            df,
            non_media_treatment_cols=model_schema.non_media_treatment_columns,
            time_col=config.date_column,
        )

    return builder.build()


class MeridianAdapter(BaseAdapter):
    """Adapter for Google Meridian MMM framework."""

    def __init__(self, config: MeridianConfig):
        """Initialize the Meridian adapter.

        Meridian needs the following

        Args:
            config: MeridianConfig object
        """
        self.config = config
        self.model_schema = config.meridian_model_config

        # response and revenue columns are constants so don't need to be set here
        # population, if provided, needs to be called "population" in the DF
        self.date_column = config.date_column
        self.channel_spend_columns = self.model_schema.channel_spend_columns
        self.media_channels = self.model_schema.media_channels

        self.model = None
        self.trace = None
        self._channel_roi_df = None
        self.is_fitted = False

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the Meridian model to data.

        Args:
            data: Training data

        """
        # build Meridian data object
        self.training_data = construct_meridian_data_object(data, self.config)

        # parse Prior object if provided
        model_spec_kwargs = dict(self.config.model_spec_config)
        if prior_spec := self.config.model_spec_config.prior:
            # FIXME: make the functional form configurable
            prior_object = prior_distribution.PriorDistribution(
                roi_m=tfp.distributions.LogNormal(prior_spec.roi_mu, prior_spec.roi_sigma,
                                                  name=constants.ROI_M)
            )
            model_spec_kwargs["prior"] = prior_object

        # Create and fit the Meridian model
        model_spec = ModelSpec(
            **model_spec_kwargs
        )
        self.model = Meridian(
            input_data=self.training_data,
            model_spec=model_spec,
        )
        self.trace = self.model.sample_posterior(**dict(self.config.fit_config))

        # Compute channel contributions for ROI calculations
        #self._channel_roi_df = self._compute_channel_contributions(data)
        self.analyzer = Analyzer(self.model)
        self.is_fitted = True


    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted model.

        DOESN'T CURRENTLY WORK

        Args:
            data: Input data for prediction

        Returns:
            Predicted values

        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before prediction")
        
        # build Meridian data object
        data_object = construct_meridian_data_object(data, self.config)

        if InputDataframeConstants.RESPONSE_COL in data.columns:
            data = data.drop(columns=[InputDataframeConstants.RESPONSE_COL])
        
        # FIXME: this doesn't consider adstock carryover from the training set
        # shape (n_chans, n_draws, n_times)
        preds_tensor = self.analyzer.expected_outcome(new_data=data_object,
                                                 aggregate_geos=True, aggregate_times=False)
        return np.mean(preds_tensor, axis=(0, 1))
        

    def get_channel_roi(
        self,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> dict[str, float]:
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
        for channel, roi in zip(self.media_channels, rois_per_channel):
            rois[channel] = float(roi)
        return rois