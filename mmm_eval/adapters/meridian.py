"""WIP: Meridian adapter for MMM evaluation.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import meridian
from meridian.model.spec import ModelSpec
from meridian.model.model import Meridian
from meridian.data import data_frame_input_data_builder as data_builder

from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.configs import MeridianConfig
from mmm_eval.data.constants import InputDataframeConstants

logger = logging.getLogger(__name__)


def construct_meridian_data_object(df: pd.DataFrame, config: MeridianConfig) -> pd.DataFrame:
    # KPI, population, and control variables
    builder = (
        data_builder.DataFrameInputDataBuilder(kpi_type='non_revenue')
            .with_kpi(df, kpi_col=InputDataframeConstants.RESPONSE_COL)
            .with_revenue_per_kpi(df, revenue_col=InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL)
    )
    if "population" in df.columns:
        builder = builder.with_population(df)
    
    # controls (non-intervenable, e.g. macroeconomics)
    builder = builder.with_controls(df, control_cols=config.control_columns)

    # add paid media
    # TODO: add support for impressions/reach/frequency
    builder = builder.with_media(
        df,
        media_spend_cols=[f"{channel}_spend" for channel in config.channel_columns],
        media_channels=config.channel_columns,
    )

    # organic media
    if config.organic_media_columns:
        builder = builder.with_organic_media(
            df,
            organic_media_cols=config.organic_media_columns,
            organic_media_channels=config.organic_media_channels,
            media_time_col=config.date_column,
        )

    # non-media treatments (anything that is "intervenable", e.g. pricing/promotions)
    if config.non_media_treatments_columns:
        builder = builder.with_non_media_treatments(
            df,
            non_media_treatments_cols=config.non_media_treatments_columns,
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

        # response and revenue columns are constants so don't need to be set here
        # population, if provided, needs to be called "population" in the DF
        self.date_column = config.date_column
        self.channel_spend_columns = config.channel_columns
        # impressions, reach, frequency, etc.
        self.channel_metric_columns = config.channel_metric_columns or None
        self.organic_media_columns = config.organic_media_columns
        self.organic_media_channels = config.organic_media_channels
        self.non_media_treatments_columns = config.non_media_treatments_columns
        self.non_media_treatments_channels = config.non_media_treatments_channels

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
        data_object = construct_meridian_data_object(data, self.config)

        # Create and fit the Meridian model
        self.model = Meridian(
            input_data=data_object,
            model_spec=self.config.model_spec_config,
        )
        self.trace = self.model.sample_posterior(**self.config.fit_config)

        # Compute channel contributions for ROI calculations
        self._channel_roi_df = self._compute_channel_contributions(data)
        self.is_fitted = True


    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted model.

        Args:
            data: Input data for prediction

        Returns:
            Predicted values

        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before prediction")

        if self.model is None:
            # Placeholder implementation if Meridian is not available
            logger.warning("Using placeholder prediction - Meridian model not available")
            return np.random.normal(100, 10, len(data))

        try:
            # Remove response column if present
            if InputDataframeConstants.RESPONSE_COL in data.columns:
                data = data.drop(columns=[InputDataframeConstants.RESPONSE_COL])

            # Make predictions using Meridian model
            predictions = self.model.predict(data)
            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            # Fallback to placeholder predictions
            return np.random.normal(100, 10, len(data))

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

        if self._channel_roi_df is None:
            logger.warning("Channel ROI data not available. Returning placeholder ROIs.")
            # Return placeholder ROIs
            return pd.Series(
                {channel: np.random.uniform(0.5, 2.0) for channel in self.channel_spend_columns}
            )

        # Validate date range
        self._validate_start_end_dates(start_date, end_date, self._channel_roi_df.index)

        # Filter the contribution DataFrame by date range
        date_range_df = self._channel_roi_df.loc[start_date:end_date]

        if date_range_df.empty:
            raise ValueError(f"No data found for date range {start_date} to {end_date}")

        return pd.Series(self._calculate_rois(date_range_df))

    def _compute_channel_contributions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute channel contributions and return the DataFrame for ROI calculations.

        Args:
            data: Input DataFrame containing the training data

        Returns:
            DataFrame containing channel contributions and spend data

        """
        if self.model is None:
            logger.warning("Meridian model not available. Using placeholder contributions.")
            # Create placeholder contributions
            contribution_df = pd.DataFrame(
                np.random.uniform(10, 100, (len(data), len(self.channel_spend_columns))),
                columns=[f"{col}_response_units" for col in self.channel_spend_columns],
                index=data.index
            )
            
            # Add original data columns
            data_subset = data.filter(
                items=[
                    self.date_column,
                    InputDataframeConstants.RESPONSE_COL,
                    InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL,
                    *self.channel_spend_columns,
                ]
            ).set_index(self.date_column)

            contribution_df = pd.merge(
                contribution_df,
                data_subset,
                left_index=True,
                right_index=True,
            )

            return contribution_df

        try:
            # Compute channel contributions using Meridian model
            # This would need to be implemented based on the actual Meridian API
            # For now, return placeholder data
            contribution_df = pd.DataFrame(
                np.random.uniform(10, 100, (len(data), len(self.channel_spend_columns))),
                columns=[f"{col}_response_units" for col in self.channel_spend_columns],
                index=data.index
            )
            
            # Add original data columns
            data_subset = data.filter(
                items=[
                    self.date_column,
                    InputDataframeConstants.RESPONSE_COL,
                    InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL,
                    *self.channel_spend_columns,
                ]
            ).set_index(self.date_column)

            contribution_df = pd.merge(
                contribution_df,
                data_subset,
                left_index=True,
                right_index=True,
            )

            return contribution_df

        except Exception as e:
            logger.error(f"Error computing channel contributions: {e}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()

    def _calculate_rois(self, contribution_df: pd.DataFrame) -> dict[str, float]:
        """Calculate ROIs from a contribution DataFrame.

        Args:
            contribution_df: DataFrame containing channel contributions and spend data

        Returns:
            Dictionary mapping channel names to ROI percentages

        """
        # Calculate average revenue per unit
        avg_rev_per_unit = np.divide(
            contribution_df[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL],
            contribution_df[InputDataframeConstants.RESPONSE_COL],
            out=np.full_like(contribution_df[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL], np.nan),
            where=contribution_df[InputDataframeConstants.RESPONSE_COL] != 0,
        )

        # Forward fill three periods, then fill any remaining with the mean
        avg_rev_per_unit = pd.Series(avg_rev_per_unit, index=contribution_df.index)
        original_mean = avg_rev_per_unit.mean()
        avg_rev_per_unit = avg_rev_per_unit.fillna(method="ffill", limit=3)
        avg_rev_per_unit = avg_rev_per_unit.fillna(original_mean)

        rois = {}
        for channel in self.channel_spend_columns:
            total_spend = contribution_df[channel].sum()
            # Handle edge case where total spend is zero
            if total_spend == 0:
                rois[channel] = np.nan
                continue

            channel_revenue = contribution_df[f"{channel}_response_units"] * avg_rev_per_unit
            # Return as a percentage
            rois[channel] = 100 * (channel_revenue.sum() / total_spend - 1).item()

        return rois

    def _validate_start_end_dates(
        self,
        start_date: pd.Timestamp | None,
        end_date: pd.Timestamp | None,
        date_range: pd.DatetimeIndex,
    ) -> None:
        """Validate start/end dates passed to `get_channel_roi`.

        Args:
            start_date: Left bound for calculating ROI estimates
            end_date: Right bound for calculating ROI estimates
            date_range: Date range of the training data which will be subset

        Raises:
            ValueError: If start_date is not before end_date

        """
        if start_date is not None and end_date is not None and start_date >= end_date:
            raise ValueError(f"Start date must be before end date, but got start_date={start_date} and end_date={end_date}")

        if start_date is not None and start_date < date_range.min():
            logger.info(f"Start date is before the first date in the training data: {date_range.min()}")

        if end_date is not None and end_date > date_range.max():
            logger.info(f"End date is after the last date in the training data: {date_range.max()}")
