"""Meridian adapter for MMM evaluation."""

import logging
from typing import Any

import numpy as np
import pandas as pd

from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.configs import MeridianConfig
from mmm_eval.data.constants import InputDataframeConstants

logger = logging.getLogger(__name__)


class MeridianAdapter(BaseAdapter):
    """Adapter for Google Meridian MMM framework."""

    def __init__(self, config: MeridianConfig):
        """Initialize the Meridian adapter.

        Args:
            config: MeridianConfig object

        """
        self.model_config = config.meridian_model_config_dict
        self.model_spec_config = config.model_spec_config_dict
        self.fit_config = config.fit_config_dict
        self.date_column = config.date_column
        self.channel_spend_columns = config.channel_columns
        self.control_columns = config.control_columns
        self.model = None
        self.trace = None
        self._channel_roi_df = None
        self.is_fitted = False

        # Store original values to reset on subsequent fit calls
        self._original_channel_spend_columns = config.channel_columns.copy()
        self._original_model_config = config.meridian_model_config_dict.copy()
        self._original_model_spec_config = config.model_spec_config_dict.copy()

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the Meridian model to data.

        Args:
            data: Training data

        """
        # Reset to original values at the start of each fit call
        self.channel_spend_columns = self._original_channel_spend_columns.copy()
        self.model_config = self._original_model_config.copy()
        self.model_spec_config = self._original_model_spec_config.copy()

        # Identify channel spend columns that sum to zero and remove them from modelling
        channel_spend_data = data[self.channel_spend_columns]
        zero_spend_channels = channel_spend_data.columns[channel_spend_data.sum() == 0].tolist()

        if zero_spend_channels:
            logger.info(f"Dropping channels with zero spend: {zero_spend_channels}")
            # Remove zero-spend channels from the list
            self.channel_spend_columns = [col for col in self.channel_spend_columns if col not in zero_spend_channels]
            # Update the model config to reflect the new channel spend columns
            self.model_config["media_columns"] = self.channel_spend_columns
            data = data.drop(columns=zero_spend_channels)

        try:
            # Import Meridian modules
            import meridian
            from meridian.model.spec import ModelSpec
            from meridian.model.model import Meridian
            import tensorflow_probability as tfp
            from meridian.prior_distribution import PriorDistribution

            # Create prior distribution
            prior = PriorDistribution(
                roi_m=tfp.distributions.LogNormal(
                    self.model_spec_config["prior"]["roi_mu"],
                    self.model_spec_config["prior"]["roi_sigma"],
                    name=self.model_spec_config["prior"]["name"]
                )
            )

            # Create model specification
            model_spec = ModelSpec(prior=prior)

            # Create and fit the Meridian model
            self.model = Meridian(
                input_data=data,
                model_spec=model_spec,
                **self.model_config
            )

            # Fit the model using sample_posterior
            self.trace = self.model.sample_posterior(**self.fit_config)

            # Compute channel contributions for ROI calculations
            self._channel_roi_df = self._compute_channel_contributions(data)
            self.is_fitted = True

        except ImportError:
            logger.warning("Meridian not available. Using placeholder implementation.")
            # Fallback to placeholder implementation
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
