"""PyMC MMM framework adapter.

N.B. we expect control variables to be scaled to 0-1 using maxabs scaling BEFORE being
passed to the PyMCAdapter.
"""

import logging

import numpy as np
import pandas as pd
from pymc_marketing.mmm import MMM

from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.configs import PyMCConfig
from mmm_eval.adapters.experimental.schemas import PyMCConfigSchema
from mmm_eval.data.constants import InputDataframeConstants
from mmm_eval.utils import PyMCConfigRehydrator

logger = logging.getLogger(__name__)


class PyMCAdapter(BaseAdapter):
    def __init__(self, config: PyMCConfig):
        """Initialize the PyMCAdapter.

        Args:
            config: PyMCConfig object

        """
        self.model_kwargs = config.pymc_model_config_dict
        self.fit_kwargs = config.fit_config_dict
        self.revenue_column = config.revenue_column
        self.response_column = config.response_column
        self.date_column = config.date_column
        self.channel_spend_columns = config.channel_columns
        self.control_columns = config.control_columns
        self.model = None
        self.trace = None
        self._channel_roi_df = None
        self.is_fitted = False

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the model and compute ROIs.

        Args:
            data: DataFrame containing the training data adhering to the PyMCInputDataSchema.

        """
        # Identify channel spend columns that sum to zero and remove them from modelling.
        # We cannot reliabily make any prediction based on these channels when making
        # predictions on new data.
        channel_spend_data = data[self.channel_spend_columns]
        zero_spend_channels = channel_spend_data.columns[channel_spend_data.sum() == 0].tolist()

        if zero_spend_channels:
            logger.info(f"Dropping channels with zero spend: {zero_spend_channels}")
            # Remove zero-spend channels from the list passed to the MMM constructor
            self.channel_spend_columns = [col for col in self.channel_spend_columns if col not in zero_spend_channels]
            # also update the model config field to reflect the new channel spend columns
            self.model_kwargs["channel_columns"] = self.channel_spend_columns

            # Check for vector priors that might cause shape mismatches
            _check_vector_priors_when_dropping_channels(self.model_kwargs["model_config"], zero_spend_channels)

            data = data.drop(columns=zero_spend_channels)

        X = data.drop(columns=[self.response_column, self.revenue_column])
        y = data[self.response_column]

        self.model = MMM(**self.model_kwargs)
        self.trace = self.model.fit(X=X, y=y, **self.fit_kwargs)

        self._channel_roi_df = self._compute_channel_contributions(data)
        self.is_fitted = True

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict the response variable for new data.

        Args:
            data: Input data for prediction

        Returns:
            Predicted values

        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fit before prediction.")

        if self.response_column in data.columns:
            data = data.drop(columns=[self.response_column])

        predictions = self.model.predict(data, extend_idata=False, include_last_observations=True)
        return predictions

    def get_channel_roi(
        self,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.Series:
        """Return the ROIs for each channel, optionally within a given date range.

        Args:
            start_date: Optional start date for ROI calculation
            end_date: Optional end date for ROI calculation

        Returns:
            Series containing ROI estimates for each channel

        """
        if not self.is_fitted or self._channel_roi_df is None:
            raise RuntimeError("Model must be fit before computing ROI.")

        _validate_start_end_dates(start_date, end_date, self._channel_roi_df.index)

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
            raise RuntimeError("Model must be fit before computing channel contributions")

        channel_contribution = self.model.compute_channel_contribution_original_scale().mean(dim=["chain", "draw"])

        contribution_df = pd.DataFrame(
            channel_contribution,
            columns=channel_contribution["channel"].to_numpy(),
            index=channel_contribution["date"].to_numpy(),
        )
        contribution_df.columns = [f"{col}_response_units" for col in self.channel_spend_columns]
        contribution_df = pd.merge(
            contribution_df,
            data[
                [
                    self.date_column,
                    InputDataframeConstants.RESPONSE_COL,
                    InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL,
                    *self.channel_spend_cols,
                ]
            ].set_index(self.date_column),
            left_index=True,
            right_index=True,
        )

        return contribution_df

    def _calculate_rois(self, contribution_df: pd.DataFrame) -> dict[str, float]:
        """Calculate ROIs from a contribution DataFrame.

        Args:
            contribution_df: DataFrame containing channel contributions and spend data

        Returns:
            dictionary mapping channel names to ROI percentages.

        """
        # if revenue is used as the response, this quotient will be 1, and the math for
        # calculating channel revenue will still be correct
        avg_rev_per_unit = np.divide(
            contribution_df[self.revenue_column],
            contribution_df[self.response_column],
            out=np.zeros_like(contribution_df[self.revenue_column]),
            where=contribution_df[self.response_column] != 0,
        )

        rois = {}
        for channel in self.channel_spend_columns:
            total_spend = contribution_df[channel].sum()
            # handle edge case where total spend is zero for the time period selected
            # (possible to have non-zero attribution due to adstock effect)
            if total_spend == 0:
                rois[channel] = np.nan
                continue

            channel_revenue = contribution_df[f"{channel}_response_units"] * avg_rev_per_unit
            # return as a percentage
            rois[channel] = 100 * (channel_revenue.sum() / total_spend - 1).item()

        return rois


def _validate_start_end_dates(
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    date_range: pd.DatetimeIndex,
) -> None:
    """Validate start/end dates passed to `get_channel_roi`.

    Args:
        start_date: left bound for calculating ROI estimates
        end_date: right bound for calculating ROI estimates
        date_range: date range of the training data which will be subset

    Raises:
        ValueError: If start_date is not before end_date.

    """
    if start_date is not None and end_date is not None and start_date >= end_date:
        raise ValueError(f"Start date must be before end date, but got start_date={start_date} and end_date={end_date}")

    if start_date is not None and start_date < date_range.min():
        logger.info(f"Start date is before the first date in the training data: {date_range.min()}")

    if end_date is not None and end_date > date_range.max():
        logger.info(f"End date is after the last date in the training data: {date_range.max()}")


def _check_vector_priors_when_dropping_channels(model_config: dict, zero_spend_channels: list[str]) -> None:
    """Check for vector priors that might cause shape mismatches when channels are dropped.

    Args:
        model_config: The model configuration dictionary containing priors
        zero_spend_channels: List of channels being dropped due to zero spend

    Warns:
        UserWarning: If vector priors are found that might cause shape mismatches

    """
    if not model_config or not zero_spend_channels:
        return

    vector_priors = []

    # Check common priors that might be vectors
    for prior_name in ["saturation_beta", "gamma_media", "beta_media"]:
        if prior_name in model_config:
            prior = model_config[prior_name]
            if hasattr(prior, "sigma"):
                sigma = prior.sigma
                if hasattr(sigma, "__len__") and len(sigma) > 1:
                    vector_priors.append(f"{prior_name}.sigma")
            if hasattr(prior, "mu"):
                mu = prior.mu
                if hasattr(mu, "__len__") and len(mu) > 1:
                    vector_priors.append(f"{prior_name}.mu")

    if vector_priors:
        logger.warning(
            "Found vector priors that may cause shape mismatches given that channels are "
            f"being dropped due to zero spend: {vector_priors}. "
            "Consider using scalar priors instead to avoid PyTensor broadcasting errors."
        )
