"""PyMC MMM framework adapter.

N.B. we expect control variables to be scaled to [-1, 1] using maxabs scaling BEFORE being
passed to the PyMCAdapter.
"""

import logging

import numpy as np
import pandas as pd
from pymc_marketing.mmm import MMM

from mmm_eval.adapters.base import BaseAdapter, PrimaryMediaRegressor
from mmm_eval.configs import PyMCConfig
from mmm_eval.data.constants import InputDataframeConstants

logger = logging.getLogger(__name__)


class PyMCAdapter(BaseAdapter):
    def __init__(self, config: PyMCConfig):
        """Initialize the PyMCAdapter.

        Args:
            config: PyMCConfig object

        """
        self.model_kwargs = config.pymc_model_config_dict
        self.fit_kwargs = config.fit_config_dict
        self.predict_kwargs = config.predict_config_dict
        self.date_column = config.date_column
        self.channel_spend_columns = config.channel_columns
        self.control_columns = config.control_columns
        self.model = None
        self.trace = None
        self._channel_roi_df = None
        self.is_fitted = False

        # Store original values to reset on subsequent fit calls
        self._original_channel_spend_columns = config.channel_columns.copy()
        self._original_model_kwargs = config.pymc_model_config_dict.copy()

    @property
    def media_channels(self) -> list[str]:
        """Return the channel names used by this adapter.

        For PyMC, this returns the channel_spend_columns which are used as the channel names
        in ROI results.

        Returns
            List of channel names

        """
        return self.channel_spend_columns

    @property
    def primary_media_regressor_type(self) -> PrimaryMediaRegressor:
        """Return the type of primary media regressors used by this adapter.

        For PyMC, this is always SPEND since the PyMC adapter uses spend as the primary
        regressor.

        Returns
            PrimaryMediaRegressor.SPEND

        """
        return PrimaryMediaRegressor.SPEND

    @property
    def primary_media_regressor_columns(self) -> list[str]:
        """Return the primary media regressor columns that should be perturbed in tests.

        For PyMC, this is always the `channel_spend_columns` since the PyMC adapter uses
        spend as the primary regressor in the model.

        Returns
            List of channel spend column names

        """
        return self.channel_spend_columns

    # TODO: require users to pass a mapping of channel names to spend columns
    def get_channel_names(self) -> list[str]:
        """Get the channel names that would be used as the index in get_channel_roi results.

        For PyMC, this returns the `channel_spend_columns` which are used as the index
        in the ROI results.

        Returns
            List of channel names

        """
        return self.channel_spend_columns

    def copy(self) -> "PyMCAdapter":
        """Create a deep copy of this adapter with all configuration.

        Returns:
            A new PyMCAdapter instance with the same configuration

        """
        # Create a new config with copied values
        new_config = PyMCConfig(
            date_column=self.date_column,
            channel_columns=self._original_channel_spend_columns.copy(),
            control_columns=self.control_columns.copy(),
            pymc_model_config_dict=self._original_model_kwargs.copy(),
            fit_config_dict=self.fit_kwargs.copy(),
            predict_config_dict=self.predict_kwargs.copy(),
        )
        
        return PyMCAdapter(new_config)

    def add_channels(self, new_channel_names: list[str]) -> dict[str, list[str]]:
        """Add new channels to the adapter's configuration.

        Args:
            new_channel_names: List of new channel names to add

        Returns:
            Dictionary mapping channel names to lists of column names that were added for each channel.
            For PyMC, channel names are the same as column names.

        """
        if self.is_fitted:
            raise RuntimeError("Cannot add channels to a fitted adapter")
        
        # For PyMC, channel names are the same as column names
        # Add to the current channel lists
        self.channel_spend_columns.extend(new_channel_names)
        
        # Update the original lists as well (for future copy operations)
        self._original_channel_spend_columns.extend(new_channel_names)
        
        # Return mapping of channel names to column names (they're the same for PyMC)
        return {channel_name: [channel_name] for channel_name in new_channel_names}

    def get_primary_media_regressor_columns_for_channels(self, channel_names: list[str]) -> list[str]:
        """Get the primary media regressor columns for specific channels.

        For PyMC, the primary media regressor columns are the same as the channel names.

        Args:
            channel_names: List of channel names to get regressor columns for

        Returns:
            List of column names that are used as primary media regressors for the given channels

        """
        return channel_names

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the model and compute ROIs.

        Args:
            data: DataFrame containing the training data adhering to the PyMCInputDataSchema.

        """
        # Reset to original values at the start of each fit call
        self.channel_spend_columns = self._original_channel_spend_columns.copy()
        self.model_kwargs = self._original_model_kwargs.copy()

        # Identify channel spend columns that sum to zero and remove them from modelling.
        # We cannot reliably make any prediction based on these channels when making
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

        X = data.drop(columns=[InputDataframeConstants.RESPONSE_COL, InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL])
        y = data[InputDataframeConstants.RESPONSE_COL]

        self.model = MMM(**self.model_kwargs)
        self.trace = self.model.fit(X=X, y=y, **self.fit_kwargs)

        self._channel_roi_df = self._compute_channel_contributions(data)
        self.is_fitted = True

    def predict(self, data: pd.DataFrame | None = None) -> np.ndarray:
        """Predict the response variable for new data.

        Args:
            data: Input data for prediction. This parameter is required for PyMC
                predictions and cannot be None.

        Returns:
            Predicted values

        Raises:
            RuntimeError: If model is not fitted
            ValueError: If data is None (PyMC requires data for prediction)

        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fit before prediction.")

        if data is None:
            raise ValueError("PyMC adapter requires data for prediction")

        if InputDataframeConstants.RESPONSE_COL in data.columns:
            data = data.drop(columns=[InputDataframeConstants.RESPONSE_COL])
        predictions = predictions = self.model.predict(
            data, extend_idata=False, include_last_observations=True, **self.predict_kwargs
        )
        return predictions

    def fit_and_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """Fit on training data and make predictions on test data.

        Arguments:
            train: training dataset
            test: test dataset

        Returns:
            model predictions.

        """
        self.fit(train)
        return self.predict(test)

    def fit_and_predict_in_sample(self, data: pd.DataFrame) -> np.ndarray:
        """Fit the model on data and return predictions for the same data.

        Args:
            data: dataset to train model on and make predictions for

        Returns:
            Predicted values for the training data.

        """
        self.fit(data)
        if self.model is None:
            raise RuntimeError("Model must be fit before prediction.")
        return self.model.predict(data, extend_idata=False, **self.predict_kwargs)

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

        _validate_start_end_dates(start_date, end_date, pd.DatetimeIndex(self._channel_roi_df.index))

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
        data = data.filter(
            items=[
                self.date_column,
                InputDataframeConstants.RESPONSE_COL,
                InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL,
                *self.channel_spend_columns,
            ]
        ).set_index(self.date_column)

        contribution_df = pd.merge(
            contribution_df,
            data,
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
            contribution_df[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL],
            contribution_df[InputDataframeConstants.RESPONSE_COL],
            out=np.full_like(contribution_df[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL], np.nan),
            # we check upstream whether exactly one of response and revenue are zero,
            # so we can safely assume that 0 response -> 0 revenue
            where=contribution_df[InputDataframeConstants.RESPONSE_COL] != 0,
        )

        # forward fill three periods, then fill any remaining with the mean
        avg_rev_per_unit = pd.Series(avg_rev_per_unit, index=contribution_df.index)
        original_mean = avg_rev_per_unit.mean()
        avg_rev_per_unit = avg_rev_per_unit.fillna(method="ffill", limit=3)
        avg_rev_per_unit = avg_rev_per_unit.fillna(original_mean)

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

    if start_date is not None and not pd.isna(date_range.min()) and start_date < date_range.min():
        logger.info(f"Start date is before the first date in the training data: {date_range.min()}")

    if end_date is not None and not pd.isna(date_range.max()) and end_date > date_range.max():
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
