"""
PyMC MMM framework adapter.

N.B. we expect control variables to be scaled to 0-1 using maxabs scaling BEFORE being
passed to the PyMCAdapter.
"""

import logging
from typing import Dict, Optional

import pandas as pd
from pymc_marketing.mmm import MMM

from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.adapters.experimental.schemas import PyMCConfigSchema

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PyMCAdapter(BaseAdapter):
    def __init__(self, config: dict):
        """Initialize the PyMCAdapter.

        Args:
            config: Dictionary containing the configuration for the PyMCAdapter adhering
                to the PyMCConfigSchema.
        """
        super().__init__(config)
        PyMCConfigSchema.model_validate(config)

        self.date_col = config["date_column"]
        self.channel_spend_cols = config["channel_columns"]

        # Pop out items that are not needed for MMM constructor
        self.response_col = config.pop("response_column")
        self.revenue_col = config.pop("revenue_column")
        self.fit_kwargs = config.pop("fit_kwargs", {})

        # Everything else is passed to MMM constructor
        self.model_kwargs = config.items()
        
        # initialise fields set in `fit`
        self.model = None
        self.trace = None
        self._channel_roi_df = None

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the model to data.

        Args:
            data: DataFrame containing the training data adhering to the PyMCInputDataSchema.
        """
        # TODO: this may be redundant after an upstream schema check, remove if so
        _check_columns_in_data(data, [self.date_col, self.channel_spend_cols, self.response_col, self.revenue_col])
            
        X = data.drop(columns=[self.response_col, self.revenue_col])
        y = data[self.response_col]

        self.model = MMM(**self.model_kwargs)
        self.trace = self.model.fit(X=X, y=y, **self.fit_kwargs)

        self._channel_roi_df = self._compute_channel_contributions(data)
        self.is_fitted = True

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict the response variable for new data."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before prediction.")
        
        # TODO: this may be redundant after an upstream schema check, remove if so
        _check_columns_in_data(data, [self.date_col, self.channel_spend_cols])

        if self.response_col in data.columns:
            data = data.drop(columns=[self.response_col])
        return self.model.predict(data, extend_idata=False)

    def get_channel_roi(
        self,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        """Return the ROIs for each channel, optionally within a given date range."""
        if not self.is_fitted:
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
        channel_cont = self.model.compute_channel_contribution_original_scale().mean(
            dim=["chain", "draw"]
        )

        # FIXME: infer the index/column names from the data
        channel_response_cols = [f"{col}_units" for col in self.channel_spend_cols]
        cont_df = pd.DataFrame(
            channel_cont,
            columns=channel_response_cols,
            index=channel_cont["date"].values,
        )
        cont_df = pd.merge(
            cont_df,
            data[
                [
                    self.date_col,
                    self.response_col,
                    self.revenue_col,
                    *self.channel_spend_cols,
                ]
            ].set_index(self.date_col),
            left_index=True,
            right_index=True,
        )

        return cont_df

    def _calculate_rois(self, contribution_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate ROIs from a contribution DataFrame.

        Args:
            contribution_df: DataFrame containing channel contributions and spend data

        Returns:
            dictionary mapping channel names to ROI percentages.
        """
        avg_rev_per_unit = (
            contribution_df[self.revenue_col] / contribution_df[self.response_col]
        )

        rois = {}
        for channel in self.channel_spend_cols:
            channel_revenue = contribution_df[f"{channel}_units"] * avg_rev_per_unit
            # return as a percentage
            rois[channel] = (
                100
                * (channel_revenue.sum() / contribution_df[channel].sum() - 1).item()
            )

        return rois


def _validate_start_end_dates(
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
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
        raise ValueError(
            f"Start date must be before end date, but got start_date={start_date} and end_date={end_date}"
        )

    if start_date is not None and start_date < date_range.min():
        logger.info(
            f"Start date is before the first date in the training data: {date_range.min()}"
        )

    if end_date is not None and end_date > date_range.max():
        logger.info(
            f"End date is after the last date in the training data: {date_range.max()}"
        )


def _check_columns_in_data(data: pd.DataFrame, column_sets: list[str], ) -> None:
    """Check if column(s) are in a dataframe."""
    for column_set in column_sets:
        if set(column_set) - set(data.columns):
            raise ValueError(f"Not all column(s) in `{column_set}` found in data, which has columns `{data.columns}`")