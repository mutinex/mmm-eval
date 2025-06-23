"""
PyMC MMM framework adapter.

N.B. expects control variables to be scaled to 0-1 using maxabs scaling.
"""

from typing import Dict

import pandas as pd
from pymc_marketing.mmm import MMM

from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.configs import PyMCConfig


class PyMCAdapter(BaseAdapter):
    def __init__(self, config: PyMCConfig):
        # Rehydrate the config dictionary
        self.config = config

        # Store explicitly needed pieces
        self.model_config = self.config.model_config.config
        self.fit_config = self.config.fit_config.config
        self.target_column = self.config.target_column
        self.response_col = self.config.response_column

        self.model = None
        self.trace = None
        self._channel_rois = None

    def fit(self, data: pd.DataFrame, metadata: dict = None):
        """Fit the model and compute ROIs."""
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        self.model = MMM(**self.model_config)
        self.trace = self.model.fit(X=X, y=y, **self.fit_config)

        self._compute_rois(data)
        self.is_fitted = True

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict the response variable for new data."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before prediction.")

        if self.response_col in data.columns:
            data = data.drop(columns=[self.response_col])
        return self.model.predict(data, extend_idata=False)

    def get_channel_roi(self) -> Dict[str, float]:
        """Return the ROIs for each channel."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before computing ROI.")
        return self._channel_rois

    def _compute_rois(self, data: pd.DataFrame):
        """Estimate ROIs based on the marginal contribution of each channel."""
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

        avg_rev_per_unit = cont_df[self.revenue_col] / cont_df[self.response_col]

        rois = {}
        for channel in self.channel_spend_cols:
            channel_revenue = cont_df[f"{channel}_units"] * avg_rev_per_unit
            # return as a percentage
            rois[channel] = (
                100 * (channel_revenue.sum() / cont_df[channel].sum() - 1).item()
            )

        self._channel_rois = rois
