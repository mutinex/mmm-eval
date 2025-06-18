"""
PyMC MMM framework adapter.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from .base import BaseAdapter

import pandas as pd
from pymc_marketing.mmm import MMM
from typing import Dict

from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.utils import PyMCConfigRehydrator
from mmm_eval.adapters.schemas import PyMCConfigSchema

class PyMCAdapter(BaseAdapter):
    def __init__(self, config: dict):
        # Create a copy of the config to avoid modifying the original
        self.config = PyMCConfigRehydrator(config.copy()).rehydrate_config()
        
        # Validate config against schema
        try:
            PyMCConfigSchema(**self.config)
        except Exception as e:
            raise ValueError(f"Invalid config: {str(e)}")
        
        # Store explicitly needed pieces
        self.response_col = self.config["response_column"]
        self.date_col = self.config["date_column"]
        self.channel_spend_cols = self.config["channel_columns"]
        self.revenue_col = self.config.pop("revenue_column") if "revenue_column" in self.config else None
        
        # Pass everything else (after extracting response_col) to MMM constructor
        self.model_kwargs = {
            k: v for k, v in self.config.items()
            if k != "response_column" and
            k != "fit_kwargs"
        }
        self.fit_kwargs = self.config.get("fit_kwargs", {})

        self.model = None
        self.trace = None
        self._channel_rois = None

    def fit(self, data: pd.DataFrame, metadata: dict = None):
        X = data.drop(columns=[self.response_col, self.revenue_col])
        print(X.columns)
        y = data[self.response_col]

        self.model = MMM(**self.model_kwargs)
        self.trace = self.model.fit(X=X, y=y, **self.fit_kwargs)

        self._compute_rois(data)
        self.is_fitted = True

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before prediction.")

        X_new = data.drop(columns=[self.response_col])
        prediction = self.model.predict(X_new, extend_idata=False)
        return prediction
        #return prediction.mean(axis=0)  # returning posterior predictive mean

    def get_channel_roi(self) -> Dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before computing ROI.")
        return self._channel_rois
    

    def _compute_rois(self, data: pd.DataFrame):
        """
        Estimate ROI = marginal contribution / spend.
        PyMC-Marketing supports posterior samples of β weights.
        We'll use the mean of the posterior coefficients.
        """
        channel_cont = self.model.compute_channel_contribution_original_scale().mean(dim=["chain", "draw"])
        
        # FIXME: infer the index/column names from the data
        channel_response_cols = [f"{col}_units" for col in self.channel_spend_cols]
        cont_df = pd.DataFrame(channel_cont, columns=channel_response_cols, index=channel_cont["date"].values)
        cont_df = pd.merge(cont_df, data[[self.date_col, self.response_col, self.revenue_col,
                                          *self.channel_spend_cols]].set_index(self.date_col), left_index=True, right_index=True)

        avg_rev_per_unit = cont_df[self.revenue_col]/cont_df[self.response_col]

        rois = {}
        for channel in self.channel_spend_cols:
            channel_revenue = cont_df[f"{channel}_units"]*avg_rev_per_unit
            rois[channel] = (channel_revenue.sum()/cont_df[channel].sum() - 1).item()

        self._channel_rois = rois
        self._channel_rois = rois