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


class PyMCAdapter(BaseAdapter):
    def __init__(self, config: dict):
        # Store explicitly needed pieces
        self.response_col = config["response_column"]
        self.revenue_col = config.pop("revenue_column")

        # Pass everything else (after extracting response_col) to MMM constructor
        self.model_kwargs = {
            k: v for k, v in config.items()
            if k != "response_column" and
            k != "fit_kwargs"
        }
        self.fit_kwargs = config.get("fit_kwargs", {})

        self.model = None
        self.trace = None
        self._channel_rois = None

    def fit(self, data: pd.DataFrame, metadata: dict = None):
        X = data.drop(columns=[self.response_col, self.revenue_col])
        print(X.columns)
        y = data[self.response_col]

        self.model = MMM(**self.model_kwargs)
        self.trace = self.model.fit(X=X, y=y, **self.fit_kwargs)

        #self._compute_rois()
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

    def _compute_rois(self):
        """
        Estimate ROI = marginal contribution / spend.
        PyMC-Marketing supports posterior samples of β weights.
        We'll use the mean of the posterior coefficients.
        """
        betas = self.trace.posterior["beta_media"].mean(dim=["chain", "draw"]).values
        self._channel_rois = {
            ch: float(beta) for ch, beta in zip(self.channels, betas)
        }
