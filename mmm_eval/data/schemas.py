"""
Pydantic schemas for MMM data validation.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

import pandera.pandas as pa
from pandera.typing import Index, Series
from mmm_eval.data.constants import InputDataframeConstants


class ValidatedDataSchema(pa.DataFrameModel):
    """
    Schema for MMM data validation.

    Defines the bare minimum columns for MMM evaluation.
    """
    
    # Required columns
    date: Series[pa.dtypes.DateTime] = pa.Field(nullable=False)
    media_channel: Series[str] = pa.Field(nullable=False)
    spend: Series[pa.dtypes.Float64] = pa.Field(nullable=False)
    
    class Config:
        coerce = True