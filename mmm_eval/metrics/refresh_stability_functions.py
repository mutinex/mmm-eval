# This file contains the functions for calculating the refresh stability of the MMM framework.

import numpy as np
import pandas as pd

from mmm_eval.data.input_dataframe_constants import InputDataframeConstants


def calculate_absolute_percentage_change_between_series(
    baseline_series: pd.Series, comparison_series: pd.Series
) -> pd.Series:
    """
    Calculate the refresh stability of the MMM framework.
    """
    return np.abs((comparison_series - baseline_series) / baseline_series)


def filter_to_common_dates(
    baseline_data: pd.DataFrame, comparison_data: pd.DataFrame
) -> pd.DataFrame:
    """Filter the data to the common dates for stability comparison."""

    common_start_date = max(
        baseline_data[InputDataframeConstants.DATE_COL].min(),
        comparison_data[InputDataframeConstants.DATE_COL].min(),
    )
    common_end_date = min(
        baseline_data[InputDataframeConstants.DATE_COL].max(),
        comparison_data[InputDataframeConstants.DATE_COL].max(),
    )

    baseline_data_fil = baseline_data[
        baseline_data[InputDataframeConstants.DATE_COL].between(
            common_start_date, common_end_date
        )
    ]
    comparison_data_fil = comparison_data[
        comparison_data[InputDataframeConstants.DATE_COL].between(
            common_start_date, common_end_date
        )
    ]

    return baseline_data_fil, comparison_data_fil

def aggregate_via_media_channel(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate the data to the media spend per channel."""
    return (
        data.groupby(InputDataframeConstants.MEDIA_CHANNEL_COL, dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )


# def calculate_coefficents_have_flipped_signs(current_coefficients: np.ndarray, refreshed_coefficients: np.ndarray) -> bool:
#     """
#     Calculate if the coefficients have flipped signs.
#     """
#     return np.any(np.sign(current_coefficients) != np.sign(refreshed_coefficients))
