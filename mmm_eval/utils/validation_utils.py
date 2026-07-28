import numpy as np
import pandas as pd


def distribution_to_dataframe(distribution: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Generate a DataFrame from a distribution array.

    Args:
        distribution (np.ndarray): The distribution array, shape (samples x time)
        index (pd.Index): The index of the actual values, shape (time,)

    Returns:
        A DataFrame, with sample*time number of rows.

    """
    name = index.name
    distribution_df = (
        pd.DataFrame(distribution, columns=index)
        .reset_index(names="sample_index")
        .melt(id_vars="sample_index", var_name=name, value_name="pred_distribution")
    )
    distribution_df[name] = pd.to_datetime(distribution_df[name])
    return distribution_df
