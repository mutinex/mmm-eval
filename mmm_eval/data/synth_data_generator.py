"""Generate synthetic data for testing.

Based on:
https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_example.html
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation


def generate_pymc_data():
    """Generate synthetic MMM data for testing purposes.

    Returns
        DataFrame containing synthetic MMM data with media channels, controls, and response variables

    """
    seed: int = sum(map(ord, "mmm"))
    rng: np.random.Generator = np.random.default_rng(seed=seed)

    # date range
    min_date = pd.to_datetime("2018-04-01")
    max_date = pd.to_datetime("2021-09-01")

    df = pd.DataFrame(data={"date_week": pd.date_range(start=min_date, end=max_date, freq="W-MON")}).assign(
        year=lambda x: x["date_week"].dt.year,
        month=lambda x: x["date_week"].dt.month,
        dayofyear=lambda x: x["date_week"].dt.dayofyear,
    )

    n = df.shape[0]

    # media spend data
    channel_1 = 100 * rng.uniform(low=0.0, high=1, size=n)
    df["channel_1"] = np.where(channel_1 > 90, channel_1, channel_1 / 2)

    channel_2 = 100 * rng.uniform(low=0.0, high=1, size=n)
    df["channel_2"] = np.where(channel_2 > 80, channel_2, 0)

    # apply geometric adstock transformation
    alpha1: float = 0.4
    alpha2: float = 0.2

    df["channel_1_adstock"] = (
        geometric_adstock(x=df["channel_1"].to_numpy(), alpha=alpha1, l_max=8, normalize=True).eval().flatten()
    )

    df["channel_2_adstock"] = (
        geometric_adstock(x=df["channel_2"].to_numpy(), alpha=alpha2, l_max=8, normalize=True).eval().flatten()
    )

    # apply saturation transformation
    lam1: float = 4.0
    lam2: float = 3.0

    df["channel_1_adstock_saturated"] = logistic_saturation(x=df["channel_1_adstock"].to_numpy(), lam=lam1).eval()

    df["channel_2_adstock_saturated"] = logistic_saturation(x=df["channel_2_adstock"].to_numpy(), lam=lam2).eval()

    # trend + seasonal
    df["trend"] = (np.linspace(start=0.0, stop=50, num=n) + 10) ** (1 / 4) - 1

    df["cs"] = -np.sin(2 * 2 * np.pi * df["dayofyear"] / 365.5)
    df["cc"] = np.cos(1 * 2 * np.pi * df["dayofyear"] / 365.5)
    df["seasonality"] = 0.5 * (df["cs"] + df["cc"])

    # controls
    df["event_1"] = (df["date_week"] == "2019-05-13").astype(float)
    df["event_2"] = (df["date_week"] == "2020-09-14").astype(float)

    # generate quantity
    df["intercept"] = 1000.0  # Base quantity
    # noise
    df["epsilon"] = rng.normal(loc=0.0, scale=50.0, size=n)

    # amplitude = 1
    beta_1 = 400
    beta_2 = 150

    # Generate price with seasonal fluctuations
    base_price = 5
    price_seasonality = 0.03 * (df["cs"] + df["cc"])
    price_trend = np.linspace(0, 2, n)  # Gradual price increase
    df["price"] = base_price + price_seasonality + price_trend

    df["quantity"] = (
        df["intercept"]
        + df["trend"] * 100
        + df["seasonality"] * 200
        + df["price"] * -50
        + 150 * df["event_1"]
        + 250 * df["event_2"]
        + beta_1 * df["channel_1_adstock_saturated"]
        + beta_2 * df["channel_2_adstock_saturated"]
        + df["epsilon"]
    )
    # Calculate revenue
    df["revenue"] = df["price"] * df["quantity"]

    columns_to_keep = [
        "date_week",
        "quantity",
        "price",
        "revenue",
        "channel_1",
        "channel_2",
        "event_1",
        "event_2",
        "dayofyear",
    ]

    df = df[columns_to_keep]
    return df


def generate_meridian_data():
    """Load and process a Meridian-compatible dataset for E2E testing.

    Returns
        DataFrame containing Meridian-compatible data with media channels, controls, and
        response variables

    """
    # Path to the local parquet file
    parquet_path = Path(__file__).parent / "sample_data" / "meridian_geo_media.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Meridian sample data file not found at {parquet_path}. "
            "Please ensure the meridian_geo_media.parquet file is present in the sample_data directory."
        )

    df = pd.read_parquet(parquet_path)
    df_mod = df.copy()
    df_mod["revenue"] = df_mod["revenue_per_conversion"] * df_mod["conversions"]

    df_mod = df.iloc[:, 1:]
    # restrict to only two geos
    df_mod = df_mod[df_mod["geo"].isin(["Geo0", "Geo1"])]
    df_mod["revenue"] = df_mod["revenue_per_conversion"] * df_mod["conversions"]
    df_mod = df_mod.drop(columns="revenue_per_conversion")

    # restrict to only post-2023
    df_mod = df_mod[pd.to_datetime(df_mod["time"]) > pd.Timestamp("2023-01-01")]
    df_mod = df_mod.rename(columns={"time": "date"})
    return df_mod
