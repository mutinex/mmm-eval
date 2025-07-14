import logging

import pandas as pd

import mmm_eval.comparison.constants as const
import mmm_eval.comparison.dataloader as dl
from mmm_eval.comparison.utils import DownsampleMethod
from ast import literal_eval

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def to_sql_list(except_columns: list) -> str:
    """Transform a list of columns to an SQL-readable list of columns."""
    return ", ".join(except_columns)


def get_available_table_names(bq_loader: dl.BigQueryLoader, project_id: str, dataset_name: str) -> list[str]:
    """Return a list of available tables in a given datamart for a project.

    Parameters
    ----------
    project_id : str
        The project ID.
    customer_id : str
        The customer ID.

    Returns
    -------
    list
        A list of table names.

    """
    if "_datamart" in dataset_name:
        query = f"""
        SELECT * FROM `{project_id}.{dataset_name}.__TABLES__` 
        WHERE table_id like '%_snapshot'
        AND table_id not like '%_dbt_tmp'"""
    if "_feature_store" in dataset_name:
        query = f"""
        SELECT * FROM `{project_id}.{dataset_name}.__TABLES__` 
        WHERE table_id not like '%data_version_logbook%'
        """
    table_list = bq_loader.load_data(query)
    table_names = table_list.table_id.unique().tolist()
    return table_names


def get_dataset_snapshots(
    bq_loader: dl.BigQueryLoader, project_id: str, dataset_name: str, datamart_version_datetime_utc: str
) -> dict:
    """Return the dataset snapshots for a given datamart version.

    Parameters
    ----------
    project_id : str
        The project ID.
    customer_id : str
        The customer ID.
    datamart_version_datetime_utc : str
        The datetime of the datamart version in UTC.

    Returns
    -------
    dict
        The dataset snapshots.

    """
    query = (
        f"SELECT * FROM {project_id}.{dataset_name}.datamart_version_log "
        f"WHERE datamart_version_datetime_utc = '{datamart_version_datetime_utc}'"
    )
    snapshots = bq_loader.load_data(query)
    return snapshots.datamart_version.to_numpy()[0]


def get_datasets(project_id: str, dataset_name: str, data_version: str | None = None,
                 node_filter: str | None = None) -> dict:
    """Get the datasets for the given datamart version.

    Parameters
    ----------
    data_version : str
        The datamart version.

    Returns
    -------
    dict
        The datasets.

    """
    bq_loader = dl.BigQueryLoader(project_id)

    table_names = get_available_table_names(bq_loader, project_id, dataset_name)

    if node_filter is not None:
        brand, category, product = node_filter.split(".")
        if brand == "default":
            brand_query = "IS NULL"
        else:
            brand_query = f"= '{brand}'"
        if category == "default":
            category_query = "IS NULL"
        else:
            category_query = f"= '{category}'"
        if product == "default":
            product_query = "IS NULL"
        else:
            product_query = f"= '{product}'"

    datasets = {}
    for table in table_names:
        # we don't care about CLV
        if table == "clv_snapshot":
            continue
        if "_datamart" in dataset_name:
            if data_version is None:
                raise ValueError("data_version must be provided for datamart tables")
            snapshot_dict = literal_eval(get_dataset_snapshots(bq_loader, project_id, dataset_name, data_version))
            if table in snapshot_dict:
                snapshot = snapshot_dict[table]
            else:
                logger.warning(f"{table} was not found in datamart")
                continue

            query = f"""
            SELECT * EXCEPT ({to_sql_list(const.EXCEPT_COLUMNS + const.EXCEPT_COLUMN_MAP[table])})
            FROM {project_id}.{dataset_name}.{table}
            WHERE snapshot_datetime_utc = '{snapshot}'
            """
            date_col = "date"
        if "_feature_store" in dataset_name:
            if data_version is None:
                query = f"""SELECT * 
                FROM {project_id}.{dataset_name}.{table}
                WHERE AS_AT = (
                    SELECT MAX(AS_AT) 
                    FROM {project_id}.{dataset_name}.{table})
                """
            else:
                query = f"""SELECT * 
                FROM {project_id}.{dataset_name}.{table}
                WHERE AS_AT = '{data_version}'
                """
            date_col = "period_start"
        logger.info(f"Loading {table}")

        if node_filter:
            query += f"""
            AND brand {brand_query}
            AND category {category_query}
            AND product {product_query}
            """
        df = bq_loader.load_data(query)
        df[date_col] = pd.to_datetime(df[date_col])
        df.loc[:, df.dtypes == "object"] = df.loc[:, df.dtypes == "object"].fillna("default")
        if set(const.BCP_COLS).issubset(set(df.columns)):
            df["node"] = df[["brand", "category", "product"]].agg("_".join, axis=1)
        datasets[table] = df
        logger.info(f"Loaded {table} with {df.shape[0]} rows")

    return datasets


def aggregate_to_node_level(
    df: pd.DataFrame, date_col: str = const.DATE_COL, extra_group_cols: list = [], agg_mapping: dict = {}
) -> pd.DataFrame:
    """Aggregate a dataframe to node level.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to aggregate.
    date_col : str, optional
        The column name for date. Defaults to 'date'.
    extra_group_cols : list, optional
        Additional columns to group by. Defaults to [].
    agg_mapping : dict, optional
        A dictionary of column names and aggregation functions to apply.
        Defaults to {}.

    Returns
    -------
    pd.DataFrame
        The aggregated dataframe.

    """
    group_cols = const.BCP_COLS + [date_col, "node"] + extra_group_cols
    df = df.groupby(group_cols, dropna=False).agg(agg_mapping).reset_index()
    return df


def convert_to_daily(
    df: pd.DataFrame,
    freq: const.DateFrequencyAlias,
    numerical_columns: list[str],
    downsample_method: dict[str, DownsampleMethod] | DownsampleMethod,
    date_col: str,
    freq_col: str,
) -> pd.DataFrame:
    """Convert a DataFrame with a specified frequency to daily frequency.

    Args:
        df (pd.DataFrame): The input DataFrame containing dates, spend, and frequency.
        freq (cons.DateFrequencyAlias): The frequency of the input DataFrame (e.g., "M", "W-MON").
        numerical_columns (list[str]): The list of numerical columns in the DataFrame.
        downsample_method (dict[str, DownsampleMethod] | DownsampleMethod): The distribution method to use.
        date_col (str): The name of the date column.
        freq_col (str): The name of the frequency column.

    Returns:
        pd.DataFrame: A DataFrame converted to daily frequency.

    """
    if df.empty:
        return df

    # The index will not be preserved.
    _df = df.copy().reset_index(drop=True)

    if freq in const.DateFrequencyAlias.weekly_frequencies():
        _df["_start_date"] = _df[date_col]
        _df["_end_date"] = _df[date_col] + pd.to_timedelta(6, unit="D")
    elif freq in const.DateFrequencyAlias.monthly_frequencies():
        _df["_start_date"] = _df[date_col].dt.to_period("M").dt.to_timestamp()
        _df["_end_date"] = _df["_start_date"] + pd.offsets.MonthEnd()
    else:
        raise ValueError("Frequency should be weekly or monthly, got {freq}")

    _df["_repeat_count"] = (_df["_end_date"] - _df["_start_date"]).dt.days + 1
    daily_df: pd.DataFrame = _df.loc[_df.index.repeat(_df["_repeat_count"])].reset_index(drop=True)
    daily_df[date_col] = (
        _df.apply(lambda row: pd.date_range(row["_start_date"], row["_end_date"]), axis=1)
        .explode()
        .reset_index(drop=True)
    )

    downsample_method_per_columns: dict[str, DownsampleMethod] = (
        {numerical_column: downsample_method for numerical_column in numerical_columns}
        if isinstance(downsample_method, DownsampleMethod)
        else downsample_method
    )
    uniform_downsample_cols = [
        col for col, method in downsample_method_per_columns.items() if method == DownsampleMethod.UNIFORM
    ]
    daily_df[uniform_downsample_cols] = daily_df[uniform_downsample_cols].div(daily_df["_repeat_count"], axis=0)
    daily_df = daily_df.drop(columns=["_start_date", "_end_date", "_repeat_count"])
    daily_df[freq_col] = "D"

    return daily_df


def downscale_rows_to_daily(
    df: pd.DataFrame,
    numerical_columns: list[str],
    downsample_method: dict[str, DownsampleMethod] | DownsampleMethod | None,
    date_col: str = const.DATE_COL,
    freq_col: str = const.FREQ_COL,
) -> pd.DataFrame:
    """Downscale a DataFrame to daily frequency row by row.

    Args:
        df (pd.DataFrame): The input DataFrame containing dates, spend, and frequency columns.
        numerical_columns (list[str]): The list of numerical columns in the DataFrame.
            The numerical columns will be spread across the appropriate frequency groups.
        downsample_method (DownsampleMethod, optional): The distribution method to use.
            Defaults to None.
        date_col (str, optional): The name of the date column. Defaults to const.DATE_COL.
        freq_col (str, optional): The name of the frequency column. Defaults to const.FREQ_COL.

    Returns:
        pd.DataFrame: A DataFrame where all data has been converted to daily frequency.

    """
    if df.empty:
        return df

    if len(set(df[freq_col].unique()) - set(const.DateFrequencyAlias.daily_frequencies())) and not downsample_method:
        raise ValueError("Downsample method must be specified for non-daily frequencies")

    # Apply `convert_to_daily` for each frequency group
    daily_dfs = [
        (
            convert_to_daily(
                group_df,
                const.DateFrequencyAlias(freq),
                numerical_columns,
                downsample_method,
                date_col=date_col,
                freq_col=freq_col,
            )
            if freq not in const.DateFrequencyAlias.daily_frequencies() and downsample_method
            else group_df
        )
        for freq, group_df in df.groupby(freq_col)
    ]

    # Concatenate and sort the result
    daily_df = pd.concat(daily_dfs, ignore_index=True).sort_values("date").reset_index(drop=True)
    return daily_df


def convert_df_to_weekly(
    df: pd.DataFrame,
    numerical_columns: list[str],
    agg_method_to_weekly: str | list | dict,
    downsample_method_to_daily: dict[str, DownsampleMethod] | DownsampleMethod | None = None,
    date_col: str = const.DATE_COL,
    freq_col: str = const.FREQ_COL,
    weekly_freq: str = const.FREQ_DEFAULT,
) -> pd.DataFrame:
    """Convert an entire DataFrame with various frequencies to weekly frequency.

    The conversion to weekly frequency is done in two steps:
    * Downsample the data to daily frequency row by row
    * Aggregate the data to weekly frequency without interpolation

    Args:
        df (pd.DataFrame): The input DataFrame containing dates, spend, and frequency columns.
        numerical_columns (list[str]): The list of numerical columns in the DataFrame.
            The numerical columns will be spread across the appropriate frequency groups.
        agg_method_to_weekly (str | list | dict): The aggregation method to use (pandas' convention).
        downsample_method_to_daily (dict[str, DownsampleMethod] | DownsampleMethod | None, optional):
            The distribution method to daily frequency.
        date_col (str, optional): The name of the date column. Defaults to const.DATE_COL.
        freq_col (str, optional): The name of the frequency column. Defaults to const.FREQ_COL.
        weekly_freq (str, optional): The weekly frequency to resample to. Defaults to const.FREQ_DEFAULT.

    Returns:
        pd.DataFrame: A DataFrame where all data has been converted to weekly frequency.

    """
    if df.empty:
        return df

    categorical_columns = list(set(df.columns) - set(numerical_columns + [date_col, freq_col]))
    daily_df = downscale_rows_to_daily(df, numerical_columns, downsample_method_to_daily, date_col, freq_col)
    # Remove 6 days is equivalent to considering the next week with sample method.
    daily_df[date_col] = (
        (daily_df[date_col] - pd.Timedelta(days=6))
        .dt.to_period(weekly_freq)
        .apply(lambda x: pd.to_datetime(x.end_time.normalize()))
    )

    weekly_df = (
        daily_df.groupby([date_col, freq_col] + categorical_columns, dropna=False)
        .agg(agg_method_to_weekly)
        .reset_index()
    )
    if not categorical_columns:
        weekly_df = weekly_df.set_index(date_col).resample(weekly_freq).first().reset_index()
    else:
        weekly_df = (
            weekly_df.groupby(categorical_columns, dropna=False)
            .apply(
                lambda x, date_col=date_col, weekly_freq=weekly_freq: x.set_index(date_col)
                .resample(weekly_freq)[numerical_columns]
                .first()
                .reset_index()
            )
            .reset_index(level=categorical_columns)
            .reset_index(drop=True)
        )
    weekly_df[freq_col] = weekly_freq
    return weekly_df[df.columns]