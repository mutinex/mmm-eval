"""Convert datasets into a dataframe format suitable for MMM evaluation."""

import logging
from functools import reduce

import pandas as pd

import mmm_eval.comparison.utils as utils
from mmm_eval.comparison.load import aggregate_to_node_level, convert_df_to_weekly
from mmm_eval.comparison.load import get_datasets

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROJECT_ID = "mtx-dataos-datalake-prod"


def _merge_left_on_index(left, right):
    return pd.merge(left, right, left_index=True, right_index=True, how="left")

def process_sales(sales_df):
    sales_weekly = convert_df_to_weekly(
        sales_df,
        numerical_columns=["quantity", "value"],
        downsample_method_to_daily={"value": utils.DownsampleMethod.UNIFORM, "quantity": utils.DownsampleMethod.UNIFORM},
        agg_method_to_weekly={"value": "sum", "quantity": "sum"},
    )

    sales = aggregate_to_node_level(
        sales_weekly, extra_group_cols=[], agg_mapping={"quantity": "sum", "value": "sum"}
    )

    sales_proc = sales[["date", "quantity", "value"]].set_index("date").rename(columns={"value": "revenue"})
    return sales_proc

def process_earned_media(earned_df):
    earned_weekly = convert_df_to_weekly(
        earned_df,
        numerical_columns=["metric_value"],
        downsample_method_to_daily={
            "metric_value": utils.DownsampleMethod.UNIFORM,
        },
        agg_method_to_weekly={
            "metric_value": "sum",
        },
    )

    earned = aggregate_to_node_level(
        earned_weekly, agg_mapping={"metric_value": "sum"}
    )
    # alternative is to include competitor prices as separate cols
    return earned.groupby("date")["metric_value"].sum().ffill().to_frame()


def process_paid_media(df):
    # already converted to weekly
    df["channel_impact"] = df["media_channel"] + "_" + df["marketing_spend_impact"]
    return df.pivot_table(index="date", columns="channel_impact", values="spend", aggfunc="sum")


def process_pricing(pricing_df, company_name: str):
    pricing_weekly = convert_df_to_weekly(
        pricing_df,
        numerical_columns=["retail_price"],
        downsample_method_to_daily={
            "retail_price": utils.DownsampleMethod.REPLICA,
        },
        agg_method_to_weekly={
            "retail_price": "mean",
        },
    )

    pricing = aggregate_to_node_level(
        pricing_weekly, extra_group_cols=["company"], agg_mapping={"retail_price": "mean"}
    )
    # alternative is to include competitor prices as separate cols
    pricing = pricing[pricing["company"].isin([company_name, "default", 'liquorland'])]
    return pricing[["date", "retail_price"]].set_index("date").ffill()

def process_events(events_df, sales_date_index):
    events_weekly = convert_df_to_weekly(
        events_df,
        numerical_columns=["duration",],
        downsample_method_to_daily=utils.DownsampleMethod.REPLICA,
        agg_method_to_weekly={
            "duration": "sum",
        },
    )

    events = aggregate_to_node_level(
        events_weekly, extra_group_cols=["event_type", "event_purpose"], agg_mapping={"duration": "sum"}
    )
    binary_matrix = pd.crosstab(events["date"], events["event_purpose"])
    binary_matrix = binary_matrix.astype(bool).astype(int)
    reindexed = binary_matrix.reindex(sales_date_index, fill_value=0)

    reindexed.columns = [f"{col}_event" for col in reindexed.columns]
    return reindexed


def process_offers(offers_df, company_name: str):
    offers_weekly = convert_df_to_weekly(
        offers_df,
        numerical_columns=["offer_value", "offer_measure"],
        downsample_method_to_daily={
            "retail_price": utils.DownsampleMethod.UNIFORM,
        },
        agg_method_to_weekly={
            "offer_measure": "sum",
            "offer_value": "sum",
        },
    )

    offers = aggregate_to_node_level(
        offers_weekly, extra_group_cols=["company_name"], agg_mapping={
                "offer_measure": "sum",
                "offer_value": "sum",
            },
    )
    # alternative is to include competitor prices as separate cols
    offers = offers[offers["company_name"].isin([company_name, "default"])]
    value_col = "offer_measure" if offers["offer_value"].isnull().all() else "offer_value"
        
    return offers[["date", value_col]].set_index("date").ffill()


def process_discounts(discounts_df):
    discounts_weekly = convert_df_to_weekly(
        discounts_df,
        numerical_columns=[
            "rebates_value",
            "discounts_value",
            "discounted_retail_price",
            "discount_percentage",
            "quantity_discounted",
        ],
        downsample_method_to_daily={
            "rebates_value": utils.DownsampleMethod.UNIFORM,
            "discounts_value": utils.DownsampleMethod.UNIFORM,
            "discounted_retail_price": utils.DownsampleMethod.REPLICA,
            "discount_percentage": utils.DownsampleMethod.REPLICA,
            "quantity_discounted": utils.DownsampleMethod.UNIFORM,
        },
        agg_method_to_weekly={
            "rebates_value": "sum",
            "discounts_value": "sum",
            "discounted_retail_price": "mean",
            "discount_percentage": "mean",
            "quantity_discounted": "sum",
        },
    )

    discounts = aggregate_to_node_level(
        discounts_weekly,
        extra_group_cols=[],
        agg_mapping={
            "discounts_value": "sum",
            "discount_percentage": "mean",
        },
    )

    value_col = "discount_percentage" if discounts["discounts_value"].isnull().all() else "discounts_value"    
    return discounts[["date", value_col]].set_index("date").ffill()


def process_datasets(datasets: dict, company_name: str, holidays: pd.DataFrame,
                     externals: pd.DataFrame):
    sales_processed = process_sales(datasets["sales_snapshot"])
    paid_media_processed = process_paid_media(datasets["paid_media_snapshot"])

    column_map = {"paid_media": paid_media_processed.columns.tolist()}
    processed = [sales_processed, paid_media_processed]

    if datasets.get("pricing_snapshot") is not None:
        if len(datasets["pricing_snapshot"]) > 0:
            pricing_processed = process_pricing(datasets["pricing_snapshot"], company_name=company_name)
            processed.append(pricing_processed)
            column_map["pricing"] = pricing_processed.columns.tolist()

    if datasets.get("pricing_energy_snapshot") is not None:
        if len(datasets["pricing_energy_snapshot"]) > 0:
            pricing_df = datasets["pricing_energy_snapshot"]
            pricing_df = pricing_df.rename(columns={'gas_consumed_price_dollars_per_mj': 'retail_price'})
            pricing_processed = process_pricing(pricing_df, company_name=company_name)
            processed.append(pricing_processed)
            column_map["pricing"] = pricing_processed.columns.tolist()

    # TODO: add mappings for other pricing schemas
    if datasets.get("pricing_beverages_snapshot") is not None:
        if len(datasets["pricing_beverages_snapshot"]) > 0:
            pricing_df = datasets["pricing_beverages_snapshot"]
            pricing_df = pricing_df[["date", "date_frequency", "brand", "category", "product", "node", "retail_price", "company"]]
            pricing_processed = process_pricing(pricing_df, company_name=company_name)
            processed.append(pricing_processed)
            column_map["pricing"] = pricing_processed.columns.tolist()

    if datasets.get("earned_media_snapshot") is not None:
        if len(datasets["earned_media_snapshot"]) > 0:
            earned_processed = process_earned_media(datasets["earned_media_snapshot"])
            processed.append(earned_processed)
            column_map["earned_media"] = earned_processed.columns.tolist()

    if datasets.get("events_snapshot") is not None:
        if len(datasets["events_snapshot"]) > 0:
            events_processed = process_events(datasets["events_snapshot"], sales_processed.index)
            processed.append(events_processed)
            column_map["events"] = events_processed.columns.tolist()

    if datasets.get("offers_snapshot") is not None:
        if len(datasets["offers_snapshot"]) > 0:
            offers_processed = process_offers(datasets["offers_snapshot"], company_name=company_name)
            processed.append(offers_processed)
            column_map["offers"] = offers_processed.columns.tolist()

    if datasets.get("discounts_snapshot") is not None:
        if len(datasets["discounts_snapshot"]) > 0:
            discounts_processed = process_discounts(datasets["discounts_snapshot"])
            processed.append(discounts_processed)
            column_map["discounts"] = discounts_processed.columns.tolist()

    holidays_processed = holidays.pivot(index="date", columns="variable", values="value")
    column_map["holidays"] = holidays_processed.columns.tolist()
    processed.append(holidays_processed)

    externals_processed = externals.pivot(index="date", columns="variable", values="value")
    column_map["externals"] = externals_processed.columns.tolist()
    processed.append(externals_processed)
    
    # Sequential merge
    merged = reduce(_merge_left_on_index, processed)

    # fill remaining datasets
    merged[column_map["paid_media"]] = merged[column_map["paid_media"]].fillna(0)
    if "pricing" in column_map:
        merged[column_map["pricing"]] = merged[column_map["pricing"]].ffill().bfill()
    if "offers" in column_map:
        merged[column_map["offers"]] = merged[column_map["offers"]].fillna(0)
    if "discounts" in column_map:
        merged[column_map["discounts"]] = merged[column_map["discounts"]].fillna(0)
    if "earned_media" in column_map:
        merged[column_map["earned_media"]] = merged[column_map["earned_media"]].fillna(0)
    if "holidays" in column_map:
        merged[column_map["holidays"]] = merged[column_map["holidays"]].fillna(0)
    if "externals" in column_map:
        merged[column_map["externals"]] = merged[column_map["externals"]].ffill().bfill()

    return merged


def load_and_process_datasets(customer_id, data_version, pipeline_data_path: str,
                              holidays_whitelist: list[str],
                              externals_whitelist: list[str],
                              node_filter: str | None = None):
    logger.info(f"Loading datasets for customer {customer_id} with data version {data_version} and node filter {node_filter}")
    datasets = get_datasets(PROJECT_ID, customer_id + "_datamart", data_version, node_filter=node_filter)

    # load paid media
    paid_media = pd.read_parquet(pipeline_data_path + "/paid_media.parquet")
    paid_media = paid_media[["period_start", "media_channel", "marketing_spend_impact", "spend"]].rename(columns={"period_start": "date"})
    datasets["paid_media_snapshot"] = paid_media

    # load holidays and filter to whitelist
    holidays = pd.read_parquet(pipeline_data_path + "/holidays.parquet")
    holidays = holidays[(holidays["variable"].isin(holidays_whitelist))].rename(columns={"period_start": "date"})

    # load externals and filter to whitelist
    try:
        externals = pd.read_parquet(pipeline_data_path + "/externals.parquet")
    except FileNotFoundError:
        externals = pd.read_parquet(pipeline_data_path + "/external_features.parquet")
    externals = externals[(externals["variable"].isin(externals_whitelist))].rename(columns={"period_start": "date"})

    company_name = "_".join(customer_id.split("_")[:-1])
    return process_datasets(datasets, company_name, holidays, externals)