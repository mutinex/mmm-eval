"""Construct datasets to be fed into MMM evaluation frameworks.

TODOs:

- Turn this into a proper script using click or argparse
- Find a better way to get selected holidays/externals from the pipeline run,
instead of having to copy them from the config like an animal 
"""

from mmm_eval.comparison.process import load_and_process_datasets
import pandas as pd

def main():
    customer_id = "xero_51"
    data_version = "2025-07-09T06:24:18.236648"
    pipeline_path = "gs://mtx-ths-uat-us-theseus-pipeline-artifacts/artifacts/816012989105/theseus-xero-20250710113538/run-feature-store-2_-4468728719174795264/model_output/export"

    holidays_whitelist = [
        'BAS Deadline (Q1)',
        'BAS Deadline (Q2)',
        'BAS Deadline (Q3)',
        'BAS Deadline (Q4)',
        'Christmas Day',
        'EOFY',
        'Lockdown Start',
        'Lodgement Deadline',
        'National Day of Mourning for Queen Elizabeth II',
        "New Year's Day",
    ]

    externals_whitelist = [
    'cpi_trimmed_mean_feat_smoothed',
    'consumer_sentiment_feat_centered',
    'employed_total_persons_seasonally_adjusted_feat_differenced',
    'target_cash_rate_rolling_mean',
                        ]
    
    datasets_processed = load_and_process_datasets(customer_id, data_version, pipeline_path,
                                                 holidays_whitelist, externals_whitelist)
                                                 #node_filter="default.consideration.offline")
    
    # save the datasets
    datasets_processed.to_parquet(f"../data/xero_29Sep.parquet")

    # save the datasets
if __name__ == "__main__":
    main()
