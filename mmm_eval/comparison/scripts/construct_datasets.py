from mmm_eval.comparison.process import load_and_process_datasets
import pandas as pd

def main():

    # customer_id = "chicken_treat_53"
    # data_version = "2025-06-25T00:18:48.379609"
    # pipeline_path = "gs://mtx-ths-uat-us-theseus-pipeline-artifacts/artifacts/816012989105/theseus-chicken-treat-20250627150301/run-feature-store-2_5949443570921373696/model_output/export"

    # holidays_whitelist = [
    # "New Year's Day",
    # 'Christmas Day',
    # 'Australia Day',
    # 'Good Friday',
    # ]
    # # why are some of these not showing up?
    # externals_whitelist = [
    #     'foot_traffic_vol',
    #     'foot_traffic_vol_feat_weekly_pct_chng_smoothed',
    #     'consumer_sentiment_feat_centered',
    #     'cpi_trimmed_mean_feat_smoothed',
    #     'any_lockdown_2020',
    #     'any_lockdown_2021',
    #     'category_sales',
    #     'category_sales_diff_decayed',
    #                     ]

    customer_id = "xero_51"
    data_version = "2025-07-09T06:24:18.236648"
    #pipeline_path = "gs://mtx-ths-dev-us-theseus-pipeline-artifacts/artifacts/757046730850/theseus-xero-20250715122215/run-feature-store-2_-6097048265306406912/model_output/export"
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
    # why are some of these not showing up?
    externals_whitelist = [
    'cpi_trimmed_mean_feat_smoothed',
    'consumer_sentiment_feat_centered',
    'employed_total_persons_seasonally_adjusted_feat_differenced',
    'target_cash_rate_rolling_mean',
                        ]

    # customer_id = "origin_energy_61"
    # data_version = "2025-07-14T16:11:41.514193"
    # pipeline_path = "gs://mtx-ths-uat-us-theseus-pipeline-artifacts/artifacts/816012989105/theseus-origin-energy-20250718110538/run-feature-store-2_5559036128956252160/model_output/export"

    # holidays_whitelist = [
    #     'eofy_aus',
    #     "new_year's_day_aus",
    #     'boxing_day_aus',
    #     'christmas_day_aus',
    #     'easter_monday_aus',
    #     'good_friday_aus',
    #     ]
    # externals_whitelist = [
    # 'weather_AUS_deg_c_above_cooling_threshold',
    # 'weather_AUS_deg_c_below_heating_threshold',
    # 'economic_indicators_AUS_cash_rate_diff_lag_52_ewm_2',
    # 'consumer_pricing_index_all_groups_global_AUS_electricity_ewm_4',
    # 'consumer_pricing_index_all_groups_global_AUS_gas_and_other_household_fuels_ewm_4'
    #                     ]
    
    # datasets_processed = load_and_process_datasets(customer_id, data_version, pipeline_path,
    #                                              holidays_whitelist, externals_whitelist,
    #                                              node_filter="default.core.gas")

    # customer_id = "coles_liquor_group_85"
    # data_version = "2025-07-17T03:50:52.065435"
    # pipeline_path = "gs://mtx-ths-uat-us-theseus-pipeline-artifacts/artifacts/816012989105/theseus-coles-liquor-group-20250717030613/run-feature-store-2_8783622258246549504/model_output/export"

    # holidays_whitelist = [
    #     'anzac_day_aus',
    #     'australia_day_aus',
    #     "new_year's_day_aus",
    #     'christmas_day_aus',
    #     'good_friday_aus',
    #     'grand_final_day_aus',
    #     'black_friday_aus',
    #     ]
    # externals_whitelist = [
    # 'inflation_AUS_consumer_price_index_quarterly_change',
    # 'weather_AUS_maximum_temperature_c_logit_ewm_8_avg_mean_centered',
    # 'foot_traffic_hourly_AUS_median_pedestrian_count_ewm_8_16-19',
    # 'foot_traffic_hourly_AUS_median_pedestrian_count_ewm_8_20-23',
    # ]
    
    datasets_processed = load_and_process_datasets(customer_id, data_version, pipeline_path,
                                                 holidays_whitelist, externals_whitelist)
                                                 #node_filter="default.consideration.offline")
    
    # save the datasets
    datasets_processed.to_parquet(f"../data/xero_29Sep.parquet")

    # save the datasets
if __name__ == "__main__":
    main()
