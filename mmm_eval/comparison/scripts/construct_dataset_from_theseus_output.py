# import argparse
import os

import pandas as pd


def main(theseus_path: str, output_path: str):
    """Construct datasets for each model in the Theseus output folder."""
    # parser = argparse.ArgumentParser(
    #    description="Construct dataset from Theseus output",
    #    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    # )

    # parser.add_argument("--theseus-path", type=str, help="Path to the Theseus output")

    # parser.add_argument("--output-path", type=str, help="Path to the output dataset")
    # args = parser.parse_args()

    models = os.listdir(theseus_path)
    for model in models:
        dataset_path = os.path.join(theseus_path, model, "dataset_space")
        datasets_available = os.listdir(dataset_path)
        dates = pd.read_parquet(os.path.join(dataset_path, "date_range.parquet"))

        all_data = []
        for dataset in datasets_available:
            data = pd.read_parquet(os.path.join(dataset_path, dataset))
            if dataset in [
                #    "events.parquet",
                #    "holidays.parquet",
                "externals.parquet",
                #    "owend_media.parquet",
                #    "earned_media.parquet",
                #    "trade.parquet",
                "pricing.parquet",
            ]:
                data = data.loc[dates["period_start"]]
                all_data.append(data)
            if dataset == "media_spend.parquet":
                data = data.loc[dates["period_start"]]
                channels = data.columns.tolist()
                data.columns = [
                    col if col.endswith("_brand") or col.endswith("_category") else col + "_product" for col in channels
                ]
                has_zero_spend = data.sum(axis=0) == 0
                if has_zero_spend.any():
                    data = data.loc[:, ~has_zero_spend]
                drop_columns = ["magazines_brand", "cinema_brand"]
                for column in drop_columns:
                    if column in data:
                        data.drop(columns=column, inplace=True)
                all_data.append(data)
            if dataset == "sales.parquet":
                data = data.loc[dates["period_start"]]
                data.columns = ["revenue"]
                all_data.append(data)
            if dataset == "volume.parquet":
                data = data.loc[dates["period_start"]]
                data.columns = ["quantity"]
                all_data.append(data)

        all_data = pd.concat(all_data, axis=1)
        all_data.index.name = "date"

        os.makedirs(output_path, exist_ok=True)
        all_data.to_parquet(os.path.join(output_path, f"{model}_data.parquet"))


if __name__ == "__main__":
    main()
