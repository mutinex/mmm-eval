import argparse
import os

from mmm_eval.comparison.scripts.construct_dataset_from_theseus_output import (
    main as construct_dataset_from_theseus_output,
)
from mmm_eval.comparison.scripts.run import main as run_evaluation


def main():
    """Run all nodes for one customer from Theseus output."""
    # parser = argparse.ArgumentParser(
    #    description="Run for one customer from Theseus outputt",
    #    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    # )
    # parser.add_argument("--customer", type=str, help="Customer")

    # parser.add_argument("--theseus-path", type=str, help="Path to the Theseus output")

    # parser.add_argument("--output-path", type=str, help="Path to the output dataset")
    # args = parser.parse_args()

    output_path = "../../../../mmm_eval_results"
    base_model_path = "../../../../campaign/cmp_media_pert/"

    customers = [
        "bendigo_bank",
        "amaysim",
        "honda",
        "one_nz",
        "country_road_group",
        "domino_s_nl",
        "medibank",
        "tourism_nz",
        "alinta",
        "freedom",
    ]
    for customer in customers:
        print(customer)

        customer_output = os.path.join(output_path, customer)
        data_path = os.path.join(customer_output, "data")
        theseus_path = os.path.join(base_model_path, customer, "base")

        # construct_dataset_from_theseus_output(theseus_path=theseus_path, output_path=data_path)

        models = os.listdir(data_path)
        framework = "meridian"
        for model in models:
            if model == "default_consumer_prepay_travel_data.parquet":
                continue
            print(model)
            framework_output = os.path.join(customer_output, framework + "_crps")
            if os.path.exists(os.path.join(framework_output, model.replace("_data.parquet", ""))):
                continue

            run_evaluation(
                dataset_path=os.path.join(data_path, model),
                framework=framework,
                output=os.path.join(framework_output, model.replace("_data.parquet", "")),
            )


if __name__ == "__main__":
    main()
