import logging
from datetime import datetime
from pathlib import Path

import click
import pandas as pd

from mmm_eval.adapters import ADAPTER_REGISTRY
from mmm_eval.configs import get_config
from mmm_eval.core.validation_tests_models import ValidationTestNames
from mmm_eval.data.loaders import DataLoader
from mmm_eval.core import run_evaluation

logger = logging.getLogger(__name__)


def save_results(results: pd.DataFrame, framework: str, output_path: str) -> None:
    """Save the results to a CSV file.
    
    Args:
        results: The dataframe of results to save.
        framework: The name of the framework that was evaluated.
        output_path: The path to save the results to.
    """
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mmm_eval_{framework}_{timestamp}.csv"

    results.to_csv(output_path_obj / filename)
    logger.info(f"Saved results to {output_path_obj / filename}")


@click.command()
@click.option(
    "--framework",
    type=click.Choice(list(ADAPTER_REGISTRY.keys())),
    required=True,
    help="Open source MMM framework to evaluate",
)
@click.option(
    "--input-data-path",
    type=str,
    required=True,
    help="Path to input data file. Supported formats: CSV, Parquet",
)
@click.option(
    "--output-path",
    type=str,
    required=True,
    help="Directory to save evaluation results as a CSV file with name 'mmm_eval_<framework>_<timestamp>.csv'",
)
@click.option(
    "--config-path",
    type=str,
    help="Path to framework-specific JSON config file",
)
@click.option(
    "--test-names",
    type=click.Choice(ValidationTestNames.all_tests_as_str()),
    multiple=True,
    default=tuple(ValidationTestNames.all_tests_as_str()),
    help=(
        "Test names to run. Can specify multiple tests as space-separated values "
        "(e.g. --test-names accuracy cross_validation) or by repeating the flag "
        "(e.g. --test-names accuracy --test-names cross_validation). "
        "Defaults to all tests if not specified."
    ),
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    config_path: str,
    input_data_path: str,
    test_names: tuple[str, ...],
    framework: str,
    output_path: str,
    verbose: bool,
):
    """Evaluate MMM frameworks using the unified API."""
    # logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level)

    logger.info("Loading config...")
    config = get_config(framework, config_path)

    logger.info("Loading input data...")
    data = DataLoader(input_data_path).load()

    # Run evaluation
    logger.info(f"Running evaluation suite for {framework} framework...")
    results = run_evaluation(framework, data, config, test_names)

    # Save results
    if results.empty:
        logger.warning("Results df empty, nothing to save.")
    else:
        save_results(results, framework, output_path)

if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
