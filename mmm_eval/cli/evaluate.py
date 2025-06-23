import logging
from pathlib import Path

import click

from mmm_eval import evaluate_framework
from mmm_eval.adapters import ADAPTER_REGISTRY
from mmm_eval.configs import get_config
from mmm_eval.data.pipeline import DataPipeline
import pandas as pd
from typing import Optional, Dict, Any, List

from mmm_eval.core.evaluator import Evaluator
from mmm_eval.core.validation_tests_models import ValidationTestNames

logger = logging.getLogger(__name__)


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
    help="Path to input data CSV file",
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
    default=ValidationTestNames.all_tests_as_str(),
    help="Error metrics to compute for out-of-sample prediction. Defaults are mape and rmse. Can be a list of test names.",
)
@click.option(
    "--output-path",
    type=str,
    help="Directory to save evaluation results. If not provided, results will not be saved.",
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
    test_names: List[str],
    framework: str,
    output_path: str | None,
    verbose: bool,
):
    """Evaluate MMM frameworks using the unified API."""
    # logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level)

    # Load input data
    logger.info(f"Loading input data from {input_data_path}")

    config = get_config(framework, config_path)

    data = DataPipeline(
        data_path=input_data_path,
        date_column=config.date_column,
        response_column=config.response_column,
        revenue_column=config.revenue_column,
        control_columns=config.control_columns,
        channel_columns=config.channel_columns,
    ).run()

    output_path_obj = Path(output_path).mkdir(parents=True, exist_ok=True) if output_path else None

    # Run evaluation
    logger.info(f"Running evaluation suite for {framework} framework...")

    # Create instance of evaluator with everything that will be common to evaluate a framework
    evaluator = Evaluator(
        data=data,
        output_path=output_path_obj,
        test_names=test_names,
    )

    # Evaluate the tests for the chosen framework and config. This is left as a method as future adaptions will likely allow for multiple frameworks to be evaluated at once.
    evaluator.evaluate_framework(
        framework=framework,
        config=config,
    )


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
