# TODO:
# - Decide how to handle data loading (ie do we load the data w/ dataloader then validate it separately within each adapter class then call evaluate_framework?)

import click
import logging
from pathlib import Path
import json
import pandas as pd
from typing import Optional, Dict, Any

from mmm_eval import evaluate_framework

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load config from JSON file if provided."""
    if not config_path:
        return None
    with open(config_path) as f:
        return json.load(f)


def validate_path(path: str) -> Path:
    """Validate path is a valid file path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Invalid path:{path}")
    return path


@click.command()
@click.option(
    "--framework",
    type=click.Choice(["meridian", "pymc", "pymc3"]),
    required=True,
    help="Open source MMM framework to evaluate",
)
@click.option(
    "--config-path",
    type=str,
    help="Path to framework-specific JSON config file",
)
@click.option(
    "--input-data-path",
    type=str,
    required=True,
    help="Path to input data file",
)
@click.option(
    "--target-column",
    type=str,
    default="kpi",
    help="Name of target column in input data. Default is 'kpi'.",
)
@click.option(
    "--metrics",
    type=str,
    multiple=True,
    default=["mape", "rmse"],
    help="Error metrics to compute for out-of-sample prediction. Default is mape and rmse.",
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
    config_path: Optional[str],
    input_data_path: str,
    target_column: str,
    metrics: tuple[str, ...],
    framework: str,
    output_path: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    """An open source tool for MMM evaluation."""
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level)

    # Load input data
    data_path = validate_path(input_data_path)
    logger.info(f"Loading input data from {data_path}")
    data = pd.read_csv(data_path)

    config = None
    if config_path:
        config_path = validate_path(config_path)
        config = load_config(config_path)

    # Save results if output path provided
    output_path_obj = None
    if output_path:
        output_path_obj = validate_path(output_path)
        output_path_obj.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    logger.info(f"Running evaluation suite for {framework} framework...")
    results = evaluate_framework(
        framework=framework,
        data=data,
        config=config,
        target_column=target_column,
        metrics=list(metrics),
        output_path=output_path_obj,
    )

    return results


if __name__ == "__main__":
    main()
