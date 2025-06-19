# TODO:
# - Decide how to handle data loading (ie do we load the data w/ dataloader then validate it separately
#   within each adapter class then call evaluate_framework?)

import json
import logging
from pathlib import Path
from typing import Any

import click
import pandas as pd

from mmm_eval import evaluate_framework
from mmm_eval.data.pipeline import DataPipeline
from mmm_eval.metrics import AVAILABLE_METRICS

logger = logging.getLogger(__name__)


def load_config(config_path: str | None) -> dict[str, Any] | None:
    """Load config from JSON file if provided.

    Args:
        config_path: Path to JSON config file

    Returns:
        Configuration dictionary or None if no path provided

    """
    if not config_path:
        return None
    config_path_obj = validate_path(config_path)
    if not config_path_obj.suffix.lower() == ".json":
        raise ValueError(f"Invalid config path: {config_path}. Must be a JSON file.")
    with open(config_path_obj) as f:
        return json.load(f)


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file.

    Args:
        data_path: Path to CSV data file

    Returns:
        Loaded DataFrame

    """
    data_path_obj = validate_path(data_path)
    if not data_path_obj.suffix.lower() == ".csv":
        raise ValueError(f"Invalid data path: {data_path}. Must be a CSV file.")

    logger.info(f"Loading input data from {data_path}")
    return pd.read_csv(data_path_obj)


def validate_path(path: str) -> Path:
    """Validate path is a valid file path.

    Args:
        path: File path to validate

    Returns:
        Path object

    Raises:
        FileNotFoundError: If path doesn't exist

    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Invalid path:{path}")
    return path_obj


@click.command()
@click.option(
    "--framework",
    type=click.Choice(["pymc-marketing"]),
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
    "--metrics",
    type=click.Choice(AVAILABLE_METRICS),
    multiple=True,
    default=["mape", "rmse"],
    help="Error metrics to compute for out-of-sample prediction. Defaults are mape and rmse.",
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
    config_path: str | None,
    input_data_path: str,
    metrics: tuple[str, ...],
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

    data = DataPipeline(data_path=input_data_path).run()
    
    config = load_config(config_path)

    output_path_obj = Path(output_path).mkdir(parents=True, exist_ok=True) if output_path else None

    # Run evaluation
    logger.info(f"Running evaluation suite for {framework} framework...")

    evaluate_framework(
        framework=framework,
        data=data,
        config=config,
        metrics=list(metrics),
        output_path=output_path_obj,
    )


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
