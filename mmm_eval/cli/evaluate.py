import click
import logging
from pathlib import Path
import json
import pandas as pd
from typing import Optional, Dict, Any

from mmm_eval import evaluate_framework
from mmm_eval.metrics import AVAILABLE_METRICS

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load config from JSON file if provided."""
    if not config_path:
        return None
    config_path = validate_path(config_path)
    if not config_path.suffix.lower() == ".json":
        raise ValueError(f"Invalid config path: {config_path}. Must be a JSON file.")
    with open(config_path) as f:
        return json.load(f)


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    data_path = validate_path(data_path)
    if not data_path.suffix.lower() == ".csv":
        raise ValueError(f"Invalid data path: {data_path}. Must be a CSV file.")

    logger.info(f"Loading input data from {data_path}")
    return pd.read_csv(data_path)


def validate_path(path: str) -> Path:
    """Validate path is a valid file path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Invalid path:{path}")
    return path


@click.command()
@click.option(
    "--framework",
    type=click.Choice(["meridian", "pymc-marketing"]),
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
    config_path: Optional[str],
    input_data_path: str,
    metrics: Optional[tuple[str, ...]],
    framework: str,
    output_path: Optional[str],
    verbose: Optional[bool],
):
    """An open source tool for MMM evaluation."""
    # logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level)

    # Load input data
    logger.info(f"Loading input data from {input_data_path}")
    data = load_data(input_data_path)

    config = load_config(config_path)

    output_path_obj = (
        Path(output_path).mkdir(parents=True, exist_ok=True) if output_path else None
    )

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
    main()
