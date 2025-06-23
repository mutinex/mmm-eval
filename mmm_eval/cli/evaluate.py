import click
import logging
from pathlib import Path
from typing import Optional

from mmm_eval import evaluate_framework, load_data
from mmm_eval.metrics import AVAILABLE_METRICS
from mmm_eval.adapters import ADAPTER_REGISTRY
from mmm_eval.configs import get_config

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

    config = get_config(framework, config_path)

    data = DataPipeline(
        data_path=input_data_path,
        date_column=config["date_column"],
        response_column=config["response_column"],
        revenue_column=config["revenue_column"],
    ).run()

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
