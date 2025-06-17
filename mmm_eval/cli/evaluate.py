# TODO:
# - Decide how to handle data loading (ie do we load the data w/ dataloader then validate it separately within each adapter class then call evaluate_framework?)

import argparse
import logging
from pathlib import Path
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

from mmm_eval import evaluate_framework

logger = logging.getLogger(__name__)


def cli_parser() -> argparse.ArgumentParser:
    """Get run experiment cli arguments parser.

    Returns: Run experiment parser
    """
    parser = argparse.ArgumentParser(
        description="An open source tool for evaluating MMMs"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to framework-specific config file",
    )
    parser.add_argument(
        "--input-data-path", 
        type=str, 
        required=True, 
        help="Path to input data file"
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="kpi",
        help="Name of target column in input data. Default is 'kpi'.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["mape", "rmse"],
        help="Error metrics to compute for out-of-sample prediction. Default is mape and rmse.",
    )
    parser.add_argument(
        "--framework",
        type=str,
        required=True,
        choices=["meridian", "pymc", "pymc3"],
        help="Open source MMM framework to evaluate",
    )
    parser.add_argument(
        "--kwargs",
        type=str,
        help="Additional framework-specific parameters",
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        help="Directory to save evaluation results. If not provided, results will not be saved."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    return parser


def load_config(config_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load config from JSON file if provided."""
    if not config_path:
        return None
    with open(config_path) as f:
        return json.load(f)


def load_kwargs(kwargs_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load kwargs from JSON string if provided."""
    if not kwargs_str:
        return None
    return json.loads(kwargs_str)


def validate_path(path: str) -> Path:
    """Validate path is a valid file path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Invalid path:{path}")
    return path


def main(argv=None):
    """Run framework evaluation.

    Args:
        argv: Command line arguments
    """
    parser = cli_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    # Load input data
    data_path = validate_path(args.input_data_path)
    logger.info(f"Loading input data from {data_path}")
    data = pd.read_csv(data_path)

    config = None
    if args.config_path:
        config_path = validate_path(args.config_path)
        config = load_config(config_path)

    kwargs = load_kwargs(args.kwargs)

    # Save results if output path provided
    output_path = None
    if args.output_path:
        output_path = validate_path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    logger.info(f"Running evaluation suite for {args.framework} framework...")
    results = evaluate_framework(
        framework=args.framework,
        data=data,
        config=config,
        target_column=args.target_column,
        metrics=args.metrics,
        output_path=output_path,
        **kwargs if kwargs else {},
    )

    return results


if __name__ == "__main__":
    main()
