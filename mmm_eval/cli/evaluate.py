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


def evaluate_framework_parser() -> argparse.ArgumentParser:
    """Get run experiment cli arguments parser.

    Returns: Run experiment parser
    """
    parser = argparse.ArgumentParser(
        description="Evaluate MMM frameworks using mmm-eval"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to framework-specific config JSON file",
    )
    parser.add_argument(
        "--input-data-path", type=str, required=True, help="Path to input data"
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="kpi",
        help="Name of target/KPI column in input data",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["mape", "rmse"],
        help="Metrics to compute (e.g. mape rmse r_squared)",
    )
    parser.add_argument(
        "--framework",
        type=str,
        required=True,
        choices=["meridian", "pymc"],
        help="Framework to evaluate",
    )
    parser.add_argument(
        "--kwargs",
        type=str,
        help="JSON string of additional framework-specific parameters",
    )
    parser.add_argument(
        "--output-path", type=str, help="Path to save evaluation results JSON"
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
    parser = evaluate_framework_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    # check paths exist
    output_path = validate_path(args.output_path)

    # Load input data
    data_path = validate_path(args.input_data_path)
    logger.info(f"Loading input data from {data_path}")
    data = pd.read_csv(data_path)

    # Load config and kwargs
    config_path = validate_path(args.config_path)
    config = load_config(config_path)

    kwargs = load_kwargs(args.kwargs)

    # Run evaluation
    logger.info(f"Running evaluation suite for {args.framework} framework...")
    results = evaluate_framework(
        framework=args.framework,
        data=data,
        config=config,
        target_column=args.target_column,
        metrics=args.metrics,
        **kwargs if kwargs else {},
    )

    # Save results if output path provided
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info(f"Results saved to {args.output_path}")

    return results


if __name__ == "__main__":
    main()
