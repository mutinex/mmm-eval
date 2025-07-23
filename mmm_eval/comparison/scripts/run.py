#!/usr/bin/env python3
"""Script to run MMM evaluation for different frameworks.

This script loads data from a parquet file and runs evaluation for either PyMC or Meridian frameworks.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import tensorflow_probability as tfp
from meridian import constants
from meridian.model import prior_distribution

# PyMC imports
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation

from mmm_eval import PyMCConfig, run_evaluation

# Meridian imports
from mmm_eval.adapters.schemas import (
    MeridianInputDataBuilderSchema,
    MeridianModelSpecSchema,
    MeridianSamplePosteriorSchema,
    PyMCFitSchema,
    PyMCModelSchema,
)
from mmm_eval.comparison.dataset_processor import DatasetProcessor
from mmm_eval.configs.configs import MeridianConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(dataset_path: str) -> pd.DataFrame:
    """Load data from parquet file.

    Args:
        dataset_path: Path to the parquet file

    Returns:
        Loaded DataFrame

    """
    try:
        df = pd.read_parquet(dataset_path)
        logger.info(f"Loaded dataset with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {dataset_path}: {e}")
        raise


def get_column_map(df: pd.DataFrame, framework: str) -> dict[str, list[str]]:
    """Get column mapping for the specified framework.

    Args:
        df: Input DataFrame
        framework: Framework name ('pymc' or 'meridian')

    Returns:
        Dictionary with column mappings
    """
    all_columns = set(df.columns)

    if framework == "pymc":
        # Response column is "quantity"
        response_columns = ["quantity"] if "quantity" in all_columns else []

        # Revenue column is "revenue"
        revenue_columns = ["revenue"] if "revenue" in all_columns else []

        # Channel columns have suffix "_brand", "_category", or "_product"
        channel_columns = [
            col for col in all_columns
            if any(col.endswith(suffix) for suffix in ["_brand", "_category", "_product"])
        ]

        # Control columns are all other columns
        excluded_columns = set(response_columns + revenue_columns + channel_columns)
        control_columns = [col for col in all_columns if col not in excluded_columns]

        return {
            "response": response_columns,
            "revenue": revenue_columns,
            "channel_columns": channel_columns,
            "control_columns": control_columns,
        }

    elif framework == "meridian":
        # Media channels are columns suffixed with "_brand", "_category", or "_product"
        media_channels = [
            col for col in all_columns
            if any(col.endswith(suffix) for suffix in ["_brand", "_category", "_product"])
        ]

        # All binary columns are control_columns
        binary_columns = []
        for col in all_columns:
            if col in df.columns:
                unique_values = df[col].dropna().unique().tolist()
                # Handle both integer/boolean and float binary columns
                if len(unique_values) <= 2 and (
                    set(unique_values).issubset({0, 1, True, False}) or
                    set(unique_values).issubset({0.0, 1.0}) or
                    set(unique_values).issubset({0, 1.0}) or
                    set(unique_values).issubset({0.0, 1})
                ):
                    binary_columns.append(col)

        # Non-binary columns with "price", "offer", or "discount" in the name are non_media_treatment_columns
        non_media_treatment_columns = [
            col for col in all_columns
            if col not in binary_columns and any(keyword in col.lower() for keyword in ["price", "offer", "discount"])
        ]

        # Exclude columns that are already categorized
        excluded_columns = set(media_channels + binary_columns + non_media_treatment_columns + ["date", "quantity", "revenue"])

        # Remaining columns are control_columns
        control_columns = binary_columns + [col for col in all_columns if col not in excluded_columns]

        return {
            "media_channels": media_channels,
            "control_columns": control_columns,
            "non_media_treatment_columns": non_media_treatment_columns,
        }

    else:
        raise ValueError(f"Unsupported framework: {framework}")


def run_pymc_evaluation(processor: DatasetProcessor) -> pd.DataFrame:
    """Run PyMC evaluation.

    Args:
        processor: DatasetProcessor instance to process the dataset.

    Returns:
        Evaluation results DataFrame
    
    """
    logger.info("Setting up PyMC evaluation...")

    # Get column mapping
    col_map = processor.get_pymc_column_map()
    logger.info(f"Column mapping: {col_map}")

    # Set up PyMC configurations
    fast_fit_config = PyMCFitSchema(draws=500, tune=500, chains=4, target_accept=0.95, random_seed=42)

    model_config = PyMCModelSchema(
        date_column="date",
        channel_columns=col_map["channel_columns"],
        control_columns=col_map["control_columns"],
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        time_varying_intercept=False,
        time_varying_media=False,
        yearly_seasonality=2,
    )

    config = PyMCConfig(
        pymc_model_config=model_config, fit_config=fast_fit_config, response_column="quantity", revenue_column="revenue"
    )

    dataset = processor.get_pymc_dataset()

    # Run evaluation
    logger.info("Running PyMC evaluation...")
    results = run_evaluation(framework="pymc_marketing", data=dataset, config=config)

    logger.info(f"PyMC evaluation completed. Results shape: {results.shape}")
    return results


def run_meridian_evaluation(processor: DatasetProcessor) -> pd.DataFrame:
    """Run Meridian evaluation.

    Args:
        processor: DatasetProcessor instance to process the dataset.

    Returns:
        Evaluation results DataFrame
    
    """
    logger.info("Setting up Meridian evaluation...")

    # Get column mapping
    col_map = processor.get_meridian_column_map()
    logger.info(f"Column mapping: {col_map}")

    # Set up Meridian prior
    roi_mu = 1  # Mu for ROI prior for each media channel.
    roi_sigma = 2  # Sigma for ROI prior for each media channel.
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma, name=constants.ROI_M)
    )

    # allow for roughly one knot every two months
    dataset = processor.get_meridian_dataset()
    n_knots = len(dataset["date"].unique()) // 8

    model_spec_config = MeridianModelSpecSchema(prior=prior, knots=n_knots)
    sample_posterior_config = MeridianSamplePosteriorSchema(n_chains=4, n_adapt=300, n_burnin=300, n_keep=600)

    idb_config = MeridianInputDataBuilderSchema(
        date_column="date",
        media_channels=col_map["media_channels"],
        channel_spend_columns=col_map["media_channels"],
        non_media_treatment_columns=col_map.get("non_media_treatment_columns"),
        control_columns=col_map.get("control_columns"),
        response_column="quantity",
    )

    meridian_config = MeridianConfig(
        input_data_builder_config=idb_config,
        model_spec_config=model_spec_config,
        sample_posterior_config=sample_posterior_config,
        response_column="quantity",
        revenue_column="revenue",
    )

    #dataset = dataset.drop(columns=["tv_category"])

    # Run evaluation
    logger.info("Running Meridian evaluation...")
    start_time = time.time()
    results = run_evaluation(
        framework="meridian",
        data=dataset,
        config=meridian_config,
        test_names=("holdout_accuracy", "in_sample_accuracy", "cross_validation"),
    )
    mins_elapsed = (time.time() - start_time) / 60
    logger.info(f"Meridian evaluation completed in {round(mins_elapsed, 1)} minutes")
    return results


def generate_output_filename(base_output: str, framework: str) -> str:
    """Generate output filename with framework name and timestamp.

    Args:
        base_output: Base output filename
        framework: Framework name

    Returns:
        Generated filename with framework and timestamp

    """
    # Get current timestamp at minute level
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Split the base output into name and extension
    output_path = Path(base_output)
    name = output_path.stem
    extension = output_path.suffix

    # Create new filename with framework and timestamp
    new_filename = f"{name}_{framework}_{timestamp}{extension}"

    return new_filename


def main():
    """Run the script as the main entry point."""
    parser = argparse.ArgumentParser(
        description="Run MMM evaluation for different frameworks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("dataset_path", type=str, help="Path to the parquet dataset file")

    parser.add_argument("framework", type=str, choices=["pymc", "meridian"], help="Framework to use for evaluation")

    parser.add_argument(
        "--output", type=str, default="results.csv", help="Output file path for results (default: results.csv)"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input file exists
    if not Path(args.dataset_path).exists():
        logger.error(f"Dataset file not found: {args.dataset_path}")
        sys.exit(1)

    # Load data
    logger.info(f"Loading data from {args.dataset_path}")
    df = load_data(args.dataset_path)

    processor = DatasetProcessor.from_raw_data(df)

    # Run evaluation based on framework
    if args.framework == "pymc":
        results = run_pymc_evaluation(processor)
    elif args.framework == "meridian":
        results = run_meridian_evaluation(processor)
    else:
        raise ValueError(f"Unsupported framework: {args.framework}")

    # Generate output filename with framework name and timestamp
    output_filename = generate_output_filename(args.output, args.framework)

    # Save results
    logger.info(f"Saving results to {output_filename}")
    results.to_csv(output_filename, index=False)
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
