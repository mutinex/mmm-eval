#!/usr/bin/env python3
"""
Basic usage example for mmm-eval MMM evaluation framework.

This example demonstrates how to use the mmm-eval package to evaluate
different MMM frameworks (Meridian, PyMC, Robyn, LightweightMMM) using
a unified API.
"""

import pandas as pd
import numpy as np
from mmm_eval import evaluate_framework, get_adapter


def create_sample_mmm_data(n_weeks: int = 52) -> pd.DataFrame:
    """Create sample MMM data for demonstration."""
    np.random.seed(42)

    # Generate date range
    dates = pd.date_range("2023-01-01", periods=n_weeks, freq="W")

    # Generate media channels (spend data)
    tv_spend = np.random.uniform(1000, 5000, n_weeks)
    digital_spend = np.random.uniform(500, 3000, n_weeks)
    radio_spend = np.random.uniform(200, 1500, n_weeks)

    # Generate base factors
    base_sales = 1000
    trend = np.linspace(0, 200, n_weeks)  # Slight upward trend
    seasonality = 100 * np.sin(
        2 * np.pi * np.arange(n_weeks) / 52
    )  # Annual seasonality

    # Generate KPI with media influence
    kpi = (
        base_sales
        + trend
        + seasonality
        + 0.3 * tv_spend
        + 0.2 * digital_spend
        + 0.1 * radio_spend
        + np.random.normal(0, 50, n_weeks)
    )  # Add noise

    return pd.DataFrame(
        {
            "date": dates,
            "kpi": kpi,
            "tv_spend": tv_spend,
            "digital_spend": digital_spend,
            "radio_spend": radio_spend,
            "base_trend": trend,
            "seasonality": seasonality,
        }
    )


def main():
    """Demonstrate MMM evaluation framework usage."""

    print("=== MMM Evaluation Framework Demo ===\n")

    # Create sample data
    print("Creating sample MMM data...")
    data = create_sample_mmm_data(n_weeks=52)
    print(f"Data shape: {data.shape}")
    print(f"KPI range: {data['kpi'].min():.0f} - {data['kpi'].max():.0f}")
    print(f"Media channels: {[col for col in data.columns if 'spend' in col]}\n")

    # Example 1: Evaluate single framework using main API
    print("=== Example 1: Evaluate Meridian Framework ===")

    meridian_config = {
        "media_columns": ["tv_spend", "digital_spend", "radio_spend"],
        "base_columns": ["base_trend", "seasonality"],
    }

    try:
        results = evaluate_framework(
            framework="meridian",
            data=data,
            config=meridian_config,
            response_column="kpi",
            metrics=["mape", "rmse", "r_squared"],
        )

        print("Meridian Results:")
        print(results)
        print(f"Best metric (R²): {results.get_metric('r_squared'):.4f}\n")

    except Exception as e:
        print(f"Error evaluating Meridian: {e}\n")

    # Example 2: Compare multiple frameworks
    print("=== Example 2: Compare Multiple Frameworks ===")

    frameworks = ["meridian", "pymc", "robyn", "lightweight_mmm"]
    framework_configs = {
        "meridian": {"media_columns": ["tv_spend", "digital_spend", "radio_spend"]},
        "pymc": {
            "media_columns": ["tv_spend", "digital_spend", "radio_spend"],
            "n_samples": 500,
        },
        "robyn": {
            "media_columns": ["tv_spend", "digital_spend", "radio_spend"],
            "adstock_params": {
                "tv_spend": 0.4,
                "digital_spend": 0.2,
                "radio_spend": 0.3,
            },
        },
        "lightweight_mmm": {
            "media_columns": ["tv_spend", "digital_spend", "radio_spend"]
        },
    }

    comparison_results = {}

    for framework in frameworks:
        print(f"Evaluating {framework}...")
        try:
            result = evaluate_framework(
                framework=framework,
                data=data,
                config=framework_configs[framework],
                response_column="kpi",
                metrics=["mape", "rmse", "r_squared"],
            )
            comparison_results[framework] = result
            print(f"  MAPE: {result.get_metric('mape'):.2f}%")
            print(f"  RMSE: {result.get_metric('rmse'):.2f}")
            print(f"  R²: {result.get_metric('r_squared'):.4f}\n")

        except Exception as e:
            print(f"  Error: {e}\n")

    # Example 3: Direct adapter usage
    print("=== Example 3: Direct Adapter Usage ===")

    try:
        # Get PyMC adapter directly
        pymc_adapter = get_adapter(
            "pymc", config={"media_columns": ["tv_spend", "digital_spend"]}
        )

        # Fit and evaluate
        pymc_results = pymc_adapter.fit_and_evaluate(
            data=data, response_column="kpi", metrics=["mape", "r_squared"]
        )

        print("PyMC Direct Adapter Results:")
        print(f"Framework: {pymc_results.framework}")
        print(f"Metrics: {pymc_results.metrics}")

        # Access predictions
        if pymc_results.predictions is not None:
            print(f"Predictions sample: {pymc_results.predictions.head(3).tolist()}")

        print(f"Results dict: {pymc_results.to_dict()}\n")

    except Exception as e:
        print(f"Error with direct adapter: {e}\n")

    # Example 4: Custom metrics
    print("=== Example 4: Custom Metrics Selection ===")

    try:
        # Evaluate with different metric combinations
        lightweight_results = evaluate_framework(
            framework="lightweight_mmm",
            data=data,
            config={"media_columns": ["tv_spend", "digital_spend", "radio_spend"]},
            metrics=["mape", "symmetric_mape", "mae"],  # Different metric set
        )

        print("LightweightMMM with custom metrics:")
        for metric, value in lightweight_results.metrics.items():
            print(f"  {metric}: {value:.4f}")

    except Exception as e:
        print(f"Error with custom metrics: {e}\n")

    print("=== Demo Complete ===")
    print("\nNote: All adapters are currently placeholder implementations.")
    print("In practice, you would integrate with the actual MMM framework libraries.")


if __name__ == "__main__":
    main()
