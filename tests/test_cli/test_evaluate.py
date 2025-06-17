import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from mmm_eval.cli.evaluate import run_evaluate_framework
from mmm_eval.core.results import EvaluationResults

PYMC_CONFIG_PATH = (
    Path(__file__).parent.parent / "test_data" / "test_pymc" / "test_config.json"
)


def test_evaluate_cli(tmp_path):
    """Test the evaluate CLI command."""
    # Create sample data
    n_weeks = 52
    dates = pd.date_range(start="2023-01-01", periods=n_weeks, freq="W")

    # Generate media spend data
    tv_spend = np.random.uniform(1000, 5000, n_weeks)
    digital_spend = np.random.uniform(500, 3000, n_weeks)
    radio_spend = np.random.uniform(200, 1000, n_weeks)

    # Generate KPI data
    base_sales = 1000
    trend = np.linspace(0, 200, n_weeks)
    seasonality = 100 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)

    kpi = (
        base_sales
        + trend
        + seasonality
        + 0.3 * tv_spend
        + 0.2 * digital_spend
        + 0.1 * radio_spend
        + np.random.normal(0, 50, n_weeks)
    )

    # Create DataFrame
    data = pd.DataFrame(
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

    # Save data to temporary CSV
    data_path = tmp_path / "test_data.csv"
    data.to_csv(data_path, index=False)

    # Copy test config to temporary directory

    # Test evaluate command with Meridian framework
    result = run_evaluate_framework(
        [
            "--input-data-path",
            str(data_path),
            "--framework",
            "pymc3",
            "--target-column",
            "kpi",
            "--output-path",
            str(tmp_path / "output_path"),
            "--config-path",
            str(PYMC_CONFIG_PATH),
        ]
    )

    # Verify result type and content
    assert isinstance(result, EvaluationResults)
    assert "mape" in result.metrics
    assert "rmse" in result.metrics
    assert "r_squared" in result.metrics

    # Verify metric values are reasonable
    assert 0 <= result.get_metric("r_squared") <= 1
    assert result.get_metric("mape") >= 0
    assert result.get_metric("rmse") >= 0


def test_evaluate_cli_invalid_framework(tmp_path):
    """Test evaluate CLI with invalid framework."""
    # Create minimal test data
    data = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=10, freq="W"),
            "kpi": np.random.normal(1000, 100, 10),
            "tv_spend": np.random.uniform(1000, 5000, 10),
        }
    )

    data_path = tmp_path / "test_data.csv"
    data.to_csv(data_path, index=False)

    # Test with invalid framework
    with pytest.raises(SystemExit):  # argparse raises SystemExit for invalid choices
        run_evaluate_framework(
            [
                "--input-data",
                str(data_path),
                "--framework",
                "invalid_framework",
                "--target-column",
                "kpi",
                "--output-path",
                str(tmp_path / "output.json"),
                "--config-path",
                str(PYMC_CONFIG_PATH),
            ]
        )
