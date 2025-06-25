"""Test CLI evaluation functionality."""

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

from mmm_eval.cli.evaluate import main
from mmm_eval.configs import PyMCConfig

DUMMY_MODEL = MMM(
    date_column="date",
    channel_columns=["channel_1", "channel_2"],
    adstock=GeometricAdstock(l_max=4),
    saturation=LogisticSaturation(),
    yearly_seasonality=2,
)
FIT_KWARGS = {"target_accept": 0.9, "chains": 4}
REVENUE_COLUMN = "revenue"


def _create_test_data() -> pd.DataFrame:
    """Create test data with required columns."""
    return pd.DataFrame(
        {
            "revenue": np.random.randint(0, 100, size=21),
            "spend": np.random.randint(0, 100, size=21),
            "date": pd.date_range(start="2021-01-01", periods=21),
            "response": np.ones(21),
            "control_var1": [0.5] * 21,  # Control column
            "channel_1": [100.0] * 21,  # Channel column
            "channel_2": [100.0] * 21,  # Channel column
        }
    )


@pytest.mark.parametrize(
    "cli_args,expected_exit_code,test_name",
    [
        (
            [
                "--input-data-path",
                "{data_path}",
                "--framework",
                "pymc-marketing",
                "--output-path",
                "{output_path}",
                "--config-path",
                "{config_path}",
            ],
            0,
            "successful_execution",
        ),
        (
            [
                "--framework",
                "pymc-marketing",
                "--output-path",
                "{output_path}",
                "--config-path",
                "{config_path}",
            ],
            2,
            "missing_required_argument",
        ),
        (
            [
                "--input-data-path",
                "{data_path}",
                "--framework",
                "NotAFramework",
                "--output-path",
                "{output_path}",
                "--config-path",
                "{config_path}",
            ],
            2,
            "invalid_framework",
        ),
    ],
)
def test_cli_scenarios(tmp_path, cli_args, expected_exit_code, test_name):
    """Test different CLI scenarios using parametrized tests."""
    # Set up paths
    tmp_path.mkdir(parents=True, exist_ok=True)
    data_path = tmp_path / "data.csv"
    output_path = tmp_path / "output"

    # Create test data and save to CSV (only if data_path is needed)
    if "{data_path}" in cli_args:
        test_data = _create_test_data()
        test_data.to_csv(data_path, index=False)

    config = PyMCConfig.from_model_object(DUMMY_MODEL, fit_kwargs=FIT_KWARGS, revenue_column=REVENUE_COLUMN)
    config.save_model_object_to_json(tmp_path, "test_config")
    config_path = tmp_path / "test_config.json"

    # Format CLI arguments with actual file paths
    formatted_args = []
    for arg in cli_args:
        if arg == "{data_path}":
            formatted_args.append(str(data_path))
        elif arg == "{output_path}":
            formatted_args.append(str(output_path))
        elif arg == "{config_path}":
            formatted_args.append(str(config_path))
        else:
            formatted_args.append(arg)

    # Use Click's test runner
    runner = CliRunner()
    result = runner.invoke(main, formatted_args)

    # Print debug info
    print(f"Test: {test_name}")
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    if result.exception:
        print(f"Exception: {result.exception}")

    # Check that the CLI command returned the expected exit code
    assert (
        result.exit_code == expected_exit_code
    ), f"Test '{test_name}' failed: Expected exit code {expected_exit_code} but got {result.exit_code}. Args: {formatted_args}. Output: {result.output}"
