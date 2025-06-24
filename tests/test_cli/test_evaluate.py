"""Test CLI evaluation functionality."""

import subprocess

import numpy as np
import pandas as pd
import pytest
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

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


@pytest.mark.parametrize(
    "cmd_template,expected_return_code",
    [
        (
            "mmm-eval --input-data-path {data_path} --framework pymc-marketing "
            "--output-path {output_path} --config-path {config_path}",
            0,  # Should work fine
        ),
        (
            "mmm-eval --output-path {output_path} --config-path {config_path}",
            2,
        ),
        (
            "mmm-eval --input-data-path {data_path} --framework NotAFramework "
            "--output-path {output_path} --config-path {config_path}",
            2,  # Usage error
        ),
    ],
)
def test_cli_as_subprocess(tmp_path, cmd_template, expected_return_code):
    """Test the evaluate CLI command as a subprocess."""
    # Set up paths
    tmp_path.mkdir(parents=True, exist_ok=True)
    data_path = tmp_path / "data.csv"
    output_path = tmp_path / "output"

    # Create dummy input files
    pd.DataFrame(
        {
            "revenue": np.random.randint(0, 100, size=21),
            "spend": np.random.randint(0, 100, size=21),
            "date": pd.date_range(start="2021-01-01", periods=21),
            "response": np.random.randint(0, 100, size=21),
        }
    ).to_csv(data_path, index=False)
    config = PyMCConfig(DUMMY_MODEL, fit_kwargs=FIT_KWARGS, revenue_column=REVENUE_COLUMN)
    config.save_config(tmp_path, "test_config")
    config_path = tmp_path / "test_config.json"

    # Format command string with actual file paths
    cmd = cmd_template.format(
        data_path=str(data_path),
        config_path=str(config_path),
        output_path=str(output_path),
    )

    result = subprocess.run(cmd, capture_output=False, text=True, shell=True)

    assert (
        result.returncode == expected_return_code
    ), f"Expected return code {expected_return_code} but got {result.returncode} for command: {cmd}"
