"""Test CLI evaluation functionality."""

import json
import os
import subprocess

import pandas as pd
import pytest


@pytest.mark.parametrize(
    "cmd_template,expected_return_code",
    [
        (
            "mmm-eval --input-data-path {data_path} --framework pymc-marketing "
            "--output-path {output_path} --config-path {config_path}",
            1,  # Should work fine
        ),
        (
            "mmm-eval --output-path {output_path} --config-path {config_path}",
            2,  # Usage error
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
    data_path = tmp_path / "data.csv"
    config_path = tmp_path / "test_config.json"
    output_path = tmp_path / "output"

    # Create dummy input files
    pd.DataFrame({"kpi": [1, 2, 3]}).to_csv(data_path, index=False)
    with open(config_path, "w") as f:
        json.dump({"dummy": "config"}, f)
    os.makedirs(output_path, exist_ok=True)

    # Format command string with actual file paths
    cmd = cmd_template.format(
        data_path=str(data_path),
        config_path=str(config_path),
        output_path=str(output_path),
    )

    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)

    assert (
        result.returncode == expected_return_code
    ), f"Expected return code {expected_return_code} but got {result.returncode} for command: {cmd}"
