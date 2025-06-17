import pytest
import subprocess
import pandas as pd
import json
import os


@pytest.mark.parametrize(
    "cmd_template,expected",
    [
        (
            "mmm-eval --input-data-path {data_path} --framework pymc3 --target-column kpi --output-path {output_path} --config-path {config_path}",
            True,
        ),
        (
            "mmm-eval --target-column kpi --output-path {output_path} --config-path {config_path}",
            False,
        ),
        (
            "mmm-eval --input-data-path {data_path} --framework meridian",
            True,
        ),
        (
            "mmm-eval --input-data-path {data_path} --framework robyn --target-column kpi --output-path {output_path} --config-path {config_path}",
            False,
        ),
    ],
)
def test_cli_as_subprocess(tmp_path, cmd_template, expected):
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

    if expected:
        assert result.returncode == 0
    else:
        assert result.returncode != 0, f"Expected failure but got success: {cmd}"
