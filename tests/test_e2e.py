import json
import os

from click.testing import CliRunner

from mmm_eval.cli.evaluate import main
from mmm_eval.data.synth_data_generator import generate_data
from tests.test_configs.test_configs import SAMPLE_CONFIG_JSON


def test_cli_e2e_pymc_marketing(tmp_path):
    """End-to-end test for PyMC Marketing framework using Click test runner."""
    # Set up test data
    data_path = tmp_path / "data.csv"
    config_path = tmp_path / "test_config.json"
    output_path = tmp_path / "output"

    # Create sample data with required columns
    sample_data = generate_data()
    sample_data.to_csv(data_path, index=False)

    # Save dummy config and pass in, we mock this later
    # with open(config_path, "w") as f:
    #     json.dump({"dummy": "config"}, f, indent=2)
    with open(config_path, "w") as f:
        json.dump(SAMPLE_CONFIG_JSON, f, default=str, indent=2)

    os.makedirs(output_path, exist_ok=True)

    # Mock the config loading to return the full valid config
    # valid_config = valid_hydration_config_1()

    # with patch("mmm_eval.cli.evaluate.load_config", return_value=valid_config):
    # Test using Click's test runner (enables debugging)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input-data-path",
            str(data_path),
            "--framework",
            "pymc-marketing",
            "--output-path",
            str(output_path),
            "--config-path",
            str(config_path),
            "--verbose",
        ],
    )

    # Print debug info
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    # Check that the CLI command succeeded
    assert result.exit_code == 0, f"CLI command failed with exit code {result.exit_code}. Output: {result.output}"
