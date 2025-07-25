"""Test CLI evaluation functionality."""

import pytest
from click.testing import CliRunner
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

from mmm_eval.cli.evaluate import main
from mmm_eval.configs import PyMCConfig
from mmm_eval.data.synth_data_generator import generate_pymc_data

DUMMY_MODEL = MMM(
    date_column="date_week",
    channel_columns=["channel_1", "channel_2"],
    control_columns=["price", "event_1", "event_2"],
    adstock=GeometricAdstock(l_max=4),
    saturation=LogisticSaturation(),
    yearly_seasonality=2,
)
FIT_KWARGS = {"target_accept": 0.9, "chains": 1, "draws": 50, "tune": 50, "random_seed": 123}
REVENUE_COLUMN = "revenue"
RESPONSE_COLUMN = "quantity"


@pytest.mark.parametrize(
    "cli_args,expected_exit_code,test_name",
    [
        (
            [
                "--input-data-path",
                "{data_path}",
                "--framework",
                "pymc_marketing",
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
                "pymc_marketing",
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
        # test_data = _create_test_data()
        test_data = generate_pymc_data()
        test_data.to_csv(data_path, index=False)

    config = PyMCConfig.from_model_object(
        DUMMY_MODEL, fit_kwargs=FIT_KWARGS, revenue_column=REVENUE_COLUMN, response_column=RESPONSE_COLUMN
    )
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

    # Check that the CLI command returned the expected exit code
    assert (
        result.exit_code == expected_exit_code
    ), f"""Test '{test_name}' failed: Expected exit code {expected_exit_code} but got {result.exit_code}. 
           Args: {formatted_args}. Output: {result.output}
           Exception: {result.exception}
        """
