import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from click.testing import CliRunner

from mmm_eval.cli.evaluate import main
from mmm_eval.data.synth_data_generator import generate_meridian_data, generate_pymc_data
from tests.test_configs.test_configs import SAMPLE_CONFIG_JSON

# Meridian config JSON for e2e testing
SAMPLE_MERIDIAN_CONFIG_JSON = {
    "input_data_builder_config": {
        "date_column": "date",
        "media_channels": ["Channel0", "Channel1"],
        "channel_spend_columns": ["Channel0_spend", "Channel1_spend"],
        "response_column": "conversions",
        "control_columns": ["GQV", "Competitor_Sales"],
        "non_media_treatment_columns": ["Discount"],
        "organic_media_columns": ["Channel5_impression"],
        "organic_media_channels": ["Channel5"],
    },
    "model_spec_config": {
        "prior": "PriorDistribution(roi_m=tfp.distributions.LogNormal(0.2, 0.9))",
        "media_effects_dist": "log_normal",
        "hill_before_adstock": False,
        "max_lag": 8,
        "organic_media_prior_type": "contribution",
        "non_media_treatments_prior_type": "contribution",
    },
    "sample_posterior_config": {
        "n_chains": 1,
        "n_adapt": 10,
        "n_burnin": 10,
        "n_keep": 10,
        "seed": 123,
    },
    "revenue_column": "revenue",
    "response_column": "conversions",
}


def test_cli_e2e_pymc_marketing(tmp_path):
    """End-to-end test for PyMC Marketing framework using Click test runner."""
    # Set up test data
    data_path = tmp_path / "data.csv"
    config_path = tmp_path / "test_config.json"
    output_path = tmp_path / "output"

    # Create sample data with required columns
    sample_data = generate_pymc_data()
    sample_data.to_csv(data_path, index=False)

    # Save dummy config and pass in, we mock this later
    with open(config_path, "w") as f:
        json.dump(SAMPLE_CONFIG_JSON, f, default=str, indent=2)

    os.makedirs(output_path, exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input-data-path",
            str(data_path),
            "--framework",
            "pymc_marketing",
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


def test_cli_e2e_meridian(tmp_path):
    """End-to-end test for Meridian framework with model training mocked."""
    # Set up test data
    data_path = tmp_path / "data.csv"
    config_path = tmp_path / "test_config.json"
    output_path = tmp_path / "output"

    # Create sample data
    sample_data = generate_meridian_data()
    sample_data.to_csv(data_path, index=False)

    # Use the proper Meridian config structure
    config_json = SAMPLE_MERIDIAN_CONFIG_JSON.copy()

    with open(config_path, "w") as f:
        json.dump(config_json, f, default=str, indent=2)

    os.makedirs(output_path, exist_ok=True)

    runner = CliRunner()

    # Patch the Meridian model's sample_posterior method to avoid actual training
    with (
        patch("mmm_eval.adapters.meridian.Meridian.sample_posterior", return_value=MagicMock(name="FakeTrace")),
        patch("mmm_eval.adapters.meridian.MeridianAdapter.fit", return_value=None),
        patch("mmm_eval.adapters.meridian.MeridianAdapter.fit_and_predict") as mock_fit_and_predict,
        patch("mmm_eval.adapters.meridian.MeridianAdapter.get_channel_roi") as mock_get_channel_roi,
        patch("mmm_eval.adapters.meridian.Analyzer.expected_outcome", return_value=MagicMock()),
    ):

        # Create a simple mock that returns predictions matching the test data length
        def mock_fit_and_predict_side_effect(*args, **kwargs):
            # The test data should be the last argument
            test_data = args[-1] if args else kwargs.get("test")
            if test_data is not None:
                # The actual data is grouped by date, so we need to count unique dates
                unique_dates = test_data["date"].nunique()
                return np.ones(unique_dates) * 100.0
            else:
                # Fallback to a reasonable default
                return np.ones(10) * 100.0

        # Mock get_channel_roi to return a simple Series
        def mock_get_channel_roi_side_effect(*args, **kwargs):
            return pd.Series({"Channel0": 1.5, "Channel1": 2.0})

        mock_fit_and_predict.side_effect = mock_fit_and_predict_side_effect
        mock_get_channel_roi.side_effect = mock_get_channel_roi_side_effect

        result = runner.invoke(
            main,
            [
                "--input-data-path",
                str(data_path),
                "--framework",
                "meridian",
                "--output-path",
                str(output_path),
                "--config-path",
                str(config_path),
                "--verbose",
            ],
        )

    assert result.exit_code == 0, f"CLI command failed with exit code {result.exit_code}. Output: {result.output}"
