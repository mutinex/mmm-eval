import json
import logging
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from click.testing import CliRunner

from mmm_eval.adapters.meridian import MeridianAdapter
from mmm_eval.cli.evaluate import main
from mmm_eval.data.synth_data_generator import generate_meridian_data, generate_pymc_data
from tests.test_configs.test_configs import SAMPLE_CONFIG_JSON

# Set up logging to see debug output
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

    # Create a mock adapter class that inherits from MeridianAdapter
    class MockMeridianAdapter(MeridianAdapter):
        def __init__(self, config):
            super().__init__(config)
            # Set up mock methods
            self.is_fitted = False

        def fit(self, data, max_train_date=None):
            self.is_fitted = True
            return None

        def fit_and_predict(self, train, test):
            # The test data should be the last argument
            if test is not None:
                # The actual data is grouped by date, so we need to count unique dates
                unique_dates = test["date"].nunique()
                return np.ones(unique_dates) * 100.0
            else:
                # Fallback to a reasonable default
                return np.ones(10) * 100.0

        def fit_and_predict_in_sample(self, data):
            if data is not None:
                # The test groups by date and sums the response column
                # Check which column name is available
                response_col = None
                if "response" in data.columns:
                    response_col = "response"
                elif "conversions" in data.columns:
                    response_col = "conversions"
                else:
                    # Fallback to a reasonable default
                    return np.ones(10) * 100.0

                # So we need to return predictions matching the number of unique dates
                grouped = data.groupby("date")[response_col].sum()
                predictions = np.ones(len(grouped)) * 100.0
                return predictions
            else:
                # Fallback to a reasonable default
                return np.ones(10) * 100.0

        def get_channel_roi(self, start_date=None, end_date=None):
            result = pd.Series({"Channel0": 1.5, "Channel1": 2.0})
            return result

    # Mock the adapter factory to return our mock adapter
    def mock_get_adapter(framework, config):
        if framework == "meridian":
            return MockMeridianAdapter(config)
        else:
            # For other frameworks, use the original logic
            from mmm_eval.adapters import get_adapter as original_get_adapter

            return original_get_adapter(framework, config)

    # Apply patches
    with (
        patch("mmm_eval.adapters.meridian.Meridian.sample_posterior", return_value=MagicMock(name="FakeTrace")),
        patch("mmm_eval.core.evaluator.get_adapter", side_effect=mock_get_adapter),
        patch("mmm_eval.adapters.meridian.Analyzer.expected_outcome", return_value=MagicMock()),
    ):
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
