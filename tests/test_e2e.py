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
            # Completely override the fit method to avoid calling real Meridian
            self.is_fitted = True
            # Set up mock attributes that other methods expect
            self.training_data = MagicMock()
            self.model = MagicMock()
            self.trace = MagicMock()
            self.analyzer = MagicMock()
            return None

        def fit_and_predict(self, train, test):
            # Return predictions matching the number of unique dates in test data
            # The data pipeline renames 'conversions' to 'response', so we need to use the processed column name
            unique_dates = test["date"].nunique()
            return np.ones(unique_dates) * 100.0

        def fit_and_predict_in_sample(self, data):
            # Return predictions matching the number of unique dates in full data
            # The data pipeline renames 'conversions' to 'response'
            unique_dates = data["date"].nunique()
            return np.ones(unique_dates) * 100.0

        def get_channel_roi(self, start_date=None, end_date=None):
            # Dynamically create ROI results based on current media channels
            roi_dict = {}
            for channel in self.media_channels:
                if channel.endswith("_shuffled"):
                    # Give shuffled channels a low ROI (should pass threshold of -50%)
                    roi_dict[channel] = -60.0
                else:
                    # Give original channels normal ROI
                    roi_dict[channel] = 1.5 if channel == "Channel0" else 2.0
            return pd.Series(roi_dict)

        @property
        def primary_media_regressor_type(self):
            """Return the type of primary media regressors."""
            from mmm_eval.adapters.base import PrimaryMediaRegressor

            return PrimaryMediaRegressor.SPEND

        @property
        def primary_media_regressor_columns(self) -> list[str]:
            """Return the primary media regressor columns."""
            return self.channel_spend_columns

        def get_channel_names(self) -> list[str]:
            """Get the channel names that would be used as the index in get_channel_roi results."""
            return self.media_channels

        def copy(self) -> "MockMeridianAdapter":
            """Create a deep copy of this adapter with all configuration."""
            new_adapter = MockMeridianAdapter(self.config)
            new_adapter.channel_spend_columns = self.channel_spend_columns.copy()
            # Don't try to set media_channels directly since it's a property
            return new_adapter

        def get_primary_media_regressor_columns_for_channels(self, channel_names: list[str]) -> list[str]:
            """Get the primary media regressor columns for specific channels."""
            return channel_names

        def _get_original_channel_columns(self, channel_name: str) -> dict[str, str]:
            """Get the original column names for a channel."""
            # For mock adapter, assume channel names are the same as column names
            return {"spend": channel_name}

        def _get_shuffled_col_name(self, shuffled_channel_name: str, column_type: str, original_col: str) -> str:
            """Get the name for a shuffled column based on the mock adapter's naming convention."""
            # For mock adapter, use the same convention as Meridian (with suffix)
            return f"{shuffled_channel_name}_{column_type}"

        def _create_adapter_with_placebo_channel(
            self, original_channel: str, shuffled_channel: str, original_columns: dict[str, str]
        ) -> "MockMeridianAdapter":
            """Create a new adapter instance configured to use the placebo channel."""
            new_adapter = MockMeridianAdapter(self.config)
            new_adapter.channel_spend_columns = self.channel_spend_columns + [f"{shuffled_channel}_spend"]
            # Add to the underlying schema instead of trying to set the property
            new_adapter.input_data_builder_schema.media_channels.append(shuffled_channel)
            return new_adapter

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
