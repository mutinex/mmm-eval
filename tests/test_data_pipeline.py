"""Unit tests for DataPipeline."""

import pandas as pd
import pytest

from mmm_eval.adapters import SupportedFrameworks
from mmm_eval.data import DataPipeline
from mmm_eval.data.constants import InputDataframeConstants
from mmm_eval.data.exceptions import DataValidationError


class TestDataPipeline:
    """Test DataPipeline functionality."""

    def _get_test_df(self):
        """Create test DataFrame."""
        return pd.DataFrame(
            {
                "custom_date": pd.date_range("2023-01-01", periods=40).strftime("%Y-%m-%d"),
                "custom_response": [100.0] * 40,
                "custom_revenue": [1000.0] * 40,
                "control_var1": [0.5] * 40,  # Control column
                "facebook": [100.0] * 40,  # Channel column
            }
        )

    def test_complete_pipeline(self):
        """Test complete pipeline with valid data."""
        df = self._get_test_df()

        # Run pipeline
        pipeline = DataPipeline(
            data=df,
            framework=SupportedFrameworks.PYMC_MARKETING,
            control_columns=["control_var1"],
            channel_columns=["facebook"],
            date_column="custom_date",
            response_column="custom_response",
            revenue_column="custom_revenue",
        )
        result = pipeline.run()

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (40, 5)  # 5 columns including the renamed ones
        assert "custom_date" in result.columns
        assert InputDataframeConstants.RESPONSE_COL in result.columns
        assert InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["custom_date"])

    def test_pipeline_with_default_settings(self):
        """Test pipeline with default column names."""
        # Create test CSV with default column names
        df = self._get_test_df()

        # Run pipeline with default settings
        pipeline = DataPipeline(
            data=df,
            framework=SupportedFrameworks.PYMC_MARKETING,
            control_columns=["control_var1"],
            channel_columns=["facebook"],
            date_column="custom_date",
            response_column="custom_response",
            revenue_column="custom_revenue",
        )
        result = pipeline.run()

        assert isinstance(result, pd.DataFrame)
        assert "custom_date" in result.columns
        assert InputDataframeConstants.RESPONSE_COL in result.columns
        assert InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["custom_date"])

    def test_pipeline_with_custom_settings(self):
        """Test pipeline with custom column names."""
        # Create test CSV
        df = self._get_test_df()

        # Run pipeline with custom column names
        pipeline = DataPipeline(
            data=df,
            framework=SupportedFrameworks.PYMC_MARKETING,
            control_columns=["control_var1"],
            channel_columns=["facebook"],
            date_column="custom_date",
            response_column="custom_response",
            revenue_column="custom_revenue",
        )
        result = pipeline.run()

        assert isinstance(result, pd.DataFrame)
        assert "custom_date" in result.columns
        assert InputDataframeConstants.RESPONSE_COL in result.columns  # Should be renamed
        assert InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL in result.columns  # Should be renamed
        assert pd.api.types.is_datetime64_any_dtype(result["custom_date"])

    def test_pipeline_fails_with_invalid_data(self):
        """Test pipeline fails with invalid data."""
        # Create test CSV with insufficient data
        df = self._get_test_df()
        df = df.head(10)

        # Run pipeline with strict requirements
        pipeline = DataPipeline(
            data=df,
            framework=SupportedFrameworks.PYMC_MARKETING,
            control_columns=["control_var1"],
            channel_columns=["facebook"],
            date_column="custom_date",
            response_column="custom_response",
            revenue_column="custom_revenue",
        )

        with pytest.raises(DataValidationError):
            pipeline.run()
