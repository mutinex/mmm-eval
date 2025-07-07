"""Unit tests for DataPipeline."""

import pandas as pd
import pytest

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

    def _get_meridian_test_df(self):
        """Create test DataFrame for Meridian framework."""
        return pd.DataFrame(
            {
                "geo": ["US"] * 40,
                "time": pd.date_range("2023-01-01", periods=40).strftime("%Y-%m-%d"),
                "Channel0_impression": [1000000] * 40,
                "Channel1_impression": [800000] * 40,
                "Channel2_impression": [600000] * 40,
                "Channel3_impression": [400000] * 40,
                "Channel4_impression": [200000] * 40,
                "Channel5_impression": [150000] * 40,
                "Competitor_Sales": [0.5] * 40,
                "Discount": [0.1] * 40,
                "GQV": [0.3] * 40,
                "Channel0_spend": [50000] * 40,
                "Channel1_spend": [40000] * 40,
                "Channel2_spend": [30000] * 40,
                "Channel3_spend": [20000] * 40,
                "Channel4_spend": [10000] * 40,
                "Channel5_spend": [5000] * 40,
                "response": [1000] * 40,
                "revenue": [10.0] * 40,
                "population": [1000000] * 40,
            }
        )

    def test_complete_pipeline(self):
        """Test complete pipeline with valid data."""
        df = self._get_test_df()

        # Run pipeline
        pipeline = DataPipeline(
            data=df,
            framework="pymc_marketing",
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

    def test_meridian_pipeline(self):
        """Test complete pipeline with valid Meridian data."""
        df = self._get_meridian_test_df()

        # Run pipeline for Meridian framework
        pipeline = DataPipeline(
            data=df,
            framework="meridian",
            control_columns=["Competitor_Sales", "GQV"],
            channel_columns=[
                "Channel0_spend",
                "Channel1_spend",
                "Channel2_spend",
                "Channel3_spend",
                "Channel4_spend",
                "Channel5_spend",
            ],
            date_column="time",
            response_column="response",
            revenue_column="revenue",
        )
        result = pipeline.run()

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (40, 20)  # All original columns plus renamed ones
        assert "time" in result.columns
        assert InputDataframeConstants.RESPONSE_COL in result.columns
        assert InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["time"])

        # Check that Meridian-specific columns are preserved
        assert "geo" in result.columns
        assert "population" in result.columns
        assert "Channel0_impression" in result.columns
        assert "Channel5_impression" in result.columns
        assert "Competitor_Sales" in result.columns
        assert "GQV" in result.columns
        assert "Discount" in result.columns

    def test_pipeline_with_default_settings(self):
        """Test pipeline with default column names."""
        # Create test CSV with default column names
        df = self._get_test_df()

        # Run pipeline with default settings
        pipeline = DataPipeline(
            data=df,
            framework="pymc_marketing",
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
            framework="pymc_marketing",
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
            framework="pymc_marketing",
            control_columns=["control_var1"],
            channel_columns=["facebook"],
            date_column="custom_date",
            response_column="custom_response",
            revenue_column="custom_revenue",
        )

        with pytest.raises(DataValidationError):
            pipeline.run()

    def test_meridian_pipeline_fails_with_invalid_data(self):
        """Test Meridian pipeline fails with invalid data."""
        # Create test CSV with insufficient data
        df = self._get_meridian_test_df()
        df = df.head(10)  # Only 10 observations

        # Run pipeline with strict requirements
        pipeline = DataPipeline(
            data=df,
            framework="meridian",
            control_columns=["Competitor_Sales", "GQV"],
            channel_columns=[
                "Channel0_spend",
                "Channel1_spend",
                "Channel2_spend",
                "Channel3_spend",
                "Channel4_spend",
                "Channel5_spend",
            ],
            date_column="time",
            response_column="response",
            revenue_column="revenue",
        )

        with pytest.raises(DataValidationError):
            pipeline.run()
