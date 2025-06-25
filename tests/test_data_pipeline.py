"""Unit tests for DataPipeline."""

import pandas as pd
import pytest

from mmm_eval.data import DataPipeline
from mmm_eval.data.constants import DataLoaderConstants, InputDataframeConstants
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
                "facebook": ["facebook"] * 40,
                "spend": [1000.0] * 40,
            }
        )

    def test_complete_pipeline(self, tmp_path):
        """Test complete pipeline with valid data."""
        df = self._get_test_df()
        csv_path = tmp_path / f"test.{DataLoaderConstants.ValidDataExtensions.CSV}"
        df.to_csv(csv_path, index=False)

        # Run pipeline
        pipeline = DataPipeline(
            data_path=csv_path,
            control_columns=["control_var1"],
            channel_columns=["facebook"],
            date_column="custom_date",
            response_column="custom_response",
            revenue_column="custom_revenue",
            min_number_observations=40,
        )
        result = pipeline.run()

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (40, 5)  # 5 columns including the renamed ones
        assert InputDataframeConstants.DATE_COL in result.columns
        assert InputDataframeConstants.RESPONSE_COL in result.columns
        assert InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result[InputDataframeConstants.DATE_COL])

    def test_pipeline_with_default_settings(self, tmp_path):
        """Test pipeline with default column names."""
        # Create test CSV with default column names
        df = pd.DataFrame(
            {
                InputDataframeConstants.DATE_COL: pd.date_range("2023-01-01", periods=40).strftime("%Y-%m-%d"),
                InputDataframeConstants.RESPONSE_COL: [100.0] * 40,
                InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL: [1000.0] * 40,
                "facebook": ["facebook"] * 40,
                "spend": [1000.0] * 40,
            }
        )
        csv_path = tmp_path / f"test.{DataLoaderConstants.ValidDataExtensions.CSV}"
        df.to_csv(csv_path, index=False)

        # Run pipeline with default settings
        pipeline = DataPipeline(data_path=csv_path, min_number_observations=40)
        result = pipeline.run()

        assert isinstance(result, pd.DataFrame)
        assert InputDataframeConstants.DATE_COL in result.columns
        assert InputDataframeConstants.RESPONSE_COL in result.columns
        assert InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result[InputDataframeConstants.DATE_COL])

    def test_pipeline_with_custom_settings(self, tmp_path):
        """Test pipeline with custom column names."""
        # Create test CSV
        df = self._get_test_df()
        csv_path = tmp_path / f"test.{DataLoaderConstants.ValidDataExtensions.CSV}"
        df.to_csv(csv_path, index=False)

        # Run pipeline with custom column names
        pipeline = DataPipeline(
            data_path=csv_path,
            control_columns=["control_var1"],
            channel_columns=["facebook"],
            date_column="custom_date",
            response_column="custom_response",
            revenue_column="custom_revenue",
            min_number_observations=10,
        )
        result = pipeline.run()

        assert isinstance(result, pd.DataFrame)
        assert InputDataframeConstants.DATE_COL in result.columns  # Should be renamed
        assert InputDataframeConstants.RESPONSE_COL in result.columns  # Should be renamed
        assert InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL in result.columns  # Should be renamed
        assert pd.api.types.is_datetime64_any_dtype(result[InputDataframeConstants.DATE_COL])

    def test_pipeline_fails_with_invalid_data(self, tmp_path):
        """Test pipeline fails with invalid data."""
        # Create test CSV with insufficient data
        df = self._get_test_df()
        df = df.head(10)
        csv_path = tmp_path / f"test.{DataLoaderConstants.ValidDataExtensions.CSV}"
        df.to_csv(csv_path, index=False)

        # Run pipeline with strict requirements
        pipeline = DataPipeline(
            data_path=csv_path,
            control_columns=["control_var1"],
            channel_columns=["facebook"],
            date_column="custom_date",
            response_column="custom_response",
            revenue_column="custom_revenue",
            min_number_observations=40,
        )

        with pytest.raises(DataValidationError):
            pipeline.run()
