"""Unit tests for DataProcessor."""

import pandas as pd
import pytest

from mmm_eval.data import DataProcessor
from mmm_eval.data.constants import InputDataframeConstants
from mmm_eval.data.exceptions import InvalidDateFormatError, MissingRequiredColumnsError


class TestDataProcessor:
    """Test DataProcessor functionality."""

    def _get_test_df(self):
        """Create test DataFrame."""
        return pd.DataFrame(
            {
                "custom_date": ["2023-01-01", "2023-01-02"],
                "custom_response": [100.0, 150.0],
                "custom_revenue": [1000.0, 1500.0],
                "control_var1": [0.5, 0.6],  # Control column
                "facebook": [100.0, 150.0],  # Channel column
            }
        )

    def test_parse_dates_and_rename_columns(self):
        """Test date parsing and column renaming functionality."""
        df = self._get_test_df()

        processor = DataProcessor(
            control_columns=["control_var1"],
            channel_columns=["facebook"],
            date_column="custom_date",
            response_column="custom_response",
            revenue_column="custom_revenue",
        )
        result = processor.process(df)

        # Check that columns are renamed to standard names
        assert "custom_date" in result.columns
        assert InputDataframeConstants.RESPONSE_COL in result.columns
        assert InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL in result.columns

        # Check that date is parsed correctly
        assert pd.api.types.is_datetime64_any_dtype(result["custom_date"])
        assert result["custom_date"].iloc[0] == pd.Timestamp("2023-01-01")

        # Check that other columns are preserved
        assert "control_var1" in result.columns
        assert "facebook" in result.columns

    def test_missing_date_column(self):
        """Test error when date column is missing."""
        df = pd.DataFrame(
            {
                InputDataframeConstants.RESPONSE_COL: [100.0, 150.0],
                InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL: [1000.0, 1500.0],
                "control_var1": [0.5, 0.6],  # Control column
                "facebook": [100.0, 150.0],  # Channel column
            }
        )

        processor = DataProcessor(
            control_columns=["control_var1"],
            channel_columns=["facebook"],
        )
        with pytest.raises(MissingRequiredColumnsError):
            processor.process(df)

    def test_missing_response_column(self):
        """Test error when response column is missing."""
        df = pd.DataFrame(
            {
                InputDataframeConstants.DATE_COL: ["2023-01-01", "2023-01-02"],
                InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL: [1000.0, 1500.0],
                "control_var1": [0.5, 0.6],  # Control column
                "facebook": [100.0, 150.0],  # Channel column
            }
        )

        processor = DataProcessor(
            control_columns=["control_var1"],
            channel_columns=["facebook"],
        )
        with pytest.raises(MissingRequiredColumnsError):
            processor.process(df)

    def test_missing_revenue_column(self):
        """Test error when revenue column is missing."""
        df = pd.DataFrame(
            {
                InputDataframeConstants.DATE_COL: ["2023-01-01", "2023-01-02"],
                InputDataframeConstants.RESPONSE_COL: [100.0, 150.0],
                "control_var1": [0.5, 0.6],  # Control column
                "facebook": [100.0, 150.0],  # Channel column
            }
        )

        processor = DataProcessor(
            control_columns=["control_var1"],
            channel_columns=["facebook"],
        )
        with pytest.raises(MissingRequiredColumnsError):
            processor.process(df)

    def test_invalid_date_format(self):
        """Test error when date format cannot be parsed."""
        df = pd.DataFrame(
            {
                "custom_date": ["2023-01-01", "not-a-date", "2023-01-03"],
                "custom_response": [100.0, 150.0, 200.0],
                "custom_revenue": [1000.0, 1500.0, 2000.0],
                "control_var1": [0.5, 0.6, 0.7],  # Control column
                "facebook": [100.0, 150.0, 200.0],  # Channel column
            }
        )

        processor = DataProcessor(
            control_columns=["control_var1"],
            channel_columns=["facebook"],
            date_column="custom_date",
            response_column="custom_response",
            revenue_column="custom_revenue",
        )
        with pytest.raises(InvalidDateFormatError):
            processor.process(df)
