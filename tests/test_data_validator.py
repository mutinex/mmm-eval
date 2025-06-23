"""Unit tests for DataValidator."""

import pandas as pd
import pandera.pandas as pa
import pytest

from mmm_eval.data import DataValidator
from mmm_eval.data.constants import InputDataframeConstants
from mmm_eval.data.exceptions import DataValidationError, EmptyDataFrameError
from mmm_eval.data.schemas import ValidatedDataSchema


class TestDataValidator:
    """Test DataValidator functionality."""

    def test_valid_data(self):
        """Test validation of valid data."""
        df = pd.DataFrame(
            {
                InputDataframeConstants.DATE_COL: pd.date_range("2023-01-01", periods=25),
                InputDataframeConstants.RESPONSE_COL: [100.0] * 25,
                InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL: [1000.0] * 25,
            }
        )

        validator = DataValidator(min_data_size=21)
        validator.run_validations(df)  # Should not raise any exceptions

    def test_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()

        validator = DataValidator()
        with pytest.raises(EmptyDataFrameError):
            validator.run_validations(df)

    def test_insufficient_data_size(self):
        """Test validation with insufficient data size."""
        df = pd.DataFrame(
            {
                InputDataframeConstants.DATE_COL: pd.date_range("2023-01-01", periods=10),
                InputDataframeConstants.RESPONSE_COL: [100.0] * 10,
                InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL: [1000.0] * 10,
            }
        )

        validator = DataValidator(min_data_size=21)
        with pytest.raises(DataValidationError):
            validator.run_validations(df)

    def test_pandera_schema_null_validation(self):
        """Test that Pandera schema directly catches null values."""
        df = pd.DataFrame(
            {
                InputDataframeConstants.DATE_COL: pd.date_range("2023-01-01", periods=25),
                InputDataframeConstants.RESPONSE_COL: [100.0] * 25,
                InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL: [1000.0] * 25,
            }
        )
        df.loc[0, InputDataframeConstants.RESPONSE_COL] = None

        # Test direct Pandera schema validation
        with pytest.raises(pa.errors.SchemaError):  # Should raise Pandera SchemaError directly
            ValidatedDataSchema.validate(df)

    def test_pandera_schema_missing_columns(self):
        """Test that Pandera schema catches missing required columns."""
        df = pd.DataFrame(
            {
                InputDataframeConstants.DATE_COL: pd.date_range("2023-01-01", periods=25),
                InputDataframeConstants.RESPONSE_COL: [100.0] * 25,
                # Missing revenue column
            }
        )

        # Test direct Pandera schema validation
        with pytest.raises(pa.errors.SchemaError):  # Should raise Pandera SchemaError for missing column
            ValidatedDataSchema.validate(df)
