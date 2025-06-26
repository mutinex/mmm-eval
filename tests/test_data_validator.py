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
                "control_var1": [0.5] * 25,  # Control column
            }
        )

        validator = DataValidator(
            date_column=InputDataframeConstants.DATE_COL,
            response_column=InputDataframeConstants.RESPONSE_COL,
            revenue_column=InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL,
            control_columns=["control_var1"],
            min_number_observations=21,
        )
        validator.run_validations(df)  # Should not raise any exceptions

    def test_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()

        validator = DataValidator(
            date_column=InputDataframeConstants.DATE_COL,
            response_column=InputDataframeConstants.RESPONSE_COL,
            revenue_column=InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL,
            control_columns=["control_var1"],
        )
        with pytest.raises(EmptyDataFrameError):
            validator.run_validations(df)

    def test_insufficient_data_size(self):
        """Test validation with insufficient data size."""
        df = pd.DataFrame(
            {
                InputDataframeConstants.DATE_COL: pd.date_range("2023-01-01", periods=10),
                InputDataframeConstants.RESPONSE_COL: [100.0] * 10,
                InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL: [1000.0] * 10,
                "control_var1": [0.5] * 10,  # Control column
            }
        )

        validator = DataValidator(
            date_column=InputDataframeConstants.DATE_COL,
            response_column=InputDataframeConstants.RESPONSE_COL,
            revenue_column=InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL,
            control_columns=["control_var1"],
            min_number_observations=21,
        )
        with pytest.raises(DataValidationError):
            validator.run_validations(df)

    def test_pandera_schema_null_validation(self):
        """Test that Pandera schema directly catches null values."""
        df = pd.DataFrame(
            {
                InputDataframeConstants.DATE_COL: pd.date_range("2023-01-01", periods=25),
                InputDataframeConstants.RESPONSE_COL: [100.0] * 25,
                InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL: [1000.0] * 25,
                "control_var1": [0.5] * 25,  # Control column
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
                "control_var1": [0.5] * 25,  # Control column
                # Missing revenue column
            }
        )

        # Test direct Pandera schema validation
        with pytest.raises(pa.errors.SchemaError):  # Should raise Pandera SchemaError for missing column
            ValidatedDataSchema.validate(df)
