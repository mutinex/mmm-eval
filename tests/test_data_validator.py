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
            framework="pymc_marketing",
            date_column=InputDataframeConstants.DATE_COL,
            response_column=InputDataframeConstants.RESPONSE_COL,
            revenue_column=InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL,
            control_columns=["control_var1"],
            min_number_observations=21,
        )
        validator.run_validations(df)  # Should not raise any exceptions

    def test_valid_meridian_data(self):
        """Test validation of valid Meridian data."""
        df = pd.DataFrame(
            {
                "geo": ["US"] * 25,
                "time": pd.date_range("2023-01-01", periods=25),
                "Channel0_impression": [1000000] * 25,
                "Channel1_impression": [800000] * 25,
                "Channel2_impression": [600000] * 25,
                "Channel3_impression": [400000] * 25,
                "Channel4_impression": [200000] * 25,
                "Channel5_impression": [150000] * 25,
                "Competitor_Sales": [0.5] * 25,
                "Discount": [0.1] * 25,
                "GQV": [0.3] * 25,
                "Channel0_spend": [50000] * 25,
                "Channel1_spend": [40000] * 25,
                "Channel2_spend": [30000] * 25,
                "Channel3_spend": [20000] * 25,
                "Channel4_spend": [10000] * 25,
                "Channel5_spend": [5000] * 25,
                "response": [1000] * 25,
                "revenue": [10.0] * 25,
                "population": [1000000] * 25,
            }
        )

        validator = DataValidator(
            framework="meridian",
            date_column="time",
            response_column="response",
            revenue_column="revenue",
            control_columns=["Competitor_Sales", "GQV"],
            min_number_observations=21,
        )
        validator.run_validations(df)  # Should not raise any exceptions

    def test_meridian_data_with_missing_required_columns(self):
        """Test validation of Meridian data with missing required columns."""
        df = pd.DataFrame(
            {
                "geo": ["US"] * 25,
                "time": pd.date_range("2023-01-01", periods=25),
                "Channel0_spend": [50000] * 25,
                "Channel1_spend": [40000] * 25,
                # Missing response column
                "revenue": [10.0] * 25,
            }
        )

        validator = DataValidator(
            framework="meridian",
            date_column="time",
            response_column="response",  # This column is missing from the DataFrame
            revenue_column="revenue",
            control_columns=["Competitor_Sales"],
            min_number_observations=21,
        )
        with pytest.raises(pa.errors.SchemaError):  # Schema validation catches missing columns first
            validator.run_validations(df)

    def test_meridian_data_with_insufficient_observations(self):
        """Test validation of Meridian data with insufficient observations."""
        df = pd.DataFrame(
            {
                "geo": ["US"] * 10,  # Only 10 observations
                "time": pd.date_range("2023-01-01", periods=10),
                "Channel0_impression": [1000000] * 10,
                "Channel1_impression": [800000] * 10,
                "Competitor_Sales": [0.5] * 10,
                "Discount": [0.1] * 10,
                "GQV": [0.3] * 10,
                "Channel0_spend": [50000] * 10,
                "Channel1_spend": [40000] * 10,
                "response": [1000] * 10,
                "revenue": [10.0] * 10,
                "population": [1000000] * 10,
            }
        )

        validator = DataValidator(
            framework="meridian",
            date_column="time",
            response_column="response",
            revenue_column="revenue",
            control_columns=["Competitor_Sales", "GQV"],
            min_number_observations=21,  # Require 21 observations
        )
        with pytest.raises(DataValidationError):
            validator.run_validations(df)

    def test_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()

        validator = DataValidator(
            framework="pymc_marketing",
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
            framework="pymc_marketing",
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
