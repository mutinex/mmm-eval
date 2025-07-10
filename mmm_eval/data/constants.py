"""Defines the constants for the data pipeline."""


class InputDataframeConstants:
    """Constants for the dataframe."""

    DATE_COL = "date"
    MEDIA_CHANNEL_REVENUE_COL = "revenue"
    MEDIA_CHANNEL_SPEND_COL = "spend"
    RESPONSE_COL = "response"


class DataLoaderConstants:
    """Constants for the data loader."""

    class ValidDataExtensions:
        """Valid data extensions."""

        CSV = "csv"
        PARQUET = "parquet"

        @classmethod
        def all(cls):
            """Return list of all supported file extensions."""
            return [cls.CSV, cls.PARQUET]


class DataPipelineConstants:
    """Constants for the data pipeline."""

    MIN_NUMBER_OBSERVATIONS = 40
