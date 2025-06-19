# This file defines the constants for the dataframe

class InputDataframeConstants:
    """
    Constants for the dataframe
    """

    DATE_COL = "date"
    MEDIA_CHANNEL_COL = "media_channel"
    MEDIA_CHANNEL_VOLUME_CONTRIBUTION_COL = "contribution_volume"
    MEDIA_CHANNEL_REVENUE_COL = "revenue"
    MEDIA_CHANNEL_SPEND_COL = "spend"

class DataLoaderConstants:

    class ValidDataExtensions:
        CSV = "csv"
        PARQUET = "parquet"
        
        @classmethod
        def to_list(cls):
            """Return list of all supported file extensions."""
            return [cls.CSV, cls.PARQUET]
        
class DataPipelineConstants:
    """
    Constants for the data pipeline
    """

    MIN_DATA_SIZE = 21