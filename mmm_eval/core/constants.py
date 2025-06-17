# This file contains the constants for the validation tests

class ValidationTestConstants:
    """
    Constants for the validation tests
    """

    TRAIN_TEST_SPLIT_RATIO = 0.2
    RANDOM_STATE = 42
    N_SPLITS = 5
    TIME_SERIES_CROSS_VALIDATION_TEST_SIZE = 4 # 4 representing a 4 week refresh

class ValidationDataframeConstants:
    """
    Constants for the validation dataframe
    """

    PERCENTAGE_CHANGE_CHANNEL_CONTRIBUTION_COL = "percentage_change_channel_contribution"