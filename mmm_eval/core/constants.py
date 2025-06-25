# This file contains the constants for the validation tests


class ValidationTestConstants:
    """Constants for the validation tests."""

    TRAIN_TEST_SPLIT_TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_SPLITS = 5
    TIME_SERIES_CROSS_VALIDATION_TEST_SIZE = 4  # 4 representing a 4 week refresh

    class PerturbationConstants:
        """Constants for the perturbation test."""

        GAUSSIAN_NOISE_SCALE = 0.05
        GAUSSIAN_NOISE_LOC = 0
