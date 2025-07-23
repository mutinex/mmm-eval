# This file contains the constants for the validation tests


class ValidationTestConstants:
    """Constants for the validation tests."""

    ACCURACY_TEST_SIZE = 8
    RANDOM_STATE = 42
    N_SPLITS = 3
    TIME_SERIES_CROSS_VALIDATION_TEST_SIZE = 4

    class PerturbationConstants:
        """Constants for the perturbation test."""

        GAUSSIAN_NOISE_SCALE = 0.05
        GAUSSIAN_NOISE_LOC = 0
