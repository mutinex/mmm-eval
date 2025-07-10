# This file contains the threshold constants for the different tests

# todo(): Review these figures and if we still want to use them before releasing


class AccuracyThresholdConstants:
    """Constants for the accuracy threshold."""

    MAPE = 0.15
    R_SQUARED = 0.8


class CrossValidationThresholdConstants:
    """Constants for the cross-validation threshold."""

    MEAN_MAPE = 0.15
    STD_MAPE = 0.03
    MEAN_R_SQUARED = 0.8


class RefreshStabilityThresholdConstants:
    """Constants for the refresh stability threshold."""

    MEAN_PERCENTAGE_CHANGE = 0.15
    STD_PERCENTAGE_CHANGE = 0.03


class PerturbationThresholdConstants:
    """Constants for the perturbation threshold."""

    PERCENTAGE_CHANGE = 0.08
