# This file contains the threshold constants for the different tests

# todo(): Review these figures and if we still want to use them before releasing


class AccuracyThresholdConstants:
    """Constants for the accuracy threshold."""

    MAPE = 15.0
    SMAPE = 15.0
    R_SQUARED = 0.8


class CrossValidationThresholdConstants:
    """Constants for the cross-validation threshold."""

    MEAN_MAPE = 15.0
    STD_MAPE = 3.0
    MEAN_SMAPE = 15.0
    STD_SMAPE = 3.0
    MEAN_R_SQUARED = 0.8


class RefreshStabilityThresholdConstants:
    """Constants for the refresh stability threshold."""

    MEAN_PERCENTAGE_CHANGE = 0.15
    STD_PERCENTAGE_CHANGE = 0.03


class PerturbationThresholdConstants:
    """Constants for the perturbation threshold."""

    PERCENTAGE_CHANGE = 0.08
