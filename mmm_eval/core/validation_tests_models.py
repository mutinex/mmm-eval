# This file contains the models for the validation tests

# todo(): Possibly move to the constants file
# todo(): add enum for framework names

from enum import StrEnum


class ValidationTestNames(StrEnum):
    """Define the names of the validation tests."""

    HOLDOUT_ACCURACY = "holdout_accuracy"
    IN_SAMPLE_ACCURACY = "in_sample_accuracy"
    CROSS_VALIDATION = "cross_validation"
    REFRESH_STABILITY = "refresh_stability"
    PERTURBATION = "perturbation"
    PLACEBO = "placebo"

    @classmethod
    def all_tests(cls) -> list["ValidationTestNames"]:
        """Return all validation test names as a list."""
        return list(cls)

    @classmethod
    def all_tests_as_str(cls) -> list[str]:
        """Return all validation test names as a list of strings."""
        return [test.value for test in cls]


class ValidationTestAttributeNames(StrEnum):
    """Define the names of the validation test attributes."""

    TEST_NAME = "test_name"
    METRIC_NAMES = "metric_names"
    TEST_SCORES = "test_scores"
    TIMESTAMP = "timestamp"


class ValidationResultAttributeNames(StrEnum):
    """Define the names of the validation result attributes."""

    TIMESTAMP = "timestamp"
    RESULTS = "results"
