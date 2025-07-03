# This file contains the models for the validation tests

# todo(): Possibly move to the constants file

from enum import StrEnum


class SupportedFrameworks(StrEnum):
    """Define the names of supported MMM frameworks."""

    PYMC_MARKETING = "pymc-marketing"
    MERIDIAN = "meridian"

    @classmethod
    def all_frameworks(cls) -> list["SupportedFrameworks"]:
        """Return all framework names as a list."""
        return list(cls)

    @classmethod
    def all_frameworks_as_str(cls) -> list[str]:
        """Return all framework names as a list of strings."""
        return [framework.value for framework in cls]


class ValidationTestNames(StrEnum):
    """Define the names of the validation tests."""

    ACCURACY = "accuracy"
    CROSS_VALIDATION = "cross_validation"
    REFRESH_STABILITY = "refresh_stability"
    PERTURBATION = "perturbation"

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
