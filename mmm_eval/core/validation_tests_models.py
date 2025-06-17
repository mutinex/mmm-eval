

from enum import Enum


class ValidationTestNames(Enum):

    """Define the names of the validation tests"""

    ACCURACY = "accuracy"
    STABILITY = "stability"
    CROSS_VALIDATION = "cross_validation"
    PERTUBATION = "perturbation"

class ValidationTestAttributeNames(Enum):

    """Define the names of the validation test attributes"""

    TEST_NAME = "test_name"
    PASSED = "passed"
    METRIC_NAMES = "metric_names"
    TEST_SCORES = "test_scores"
    TIMESTAMP = "timestamp"

class ValidationResultAttributeNames(Enum):

    """Define the names of the validation result attributes"""

    TIMESTAMP = "timestamp"
    ALL_PASSED = "all_passed"
    RESULTS = "results"