"""Result containers for MMM validation framework."""

from datetime import datetime
from typing import Any

import pandas as pd

from mmm_eval.core.validation_tests_models import (
    ValidationResultAttributeNames,
    ValidationTestAttributeNames,
    ValidationTestNames,
)
from mmm_eval.metrics.metric_models import (
    AccuracyMetricResults,
    CrossValidationMetricResults,
    PerturbationMetricResults,
    RefreshStabilityMetricResults,
)


class ValidationTestResult:
    """Container for individual test results.

    This class holds the results of a single validation test,
    including pass/fail status, metrics, and any error messages.
    """

    def __init__(
        self,
        test_name: ValidationTestNames,
        passed: bool,
        metric_names: list[str],
        test_scores: (
            AccuracyMetricResults
            | CrossValidationMetricResults
            | RefreshStabilityMetricResults
            | PerturbationMetricResults
        ),
    ):
        """Initialize test results.

        Args:
            test_name: Name of the test
            passed: Whether the test passed
            metric_names: List of metric names
            test_scores: Computed metric results

        """
        self.test_name = test_name
        self.passed = passed
        self.metric_names = metric_names
        self.test_scores = test_scores
        self.timestamp = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary format."""
        return {
            ValidationTestAttributeNames.TEST_NAME.value: self.test_name.value,
            # todo(): Perhaps set as false permanently or dont use if we dont want thresholds
            ValidationTestAttributeNames.PASSED.value: self.passed,
            ValidationTestAttributeNames.METRIC_NAMES.value: self.metric_names,
            ValidationTestAttributeNames.TEST_SCORES.value: self.test_scores.to_dict(),
            ValidationTestAttributeNames.TIMESTAMP.value: self.timestamp.isoformat(),
        }


class ValidationResults:
    """Container for complete validation results.

    This class holds the results of all validation tests run,
    including individual test results and overall summary.
    """

    def __init__(self, test_results: dict[ValidationTestNames, ValidationTestResult]):
        """Initialize validation results.

        Args:
            test_results: Dictionary mapping test names to their results

        """
        self.test_results = test_results
        self.timestamp = datetime.now()

    def get_test_result(self, test_name: ValidationTestNames) -> ValidationTestResult:
        """Get results for a specific test."""
        return self.test_results[test_name]

    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return all(result.passed for result in self.test_results.values())

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary format."""
        return {
            ValidationResultAttributeNames.TIMESTAMP.value: self.timestamp.isoformat(),
            ValidationResultAttributeNames.ALL_PASSED.value: self.all_passed(),
            ValidationResultAttributeNames.RESULTS.value: {
                result.test_name.value: result.to_dict() for result in self.test_results.values()
            },
        }

    def to_df(self) -> pd.DataFrame:
        """Convert nested test results to a flat DataFrame format."""
        rows = []

        for result in self.test_results.values():
            test_name = result.test_name.value
            passed = result.passed
            test_scores_dict = result.test_scores.to_dict()

            for metric_key, value in test_scores_dict.items():
                if isinstance(value, pd.Series):
                    for subkey, subval in value.items():
                        rows.append(
                            {
                                "test_name": test_name,
                                "metric_name": f"{metric_key}:{subkey}",
                                "metric_value": subval,
                                "metric_pass": passed,
                            }
                        )
                else:
                    rows.append(
                        {
                            "test_name": test_name,
                            "metric_name": metric_key,
                            "metric_value": value,
                            "metric_pass": passed,
                        }
                    )

        return pd.DataFrame(rows)
