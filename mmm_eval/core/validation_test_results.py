"""Result containers for MMM validation framework."""

from datetime import datetime

import pandas as pd

from mmm_eval.core.validation_tests_models import (
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
            metric_names: List of metric names
            test_scores: Computed metric results

        """
        self.test_name = test_name
        self.metric_names = metric_names
        self.test_scores = test_scores
        self.timestamp = datetime.now()

    def to_df(self) -> pd.DataFrame:
        """Convert test results to a flat DataFrame format."""
        test_scores_df = self.test_scores.to_df()
        test_scores_df[ValidationTestAttributeNames.TEST_NAME.value] = self.test_name.value
        test_scores_df[ValidationTestAttributeNames.TIMESTAMP.value] = self.timestamp.isoformat()
        return test_scores_df


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

    def get_test_result(self, test_name: ValidationTestNames) -> ValidationTestResult:
        """Get results for a specific test."""
        return self.test_results[test_name]

    def to_df(self) -> pd.DataFrame:
        """Convert validation results to a flat DataFrame format."""
        return pd.concat(
            [self.get_test_result(test_name).to_df() for test_name in self.test_results.keys()],
            ignore_index=True,
        )
