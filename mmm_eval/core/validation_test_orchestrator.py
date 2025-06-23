"""Test orchestrator for MMM validation framework.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)
from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.core.base_validation_test import BaseValidationTest
from mmm_eval.core.validation_tests_models import ValidationTestNames
from mmm_eval.core.validation_test_results import ValidationResults, ValidationTestResult

# Import standard tests
from mmm_eval.core.validation_tests import (
    AccuracyTest,
    CrossValidationTest,
    PerturbationTest,
    RefreshStabilityTest,
)


class ValidationTestOrchestrator:
    """Main orchestrator for running validation tests.

    This class manages the test registry and executes tests
    in sequence, aggregating their results.
    """

    def __init__(self):
        """Initialize the validator with standard tests pre-registered."""
        self.tests: dict[ValidationTestNames, type[BaseValidationTest]] = {
            ValidationTestNames.ACCURACY: AccuracyTest,
            ValidationTestNames.CROSS_VALIDATION: CrossValidationTest,
            ValidationTestNames.REFRESH_STABILITY: RefreshStabilityTest,
            ValidationTestNames.PERTURBATION: PerturbationTest,
        }

    def _get_all_test_names(self) -> list[ValidationTestNames]:
        """Get all test names from the registry."""
        return list(self.tests.keys())

    def validate(
        self,
        adapter: BaseAdapter,
        data: pd.DataFrame,
        test_names: list[ValidationTestNames],
    ) -> ValidationResults:
        """Run validation tests on the model.

        Args:
            model: Model to validate
            data: Input data for validation
            test_names: List of test names to run

        Returns:
            ValidationResults containing all test results

        Raises:
            ValueError: If any requested test is not registered
        
        """
        # Run tests and collect results
        results: dict[ValidationTestNames, ValidationTestResult] = {}
        for test_name in test_names:
            logger.info(f"Running test: {test_name}")
            test_instance = self.tests[test_name]()
            test_result = test_instance.run_with_error_handling(adapter, data)
            results[test_name] = test_result

        return ValidationResults(results)
