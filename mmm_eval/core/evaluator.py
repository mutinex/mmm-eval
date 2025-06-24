"""Main evaluator for MMM frameworks."""

from pathlib import Path

import pandas as pd

from mmm_eval.configs.base import BaseConfig
from mmm_eval.adapters import get_adapter
from mmm_eval.core.exceptions import InvalidTestNameError
from mmm_eval.core.validation_test_orchestrator import ValidationTestOrchestrator
from mmm_eval.core.validation_test_results import ValidationResults
from mmm_eval.core.validation_tests_models import ValidationTestNames


class Evaluator:
    """Main evaluator class for MMM frameworks.

    This class provides a unified interface for evaluating different MMM frameworks
    using standardized validation tests.
    """

    def __init__(self, data: pd.DataFrame, output_path: Path | None = None, test_names: tuple[str, ...] | None = None):
        """Initialize the evaluator."""
        self.validation_orchestrator = ValidationTestOrchestrator()
        self.data = data
        self.output_path = output_path
        self.test_names = (
            self._get_test_names(test_names) if test_names else self.validation_orchestrator._get_all_test_names()
        )

    def _get_test_names(self, test_names: tuple[str, ...]) -> list[ValidationTestNames]:
        """Parse test names from strings to ValidationTestNames enum objects.

        Args:
            test_names: Tuple of test names as strings

        Returns:
            List of ValidationTestNames enum objects

        Raises:
            ValueError: If any test name is invalid

        """
        converted_names = []
        for test_name in test_names:
            try:
                converted_names.append(ValidationTestNames(test_name))
            except ValueError as e:
                raise InvalidTestNameError(
                    f"Invalid test name: '{test_name}'. Valid names: {ValidationTestNames.all_tests_as_str()}"
                ) from e

        return converted_names

    def evaluate_framework(
        self,
        framework: str,
        config: BaseConfig
    ) -> ValidationResults:
        """Evaluate an MMM framework using the unified API.

        Args:
            framework: Name of the MMM framework to evaluate
            config: Framework-specific configuration

        Returns:
            ValidationResult object containing evaluation metrics and predictions

        Raises:
            ValueError: If any test name is invalid

        """
        # Initialize the adapter
        adapter = get_adapter(framework, config)

        # Run validation tests
        validation_results = self.validation_orchestrator.validate(
            adapter=adapter,
            data=self.data,
            test_names=self.test_names,
        )

        # Save results if output path is provided
        if self.output_path:
            # TODO: Implement result saving logic
            pass

        return validation_results
