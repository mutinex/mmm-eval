"""Main evaluator for MMM frameworks."""

from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path

import pandas as pd

from mmm_eval.configs import EvalConfig
from .validation_test_results import ValidationResult
from mmm_eval.adapters import get_adapter
from mmm_eval.core.exceptions import InvalidTestNameError
from mmm_eval.core.validation_test_orchestrator import ValidationTestOrchestrator
from mmm_eval.core.validation_tests_models import ValidationTestNames
from mmm_eval.core.validation_test_results import ValidationResult


class Evaluator:
    """
    Main evaluator class for MMM frameworks.

    This class provides a unified interface for evaluating different MMM frameworks
    using standardized validation tests.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.validation_orchestrator = ValidationTestOrchestrator()

    def _get_test_names(self, test_names: List[str]) -> List[ValidationTestNames]:
        """
        Parse test names from strings to ValidationTestNames enum objects.

        Args:
            test_names: List of test names as strings or enum objects

        Returns:
            List of ValidationTestNames enum objects

        Raises:
            ValueError: If any test name is invalid
        """

        converted_names = []
        for test_name in test_names:
            try:
                converted_names.append(ValidationTestNames(test_name))
            except ValueError:
                raise InvalidTestNameError(
                    f"Invalid test name: '{test_name}'. Valid names: {ValidationTestNames.all_tests_as_str()}"
                )

        return converted_names

    def evaluate_framework(
        self,
        framework: str,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        test_names: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> ValidationResult:
        """
        Evaluate an MMM framework using the unified API.

        Args:
            framework: Name of the MMM framework to evaluate
            data: Input data containing media channels, KPI, and other variables
            config: Framework-specific configuration
            test_names: List of test names to run (can be strings or ValidationTestNames enum objects)
            output_path: Path to save results
            **kwargs: Additional framework-specific parameters

        Returns:
            ValidationResult object containing evaluation metrics and predictions

        Raises:
            ValueError: If any test name is invalid
        """
        # Parse test names to enum objects if needed
        test_names = (
            self._get_test_names(test_names)
            if test_names
            else self.validation_orchestrator._get_all_test_names()
        )

        # Initialize the adapter
        adapter = get_adapter(framework, config)

        # Run validation tests
        validation_results = self.validation_orchestrator.validate(
            adapter=adapter,
            data=data,
            test_names=test_names,
        )

        # Save results if output path is provided
        if output_path:
            # TODO: Implement result saving logic
            pass

        return validation_results
