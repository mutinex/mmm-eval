"""Main evaluator for MMM frameworks."""

from pathlib import Path

import pandas as pd

from mmm_eval.configs import EvalConfig
from .validation_test_results import ValidationResult
from mmm_eval.adapters import get_adapter
from mmm_eval.core.validation_test_orchestrator import ValidationTestOrchestrator
from mmm_eval.core.validation_tests_models import ValidationTestNames
from mmm_eval.core.validation_test_results import ValidationResult


def evaluate_framework(
    framework: str,
    data: pd.DataFrame,
    config: BaseConfig,
    output_path: Path | None = None,
    test_names: Optional[List[ValidationTestNames]] = None,
) -> ValidationResult:
    """
    Evaluate an MMM framework using the unified API.

    Args:
        framework: Name of the MMM framework to evaluate
        data: Input data containing media channels, KPI, and other variables
        config: Framework-specific configuration
        test_names: List of test names to run
        **kwargs: Additional framework-specific parameters

    Returns:
        EvaluationResults object containing evaluation metrics and predictions

    """

    # Initialize the adapter
    adapter = get_adapter(framework, config)

    # Initialize the validation test orchestrator
    validation_test_orchestrator = ValidationTestOrchestrator()
    
    # Run validation tests
    validation_results = validation_test_orchestrator.validate(
        model=adapter,
        data=data,
        test_names=test_names,
    )
    
    # Save results if output path is provided
    if output_path:
        # TODO: Implement result saving logic
        pass
    
    return validation_results
