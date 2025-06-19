"""
Test orchestrator for MMM validation framework.
"""

from typing import Dict, List, Any, Optional, Type
import pandas as pd

from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.core.validation_tests_models import ValidationTestNames
from mmm_eval.core.base_validation_test import BaseValidationTest
from .validation_test_results import TestResult, ValidationResult

# Import standard tests
from mmm_eval.core.validation_tests import AccuracyTest, RefreshStabilityTest, CrossValidationTest, PerturbationTest


class ValidationTestOrchestrator:
    """
    Main orchestrator for running validation tests.
    
    This class manages the test registry and executes tests
    in sequence, aggregating their results.
    """
    
    def __init__(self):
        """Initialize the validator with standard tests pre-registered."""
        self.tests: Dict[ValidationTestNames, BaseValidationTest] = {
            ValidationTestNames.ACCURACY: AccuracyTest,
            ValidationTestNames.REFRESH_STABILITY: RefreshStabilityTest,
            ValidationTestNames.CROSS_VALIDATION: CrossValidationTest,
            ValidationTestNames.PERTUBATION: PerturbationTest,
        }

    def _get_all_test_names(self) -> List[ValidationTestNames]:
        """Get all test names from the registry."""
        return list(self.tests.keys())
    
    
    def validate(
        self,
        model: BaseAdapter,
        data: pd.DataFrame,
        test_names: List[ValidationTestNames]
    ) -> ValidationResult:
        """
        Run validation tests on the model.
        
        Args:
            model: Model to validate
            data: Input data for validation
            test_names: List of test names to run
            
        Returns:
            ValidationResult containing all test results
            
        Raises:
            ValueError: If any requested test is not registered
        """
        
        # Run tests and collect results
        results: Dict[ValidationTestNames, TestResult] = {}
        for test_name in test_names:
            test_instance = self.tests[test_name]()
            test_result = test_instance.run_with_error_handling(model, data)
            results[test_name] = test_result
        
        return ValidationResult(results) 