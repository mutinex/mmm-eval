"""mmm-eval: A unified evaluation framework for Media Mix Models (MMMs).

This package provides a standardized interface for evaluating different MMM frameworks
including Meridian, PyMC, Robyn, and LightweightMMM.
"""

__version__ = "0.1.0"
__author__ = "King Kang and the Jungle Boys"
__email__ = "your.email@example.com"

# Core functionality
from .core.evaluator import Evaluator
from .core.validation_tests_models import ValidationTestNames
from .core.validation_test_results import ValidationResult
from .core.base_validation_test import BaseValidationTest
from .core.validation_test_results import TestResult, ValidationResult
from .core.base_validation_test import BaseValidationTest

# Metrics
from .metrics.accuracy_functions import calculate_mape, calculate_r_squared, calculate_mean_for_cross_validation_folds, calculate_std_for_cross_validation_folds
# Adapters
from .adapters import get_adapter
from .adapters.experimental.pymc import PyMCAdapter
from .configs import PyMCConfig, PyMCConfigRehydrator, get_config
from .adapters import (
    get_adapter,
    # MeridianAdapter,
    # PyMCAdapter,
)
from .configs import PyMCConfig, PyMCConfigRehydrator
from .core.evaluator import Evaluator

# Data utilities
from .data import (
    DataLoader,
    DataPipeline,
    DataProcessor,
    DataValidator,
)

__all__ = [
    "get_adapter",
    "PyMCConfig",
    "PyMCConfigRehydrator",
    "get_config",
    "evaluate_framework",
    "PyMCAdapter",
    "EvaluationResults",
    # Core API
    "Evaluator",
    "TestResult",
    "ValidationResult",
    "BaseValidationTest",
    "ValidationTestNames",
    # Metrics
    "calculate_mape",
    "calculate_r_squared",
    "calculate_mean_for_cross_validation_folds",
    "calculate_std_for_cross_validation_folds",
    "calculate_absolute_percentage_change",
    # Adapters
    "get_adapter",
    "PyMCAdapter",
    "MeridianAdapter",
    # Data utilities
    "DataLoader",
    "DataProcessor",
    "DataValidator",
    "DataPipeline",
]
