"""mmm-eval: A unified evaluation framework for Media Mix Models (MMMs).

This package provides a standardized interface for evaluating different MMM frameworks
including Meridian, PyMC, Robyn, and LightweightMMM.
"""

__version__ = "0.1.0"
__author__ = "King Kang and the Jungle Boys"
__email__ = "your.email@example.com"

# Core functionality
# Adapters
from .adapters import (
    get_adapter,
)
from .adapters.experimental.pymc import PyMCAdapter
from .configs import PyMCConfig, PyMCConfigRehydrator, get_config
from .core.base_validation_test import BaseValidationTest
from .core.evaluator import Evaluator
from .core.validation_test_results import ValidationResults, ValidationTestResult
from .core.validation_tests_models import ValidationTestNames
from .core import run_evaluation

# Data utilities
from .data import (
    DataLoader,
    DataPipeline,
    DataProcessor,
    DataValidator,
)

# Metrics
from .metrics.accuracy_functions import (
    calculate_mean_for_singular_values_across_cross_validation_folds,
    calculate_means_for_series_across_cross_validation_folds,
    calculate_std_for_singular_values_across_cross_validation_folds,
    calculate_stds_for_series_across_cross_validation_folds,
)

__all__ = [
    "get_adapter",
    "PyMCConfig",
    "PyMCConfigRehydrator",
    "get_config",
    "PyMCAdapter",
    # Core API
    "Evaluator",
    "ValidationTestResult",
    "ValidationResults",
    "BaseValidationTest",
    "ValidationTestNames",
    "run_evaluation",
    # Metrics
    "calculate_means_for_series_across_cross_validation_folds",
    "calculate_stds_for_series_across_cross_validation_folds",
    "calculate_mean_for_singular_values_across_cross_validation_folds",
    "calculate_std_for_singular_values_across_cross_validation_folds",
    # Adapters
    "get_adapter",
    "PyMCAdapter",
    # Data utilities
    "DataLoader",
    "DataProcessor",
    "DataValidator",
    "DataPipeline",
]
