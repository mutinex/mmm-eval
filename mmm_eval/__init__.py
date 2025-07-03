"""mmm-eval: A unified evaluation framework for Media Mix Models (MMMs).

This package provides a standardized interface for evaluating different MMM frameworks
such as PyMC-marketing and Google Meridian.
"""

__author__ = "King Kang and the Jungle Boys"
__email__ = "joseph.kang@mutinex.co"

# Core functionality
# Adapters
from .adapters import (
    get_adapter,
)
from .adapters.pymc import PyMCAdapter
from .adapters.meridian import MeridianAdapter
from .configs import PyMCConfig, MeridianConfig, PyMCConfigRehydrator, MeridianConfigRehydrator, get_config
from .adapters.schemas import (
    MeridianPriorDistributionSchema,
    MeridianModelSpecSchema,
    MeridianInputDataBuilderSchema,
    MeridianSamplePosteriorSchema,
)
from .core import run_evaluation
from .core.base_validation_test import BaseValidationTest
from .core.evaluator import Evaluator
from .core.validation_test_results import ValidationResults, ValidationTestResult
from .core.validation_tests_models import ValidationTestNames

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

# Utils
from .utils import save_results

# Setup logging when package is imported
from .utils.logging import setup_logging

setup_logging()

__all__ = [
    # Configs
    "PyMCConfig",
    "MeridianConfig",
    "PyMCConfigRehydrator",
    "MeridianConfigRehydrator",
    "get_config",
    # Schemas
    "MeridianPriorDistributionSchema",
    "MeridianModelSpecSchema",
    "MeridianInputDataBuilderSchema",
    "MeridianSamplePosteriorSchema",
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
    "MeridianAdapter",
    # Data utilities
    "DataLoader",
    "DataProcessor",
    "DataValidator",
    "DataPipeline",
    # Utils
    "save_results",
]
