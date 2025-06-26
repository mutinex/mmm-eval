"""
mmm-eval: A unified evaluation framework for Media Mix Models (MMMs).

This package provides a standardized interface for evaluating different MMM frameworks
including Meridian, PyMC, Robyn, and LightweightMMM.
"""

__version__ = "0.4.2"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core functionality
from mmm_eval.core.evaluator import Evaluator
from mmm_eval.core.validation_test_results import ValidationTestResult, ValidationResults
from mmm_eval.core.validation_tests_models import ValidationTestNames
from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.adapters import (
    get_adapter,
    PyMCAdapter,
)

# Metrics
from mmm_eval.metrics.accuracy_functions import calculate_mape, calculate_r_squared, calculate_mean_for_singular_values_across_cross_validation_folds, calculate_std_for_singular_values_across_cross_validation_folds, calculate_means_for_series_across_cross_validation_folds, calculate_stds_for_series_across_cross_validation_folds
from mmm_eval.adapters.experimental.pymc import PyMCAdapter
# Data utilities
from mmm_eval.data.loaders import load_csv, DataLoader

__all__ = [
    # Core API
    "Evaluator",
    "ValidationTestResult",
    "ValidationResults",
    "BaseValidationTest",
    "ValidationTestNames",
    "BaseAdapter",
    # Metrics
    "calculate_mape",
    "calculate_r_squared",
    "calculate_mean_for_singular_values_across_cross_validation_folds",
    "calculate_std_for_singular_values_across_cross_validation_folds",
    "calculate_means_for_series_across_cross_validation_folds",
    "calculate_stds_for_series_across_cross_validation_folds",
    # Adapters
    "get_adapter",
    "PyMCAdapter",
    # Data utilities
    "load_csv",
    "DataLoader",
]
