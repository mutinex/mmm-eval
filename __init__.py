"""
mmm-eval: A unified evaluation framework for Media Mix Models (MMMs).

This package provides a standardized interface for evaluating different MMM frameworks
including Meridian, PyMC, Robyn, and LightweightMMM.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core functionality
from mmm_eval.core.evaluator import evaluate_framework
from mmm_eval.core.results import EvaluationResults
from mmm_eval.adapters.base import BaseAdapter
from mmm_eval.adapters import (
    MeridianAdapter,
    PyMCAdapter,
    get_adapter,
)

# Metrics
from mmm_eval.metrics.accuracy import mape, rmse, mae, r_squared, symmetric_mape

# Data utilities
from mmm_eval.data.loaders import load_csv, load_from_database, DataLoader

__all__ = [
    # Core API
    "evaluate_framework",
    "EvaluationResults",
    "BaseAdapter",
    # Metrics
    "mape",
    "rmse",
    "mae",
    "r_squared",
    "symmetric_mape",
    # Adapters
    "get_adapter",
    # "MeridianAdapter",
    "PyMCAdapter",
    # Data utilities
    "load_csv",
    "load_from_database",
    "DataLoader",
]
