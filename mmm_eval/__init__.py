"""
mmm-eval: A unified evaluation framework for Media Mix Models (MMMs).

This package provides a standardized interface for evaluating different MMM frameworks
including Meridian, PyMC, Robyn, and LightweightMMM.
"""

__version__ = "0.1.0"
__author__ = "King Kang and the Jungle Boys"
__email__ = "your.email@example.com"

# Core functionality
from .core.evaluator import evaluate_framework
from .core.results import EvaluationResults

# Metrics
from .metrics.accuracy import mape, rmse, mae, r_squared, symmetric_mape

# Adapters
from .adapters import (
    get_adapter,
    MeridianAdapter,
    PyMCAdapter,
)

# Data utilities
from .data.loaders import load_csv, load_from_database, DataLoader, save_config

__all__ = [
    # Core API
    "evaluate_framework",
    "EvaluationResults",
    # Metrics
    "mape",
    "rmse",
    "mae",
    "r_squared",
    "symmetric_mape",
    # Adapters
    "get_adapter",
    "MeridianAdapter",
    "PyMCAdapter",
    # Data utilities
    "load_csv",
    "load_from_database",
    "DataLoader",
    "save_config",
]
