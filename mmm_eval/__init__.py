"""mmm-eval: A unified evaluation framework for Media Mix Models (MMMs).

This package provides a standardized interface for evaluating different MMM frameworks
such as PyMC-marketing.
"""

__author__ = "King Kang and the Jungle Boys"
__email__ = "joseph.kang@mutinex.co"

# Core functionality
# Adapters
from .adapters import get_adapter
from .adapters.experimental.pymc import PyMCAdapter
from .configs import PyMCConfig, PyMCConfigRehydrator, get_config
from .core.evaluator import evaluate_framework
from .core.results import EvaluationResults

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
    # Metrics
    "mape",
    "rmse",
    "mae",
    "r_squared",
    "symmetric_mape",
    # Data utilities
    "DataLoader",
    "DataProcessor",
    "DataValidator",
    "DataPipeline",
]
