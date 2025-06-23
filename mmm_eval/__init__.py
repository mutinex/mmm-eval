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
    PyMCAdapter,
    get_adapter,
)
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
    "PyMCAdapter",
    # Data utilities
    "DataLoader",
    "DataProcessor",
    "DataValidator",
    "DataPipeline",
]
