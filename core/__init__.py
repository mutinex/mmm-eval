"""
Core evaluation functionality for MMM frameworks.
"""

from .evaluator import evaluate_framework
from .results import EvaluationResults
from .base import BaseAdapter

__all__ = [
    "evaluate_framework",
    "EvaluationResults",
    "BaseAdapter",
]
