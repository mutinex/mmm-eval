"""
Core evaluation functionality for MMM frameworks.
"""

from .evaluator import evaluate_framework
from .results import EvaluationResults

__all__ = [
    "evaluate_framework",
    "EvaluationResults",
    "BaseAdapter",
]
