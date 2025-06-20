"""
Framework adapters for MMM evaluation.
"""

from typing import Dict, Any, Optional
import pandas as pd
from .meridian import MeridianAdapter
from .pymc import PyMCAdapter


# Registry of available adapters
ADAPTER_REGISTRY = {
    "meridian": MeridianAdapter,
    "pymc-marketing": PyMCAdapter,
}


def get_adapter(framework: str, data: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
    """
    Get an adapter instance for the specified framework.

    Args:
        framework: Name of the MMM framework
        data: Input data to run the model on
        config: Framework-specific configuration

    Returns:
        Adapter instance for the framework

    Raises:
        ValueError: If framework is not supported
    """
    framework = framework.lower()

    if framework not in ADAPTER_REGISTRY:
        available = list(ADAPTER_REGISTRY.keys())
        raise ValueError(f"Unsupported framework: {framework}. Available: {available}")

    adapter_class = ADAPTER_REGISTRY[framework]
    return adapter_class(config, data)


__all__ = [
    "MeridianAdapter",
    "PyMCAdapter",
    "get_adapter",
    "ADAPTER_REGISTRY",
]
