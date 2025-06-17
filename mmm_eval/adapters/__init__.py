"""
Framework adapters for MMM evaluation.
"""

from typing import Dict, Any, Optional
from .base import BaseAdapter
from .meridian import MeridianAdapter
from .pymc import PyMCAdapter


# Registry of available adapters
ADAPTER_REGISTRY = {
    "meridian": MeridianAdapter,
    "pymc": PyMCAdapter,
    "pymc3": PyMCAdapter,  # Alias for backwards compatibility
}


def get_adapter(framework: str, config: Optional[Dict[str, Any]] = None) -> BaseAdapter:
    """
    Get an adapter instance for the specified framework.

    Args:
        framework: Name of the MMM framework
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
    return adapter_class(config)


__all__ = [
    "BaseAdapter",
    "MeridianAdapter",
    "PyMCAdapter",
    "get_adapter",
    "ADAPTER_REGISTRY",
]
