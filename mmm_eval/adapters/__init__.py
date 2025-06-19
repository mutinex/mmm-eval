"""Adapters for different MMM frameworks."""

from mmm_eval.configs import PyMCConfig
from typing import Dict, Any, Optional
# from .pymc import PyMCAdapter
from .experimental.pymc import PyMCAdapter #todo(): remove this once PyMCAdapter is promoted out of experimental

# Registry of available adapters
ADAPTER_REGISTRY = {
    "pymc-marketing": PyMCAdapter,
}


def get_adapter(framework: str, config: PyMCConfig):
    """Get an adapter instance for the specified framework.

    Args:
        framework: Name of the MMM framework
        config: Framework-specific configuration

    Returns:
        Adapter instance

    Raises:
        ValueError: If framework is not supported

    """
    if framework not in ADAPTER_REGISTRY:
        raise ValueError(f"Unsupported framework: {framework}. Available: {list(ADAPTER_REGISTRY.keys())}")

    adapter_class = ADAPTER_REGISTRY[framework]
    return adapter_class(config)


__all__ = [
    "PyMCAdapter",
    "get_adapter",
    "ADAPTER_REGISTRY",
]
