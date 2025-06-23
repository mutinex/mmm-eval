"""Adapters for different MMM frameworks."""

from typing import Any

from .base import BaseAdapter
from .experimental.pymc import PyMCAdapter

# Registry of available adapters
ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {
    "pymc-marketing": PyMCAdapter,
}


def get_adapter(framework: str, config: dict[str, Any] | None = None) -> BaseAdapter:
    """Get an adapter instance for the specified framework.

    Args:
        framework: Name of the framework
        config: Configuration dictionary

    Returns:
        Adapter instance

    Raises:
        ValueError: If framework is not supported

    """
    if framework not in ADAPTER_REGISTRY:
        raise ValueError(f"Unsupported framework: {framework}. Available: {list(ADAPTER_REGISTRY.keys())}")

    adapter_class = ADAPTER_REGISTRY[framework]
    return adapter_class(config or {})


__all__ = [
    "PyMCAdapter",
    "get_adapter",
    "ADAPTER_REGISTRY",
]
