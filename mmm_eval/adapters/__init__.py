"""Adapters for different MMM frameworks."""

from enum import StrEnum

from mmm_eval.configs import PyMCConfig

from .base import BaseAdapter
from .pymc import PyMCAdapter


class SupportedFrameworks(StrEnum):
    """Define the names of supported MMM frameworks."""

    PYMC_MARKETING = "pymc-marketing"
    MERIDIAN = "meridian"

    @classmethod
    def all_frameworks(cls) -> list["SupportedFrameworks"]:
        """Return all framework names as a list."""
        return list(cls)

    @classmethod
    def all_frameworks_as_str(cls) -> list[str]:
        """Return all framework names as a list of strings."""
        return [framework.value for framework in cls]


# Registry of available adapters
ADAPTER_REGISTRY = {
    SupportedFrameworks.PYMC_MARKETING: PyMCAdapter,
}


def get_adapter(framework: SupportedFrameworks, config: PyMCConfig):
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
    "SupportedFrameworks",
    "get_adapter",
    "ADAPTER_REGISTRY",
]
