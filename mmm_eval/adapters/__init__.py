"""Adapters for different MMM frameworks."""

from mmm_eval.configs import PyMCConfig, MeridianConfig
from mmm_eval.core.validation_tests_models import FrameworkNames

from .base import BaseAdapter
from .pymc import PyMCAdapter
from .meridian import MeridianAdapter

# Registry of available adapters
ADAPTER_REGISTRY = {
    FrameworkNames.PYMC_MARKETING: PyMCAdapter,
    FrameworkNames.MERIDIAN: MeridianAdapter,
}


def get_adapter(framework: FrameworkNames, config: PyMCConfig | MeridianConfig):
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
    "MeridianAdapter",
    "get_adapter",
    "ADAPTER_REGISTRY",
]
