"""Framework configs for MMM evaluation."""

from mmm_eval.configs.base import BaseConfig

from .configs import PyMCConfig, MeridianConfig

# Registry of available configs
CONFIG_REGISTRY = {
    "pymc-marketing": PyMCConfig,
    "meridian": MeridianConfig,
}


def get_config(framework: str, config_path: str) -> BaseConfig:
    """Get an config instance for the specified framework.

    Args:
        framework: Name of the MMM framework
        config_path: Path to framework-specific configuration

    Returns:
        BaseConfig instance for the framework

    Raises:
        ValueError: If framework is not supported

    """
    framework = framework.lower()

    if framework not in CONFIG_REGISTRY:
        available = list(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unsupported framework: {framework}. Available: {available}")

    config_class = CONFIG_REGISTRY[framework]
    return config_class.load_model_config_from_json(config_path)


__all__ = [
    "PyMCConfig",
    "MeridianConfig",
    "get_config",
    "CONFIG_REGISTRY",
]
