"""Framework configs for MMM evaluation."""

from .configs import Config, EvalConfig, PyMCConfig

# Registry of available configs
CONFIG_REGISTRY = {
    "pymc-marketing": PyMCConfig,
}


def get_config(framework: str, config_path: str) -> EvalConfig:
    """Get an config instance for the specified framework.

    Args:
        framework: Name of the MMM framework
        config_path: Path to framework-specific configuration

    Returns:
        Config instance for the framework

    Raises:
        ValueError: If framework is not supported

    """
    framework = framework.lower()

    if framework not in CONFIG_REGISTRY:
        available = list(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unsupported framework: {framework}. Available: {available}")

    config_class = CONFIG_REGISTRY[framework]
    return config_class.load_config(config_path)


__all__ = [
    "EvalConfig",
    "PyMCConfig",
    "Config",
    "get_config",
    "CONFIG_REGISTRY",
]
