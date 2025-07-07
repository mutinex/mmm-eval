import ast
import inspect
import re
from typing import Any

import numpy as np
import pymc_marketing.mmm as mmm
import pymc_marketing.prior as prior


class ConfigRehydrator:
    """Rehydrate a string config dictionary."""

    def __init__(self, config):
        """Initialize the ConfigRehydrator.

        Args:
            config: The config to rehydrate.

        """
        self.init_config = config.copy()
        self.hydrated_config = None
        self.class_registry = self.build_class_registry()

    def build_class_registry(self, *modules):
        """Build a registry of classes from the given modules.

        Args:
            *modules (list): The modules to build the registry from.

        Returns:
            dict: A dictionary of classes from the given modules.

        """
        registry = {}
        for mod in modules:
            registry.update(
                {
                    name: cls
                    for name, cls in inspect.getmembers(mod, inspect.isclass)
                    if cls.__module__.startswith(mod.__name__)
                }
            )
        return registry

    def fix_numpy_list_syntax(self, s: str) -> str:
        """Fix space-separated numbers inside brackets (NumPy-style) to comma-separated.

        e.g., "[1.0 2.0]" => "[1.0, 2.0]"

        """
        return re.sub(
            r"\[([\d\.\s\-eE]+)\]",
            lambda m: "[" + ", ".join(m.group(1).split()) + "]",
            s,
        )

    def safe_eval(self, value: str) -> Any:
        """Try literal_eval first, then fallback to eval with class registry.

        Args:
            value: The value to evaluate.

        Returns:
            The evaluated value.

        """
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            try:
                value = self.fix_numpy_list_syntax(value)
                return eval(value, {"__builtins__": {}}, self.class_registry)
            except Exception:
                return value  # leave it as-is if still not evaluable

    def rehydrate_config(self) -> dict[str, Any]:
        """Recursively rehydrate stringified config values.

        Returns
            The rehydrated config.

        """
        new_config = {}
        for key, val in self.init_config.items():
            if isinstance(val, str):
                new_config[key] = self.safe_eval(val)
            elif isinstance(val, dict):
                # Create a new instance of the same class to preserve class_registry
                nested_rehydrator = type(self)(val)
                new_config[key] = nested_rehydrator.rehydrate_config()
            else:
                new_config[key] = val
        self.hydrated_config = new_config
        return self.hydrated_config


class PyMCConfigRehydrator(ConfigRehydrator):
    """Rehydrate a config with PyMC objects."""

    def __init__(self, config):
        """Initialize the PyMCConfigRehydrator.

        Args:
            config: The config to rehydrate.

        """
        super().__init__(config)
        self.class_registry = self.build_class_registry(mmm, prior, np)


# TODO: implement
class MeridianConfigRehydrator(ConfigRehydrator):
    """Rehydrate a config with Meridian objects."""

    def __init__(self, config):
        """Initialize the MeridianConfigRehydrator.

        Args:
            config: The config to rehydrate.

        """
        super().__init__(config)
        # Import Meridian modules for class registry
        import meridian.model.prior_distribution as prior_distribution
        import meridian.model.spec as model_spec
        import numpy as np
        
        self.class_registry = self.build_class_registry(prior_distribution, model_spec, np)
