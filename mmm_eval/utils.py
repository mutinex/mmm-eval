"""Utility functions for MMM evaluation."""

import inspect
import re
from typing import Any

import pymc_marketing.mmm as mmm
import pymc_marketing.prior as prior


class ConfigRehydrator:
    """Rehydrate a string config dictionary with PyMC objects."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the config rehydrator.

        Args:
            config: Configuration dictionary to rehydrate

        """
        self.config = config.copy()
        self.class_registry: dict[str, Any] = {}

    def build_class_registry(self, *modules) -> dict[str, Any]:
        """Build a registry of classes from modules.

        Args:
            *modules: Modules to scan for classes

        Returns:
            Dictionary mapping class names to classes

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

    def fix_missing_commas(self, val: Any) -> Any:
        """Fix missing commas in string representations.

        Args:
            val: Value to fix

        Returns:
            Fixed value

        """
        if not isinstance(val, str):
            return val
        # Add commas between floats in [...] brackets where missing
        return re.sub(r"(?<=\d)\s+(?=\d)", ", ", val)

    def parse_constructor_string(self, constructor_str: str) -> tuple[str, dict[str, Any]]:
        """Parse a constructor string into class name and arguments.

        Args:
            constructor_str: String representation of constructor

        Returns:
            Tuple of (class_name, kwargs)

        Raises:
            ValueError: If parsing fails

        """
        match = re.match(r"^(\w+)\((.*)\)$", constructor_str.strip(), re.DOTALL)
        if not match:
            raise ValueError(f"Invalid constructor string: {constructor_str}")

        class_name, args_str = match.groups()
        args_str = f"dict({args_str})"
        try:
            kwargs = eval(args_str, self.class_registry.copy(), {})
        except Exception as e:
            raise ValueError(f"Failed to parse args for {class_name}: {args_str}") from e

        return class_name, kwargs

    def rehydrate_value(self, val: Any) -> Any:
        """Rehydrate a value by converting strings to objects.

        Args:
            val: Value to rehydrate

        Returns:
            Rehydrated value

        """
        # If it's a constructor-style string
        if isinstance(val, str) and "(" in val and ")" in val:
            try:
                # Try to evaluate directly in registry context
                return eval(self.fix_missing_commas(val), self.class_registry.copy(), {})
            except Exception:
                return val  # Not evaluatable

        # Recurse into dicts
        elif isinstance(val, dict):
            return {k: self.rehydrate_value(v) for k, v in val.items()}

        # Recurse into lists
        elif isinstance(val, list):
            return [self.rehydrate_value(v) for v in val]

        # Base case
        return val

    def rehydrate_config(self) -> dict[str, Any]:
        """Rehydrate the entire configuration.

        Returns
            Rehydrated configuration dictionary

        """
        return self.rehydrate_value(self.config)


class PyMCConfigRehydrator(ConfigRehydrator):
    """PyMC-specific configuration rehydrator."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the PyMC config rehydrator.

        Args:
            config: Configuration dictionary to rehydrate

        """
        super().__init__(config)
        self.class_registry = self.build_class_registry(mmm, prior)
