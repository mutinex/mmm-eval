"""Utility functions for MMM evaluation."""

import inspect
import re
from typing import Any

import pymc_marketing.mmm as mmm
import pymc_marketing.prior as prior
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior


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


class ConfigBuilder:
    """Builder class for creating PyMC configuration dictionaries."""

    def __init__(self):
        """Initialize the config builder."""
        self.config = {}

    def add_date_column(self, column: str) -> "ConfigBuilder":
        """Add date column to configuration.

        Args:
            column: Name of date column

        Returns:
            Self for method chaining

        """
        self.config["date_column"] = column
        return self

    def add_channel_columns(self, columns: list[str]) -> "ConfigBuilder":
        """Add channel columns to configuration.

        Args:
            columns: List of channel column names

        Returns:
            Self for method chaining

        """
        self.config["channel_columns"] = columns
        return self

    def add_response_column(self, column: str) -> "ConfigBuilder":
        """Add response column to configuration.

        Args:
            column: Name of response column

        Returns:
            Self for method chaining

        """
        self.config["response_column"] = column
        return self

    def add_control_columns(self, columns: list[str]) -> "ConfigBuilder":
        """Add control columns to configuration.

        Args:
            columns: List of control column names

        Returns:
            Self for method chaining

        """
        self.config["control_columns"] = columns
        return self

    def add_revenue_column(self, column: str) -> "ConfigBuilder":
        """Add revenue column to configuration.

        Args:
            column: Name of revenue column

        Returns:
            Self for method chaining

        """
        self.config["revenue_column"] = column
        return self

    def add_adstock(self, adstock: GeometricAdstock) -> "ConfigBuilder":
        """Add adstock component to configuration.

        Args:
            adstock: Adstock component

        Returns:
            Self for method chaining

        """
        self.config["adstock"] = adstock
        return self

    def add_saturation(self, saturation: LogisticSaturation) -> "ConfigBuilder":
        """Add saturation component to configuration.

        Args:
            saturation: Saturation component

        Returns:
            Self for method chaining

        """
        self.config["saturation"] = saturation
        return self

    def add_model_config(self, model_config: dict[str, Prior]) -> "ConfigBuilder":
        """Add model configuration to configuration.

        Args:
            model_config: Model configuration dictionary

        Returns:
            Self for method chaining

        """
        self.config["model_config"] = model_config
        return self

    def build(self) -> dict[str, Any]:
        """Build the final configuration dictionary.

        Returns
            Complete configuration dictionary

        """
        return self.config.copy()


def create_default_pymc_config() -> dict[str, Any]:
    """Create a default PyMC configuration.

    Returns
        Default configuration dictionary

    """
    return (
        ConfigBuilder()
        .add_date_column("date_week")
        .add_channel_columns(["channel_1", "channel_2"])
        .add_response_column("quantity")
        .add_control_columns(["price", "event_1", "event_2"])
        .add_revenue_column("revenue")
        .add_adstock(GeometricAdstock(l_max=4))
        .add_saturation(LogisticSaturation())
        .add_model_config(
            {
                "intercept": Prior("Normal", mu=0.5, sigma=0.2),
                "saturation_beta": Prior("HalfNormal", sigma=[0.321, 0.123]),
                "gamma_control": Prior("Normal", mu=0, sigma=0.05),
                "gamma_fourier": Prior("Laplace", mu=0, b=0.2),
            }
        )
        .build()
    )
