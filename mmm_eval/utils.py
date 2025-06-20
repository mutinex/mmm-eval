import inspect
import re
import pymc_marketing.mmm as mmm
import pymc_marketing.prior as prior
from typing import Any
from mmm_eval.adapters.experimental.schemas import PyMCModelSchema, PyMCFitSchema

class ConfigRehydrator:
    """
    Rehydrate a string config dictionary with PyMC objects.
    """

    def __init__(self, config, schema_class: Any):
        self.config = config.copy()
        self.schema_class = schema_class

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

    def fix_missing_commas(self, val):
        """Add commas between floats in [...] brackets where missing.

        This is necessary because numpy arrays are not hydrated correctly without commas.

        Args:
            val (str): The string to fix.

        Returns:
            str: The fixed string.
        """
        if not isinstance(val, str):
            return val
        return re.sub(r"(?<=\d)\s+(?=\d)", ", ", val)

    def parse_constructor_string(self, constructor_str):
        """Parse a constructor string into a class name and kwargs.

        Args:
            constructor_str (str): The constructor string to parse.

        Raises:
            ValueError: If the constructor string is invalid.

        Returns:
            tuple: A tuple of the class name and kwargs.
        """
        match = re.match(r"^(\w+)\((.*)\)$", constructor_str.strip(), re.DOTALL)
        if not match:
            raise ValueError(f"Invalid string: {constructor_str}")

        class_name, args_str = match.groups()
        args_str = f"dict({args_str})"
        try:
            kwargs = eval(args_str, self.class_registry.copy(), {})
        except Exception as e:
            raise ValueError(
                f"Failed to parse args for {class_name}: {args_str}"
            ) from e

        return class_name, kwargs

    def rehydrate_value(self, val) -> Any:
        """Recursively rehydrate a value from a string.

        Args:
            val (Any): The value to rehydrate.

        Returns:
            Any: The rehydrated value.
        """
        # If it's a constructor-style string
        if isinstance(val, str) and "(" in val and ")" in val:
            try:
                # Try to evaluate directly in registry context
                return eval(
                    self.fix_missing_commas(val), self.class_registry.copy(), {}
                )
            except Exception:
                return f"Unable to rehydrate {val} as a python object."

        # Recurse into dicts
        elif isinstance(val, dict):
            return {k: self.rehydrate_value(v) for k, v in val.items()}

        # Recurse into lists
        elif isinstance(val, list):
            return [self.rehydrate_value(v) for v in val]

        # Base case
        return val

    def rehydrate_config(self):
        """Rehydrate the config dictionary.

        Returns:
            dict: The rehydrated config dictionary.
        """
        return self.schema_class.model_validate(self.rehydrate_value(self.config)).model_dump()


class PyMCConfigRehydrator(ConfigRehydrator):
    def __init__(self, config, schema_class: PyMCModelSchema | PyMCFitSchema):
        super().__init__(config, schema_class)
        self.class_registry = self.build_class_registry(mmm, prior)
