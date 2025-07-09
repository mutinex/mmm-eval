import ast
import inspect
import re
from typing import Any

import numpy as np
import pymc_marketing.mmm as mmm
import pymc_marketing.prior as prior
import tensorflow_probability as tfp
from meridian.model.prior_distribution import PriorDistribution


def deserialize_tfp_distribution(serialized_dist: dict[str, Any]) -> Any:
    """Deserialize a TFP distribution or bijector from the serialized format.

    This function reconstructs TensorFlow Probability distributions and bijectors
    from their serialized dictionary representation. It handles both distributions
    and bijectors, and recursively reconstructs any nested TFP objects in the
    parameters.

    Args:
        serialized_dist: A dictionary containing "dist_type" and "parameters" keys,
                        representing a serialized TFP distribution or bijector.

    Returns:
        A reconstructed TFP distribution or bijector object.

    Raises:
        AttributeError: If the type is not found in either tfp.distributions
                       or tfp.bijectors.

    Example:
        >>> serialized = {
        ...     "dist_type": "Normal",
        ...     "parameters": {"loc": 0.0, "scale": 1.0}
        ... }
        >>> dist = deserialize_tfp_distribution(serialized)
        >>> # Returns: <tfp.distributions.Normal object>

    """
    dist_type = serialized_dist["dist_type"]
    parameters = serialized_dist["parameters"]

    # Recursively reconstruct parameters
    def reconstruct_param(val: Any) -> Any:
        if isinstance(val, dict) and "dist_type" in val and "parameters" in val:
            return deserialize_tfp_distribution(val)
        elif isinstance(val, list):
            return [reconstruct_param(v) for v in val]
        elif isinstance(val, dict):
            return {k: reconstruct_param(v) for k, v in val.items()}
        else:
            return val

    reconstructed_params = {k: reconstruct_param(v) for k, v in parameters.items()}

    # Try to get from distributions, then bijectors
    if hasattr(tfp.distributions, dist_type):
        dist_class = getattr(tfp.distributions, dist_type)
    elif hasattr(tfp.bijectors, dist_type):
        dist_class = getattr(tfp.bijectors, dist_type)
    else:
        raise AttributeError(f"Neither tfp.distributions nor tfp.bijectors has attribute '{dist_type}'")

    return dist_class(**reconstructed_params)


def deserialize_prior_distribution(serialized_prior: dict[str, Any]) -> PriorDistribution:
    """Deserialize a PriorDistribution from the serialized format.

    This function reconstructs a PriorDistribution object from its serialized
    dictionary representation. It handles the conversion of serialized TFP
    distributions back to their original objects.

    Args:
        serialized_prior: A dictionary containing the serialized PriorDistribution
                         attributes, where TFP distributions are represented as
                         dictionaries with "dist_type" and "parameters" keys.

    Returns:
        A reconstructed PriorDistribution object.

    Example:
        >>> serialized = {
        ...     "roi_m": {
        ...         "dist_type": "LogNormal",
        ...         "parameters": {"loc": 0.2, "scale": 0.9}
        ...     }
        ... }
        >>> prior = deserialize_prior_distribution(serialized)
        >>> # Returns: <PriorDistribution object with roi_m as LogNormal>

    """
    # Convert serialized TFP distributions back to objects
    deserialized_prior = {}
    for key, value in serialized_prior.items():
        if isinstance(value, dict) and "dist_type" in value and "parameters" in value:
            # It's a serialized TFP distribution
            deserialized_prior[key] = deserialize_tfp_distribution(value)
        else:
            # It's a regular value
            deserialized_prior[key] = value

    return PriorDistribution(**deserialized_prior)


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


class MeridianConfigRehydrator(ConfigRehydrator):
    """Rehydrate a config with Meridian objects."""

    def __init__(self, config):
        """Initialize the MeridianConfigRehydrator.

        Args:
            config: The config to rehydrate.

        """
        super().__init__(config)
        import meridian.model.prior_distribution as prior_distribution
        import meridian.model.spec as model_spec

        self.class_registry = self.build_class_registry(prior_distribution, model_spec, np, tfp.distributions)
        self.class_registry["PriorDistribution"] = PriorDistribution
        self.class_registry["tfp"] = tfp
        self.PriorDistribution = PriorDistribution

    def rehydrate_config(self) -> dict[str, Any]:
        """Recursively rehydrate stringified config values with special handling for prior field.

        This method extends the base rehydration logic to handle the special case
        of the 'prior' field, which can be a PriorDistribution object, a dict
        in the new serialization format, or a string representation.

        Returns
            The rehydrated config.

        """

        def recursively_eval_dict(d):
            hydrated = {}
            for k, v in d.items():
                if isinstance(v, str):
                    evaluated = self.safe_eval(v)
                    if isinstance(evaluated, dict):
                        hydrated[k] = recursively_eval_dict(evaluated)
                    else:
                        hydrated[k] = evaluated
                else:
                    hydrated[k] = v
            return hydrated

        new_config = {}
        for key, val in self.init_config.items():
            if key == "prior":
                if isinstance(val, self.PriorDistribution):
                    new_config[key] = val
                elif isinstance(val, dict):
                    # Check if this is our new serialization format
                    if any(isinstance(v, dict) and "dist_type" in v and "parameters" in v for v in val.values()):
                        new_config[key] = deserialize_prior_distribution(val)
                    else:
                        hydrated_prior_dict = recursively_eval_dict(val)
                        new_config[key] = self.PriorDistribution(**hydrated_prior_dict)
                elif isinstance(val, str):
                    evaluated = self.safe_eval(val)
                    if isinstance(evaluated, self.PriorDistribution):
                        new_config[key] = evaluated
                    elif isinstance(evaluated, dict):
                        # Check if this is our new serialization format
                        if any(isinstance(v, dict) and "dist_type" in v and "parameters" in v for v in evaluated.values()):
                            new_config[key] = deserialize_prior_distribution(evaluated)
                        else:
                            hydrated_prior_dict = recursively_eval_dict(evaluated)
                            new_config[key] = self.PriorDistribution(**hydrated_prior_dict)
                    else:
                        new_config[key] = evaluated
                else:
                    new_config[key] = val
            elif isinstance(val, str):
                new_config[key] = self.safe_eval(val)
            elif isinstance(val, dict):
                nested_rehydrator = type(self)(val)
                new_config[key] = nested_rehydrator.rehydrate_config()
            else:
                new_config[key] = val
        self.hydrated_config = new_config
        return self.hydrated_config
