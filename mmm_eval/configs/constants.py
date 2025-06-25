"""Constants for the config."""


class ConfigConstants:
    """Constants for the config."""

    class ValidConfigExtensions:
        """Valid config extensions."""

        JSON = "json"

        @classmethod
        def all(cls):
            """Return list of all supported file extensions."""
            return [cls.JSON]

    class PyMCConfigAttributes:
        """Fields for the PyMC schema."""

        PYMC_MODEL_CONFIG = "pymc_model_config"
        FIT_CONFIG = "fit_config"
