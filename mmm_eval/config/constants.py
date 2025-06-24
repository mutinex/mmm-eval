class ConfigConstants:
    """Constants for the config."""

    class ValidConfigExtensions:
        """Valid config extensions."""

        JSON = "json"

        @classmethod
        def all(cls):
            """Return list of all supported file extensions."""
            return [cls.JSON]
