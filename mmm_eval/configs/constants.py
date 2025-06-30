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

    class PyMCIntersectingFitPredictKwargs:
        """Kwargs for the intersect of PyMC fit and predict schemas."""

        PROGRESS_BAR = "progressbar"
        RANDOM_SEED = "random_seed"

        @classmethod
        def all(cls):
            """Return list of all valid kwargs."""
            return [cls.PROGRESS_BAR, cls.RANDOM_SEED]

    class MeridianConfigAttributes:
        """Fields for the Meridian schema."""

        MERIDIAN_MODEL_CONFIG = "meridian_model_config"
        MODEL_SPEC_CONFIG = "model_spec_config"
        FIT_CONFIG = "fit_config"

    class MeridianIntersectingFitPredictKwargs:
        """Kwargs for the intersect of Meridian fit and predict schemas."""

        PROGRESS_BAR = "progress_bar"
        RANDOM_SEED = "random_seed"

        @classmethod
        def all(cls):
            """Return list of all valid kwargs."""
            return [cls.PROGRESS_BAR, cls.RANDOM_SEED]
