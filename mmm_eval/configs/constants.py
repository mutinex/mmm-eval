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

        INPUT_DATA_BUILDER_CONFIG = "input_data_builder_config"
        MODEL_SPEC_CONFIG = "model_spec_config"
        SAMPLE_POSTERIOR_CONFIG = "sample_posterior_config"
