"""Logging utilities for mmm-eval."""

import logging


def setup_logging():
    """Set up logging configuration for mmm-eval.

    This function configures logging to show INFO level messages for mmm_eval modules
    and suppress PyMC logging to ERROR level only. It's safe to call multiple times.
    """
    # Only setup if not already configured
    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Set PyMC loggers to ERROR level only (suppress INFO and WARNING)
    pymc_loggers = [
        "pymc",
        "pymc.sampling",
        "pymc.sampling.mcmc",
        "pymc.stats",
        "pymc.stats.convergence",
        "pymc_marketing",
    ]
    for logger_name in pymc_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
