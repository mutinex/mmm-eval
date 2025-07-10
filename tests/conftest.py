"""Pytest configuration for mmm-eval tests."""

import logging

import pytest


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for all tests."""
    # Configure logging to show INFO level messages in terminal
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Override any existing configuration
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
