"""
Pytest configuration for mmm-eval tests.
"""

import logging
import pytest


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for all tests."""
    # Configure logging to show INFO level messages in terminal
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Override any existing configuration
    )
    
    # Set specific loggers to INFO level for mmm_eval modules
    mmm_eval_loggers = [
        'mmm_eval.core.validation_tests',
        'mmm_eval.core.validation_test_orchestrator',
        'mmm_eval.core.evaluator',
        'mmm_eval.adapters',
        'mmm_eval.cli.evaluate',
        'mmm_eval.core.base_validation_test',
    ]
    
    for logger_name in mmm_eval_loggers:
        logging.getLogger(logger_name).setLevel(logging.INFO)
    
    # Set PyMC loggers to ERROR level only (suppress INFO and WARNING)
    pymc_loggers = [
        'pymc',
        'pymc.sampling',
        'pymc.sampling.mcmc',
        'pymc.stats',
        'pymc.stats.convergence',
        'pymc_marketing',
    ]
    
    for logger_name in pymc_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR) 