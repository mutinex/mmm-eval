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