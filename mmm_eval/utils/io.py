"""I/O utilities for mmm-eval."""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_results(results: pd.DataFrame, framework: str, output_path: str) -> None:
    """Save the results to a CSV file.

    Args:
        results: The dataframe of results to save.
        framework: The name of the framework that was evaluated.
        output_path: The path to save the results to.

    """
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mmm_eval_{framework}_{timestamp}.csv"

    results.to_csv(output_path_obj / filename, index=False)
    logger.info(f"Saved results to {output_path_obj / filename}")
