import pandas as pd

from mmm_eval.configs.base import BaseConfig
from mmm_eval.core.evaluator import Evaluator

# from mmm_eval.core.validation_tests_models import SupportedFrameworks  # Removed to fix circular import
from mmm_eval.data.pipeline import DataPipeline


def run_evaluation(
    framework: str,
    data: pd.DataFrame,
    config: BaseConfig,
    test_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Evaluate an MMM framework.

    Args:
        framework: The framework to evaluate.
        data: The data to evaluate.
        config: The config to use for the evaluation.
        test_names: The tests to run. If not provided, all tests will be run.

    Returns:
        A pandas DataFrame containing the evaluation results.

    """
    # validate + process the input data
    data = DataPipeline(
        data=data,
        framework=framework,
        date_column=config.date_column,
        response_column=config.response_column,
        revenue_column=config.revenue_column,
        control_columns=config.control_columns,
        channel_columns=config.channel_columns,
    ).run()

    # run the evaluation suite
    results = Evaluator(
        data=data,
        test_names=test_names,
    ).evaluate_framework(framework=framework, config=config)

    return results.to_df()
