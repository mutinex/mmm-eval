import pytest
from mmm_eval.adapters.pymc import PyMCAdapter
from tests.test_utils import valid_config_1, valid_config_2


def invalid_config():
    return {
        "date_column": "date_week",
        "channel_columns": ["channel_1", "channel_2"],
        "response_column": "quantity",
        "control_columns": ["price", "event_1", "event_2"],
        "adstock": "NotARealAdstock",
    }


@pytest.mark.parametrize("config, is_valid", [
    (valid_config_1(), True),
    (valid_config_2(), True),
    (invalid_config(), False)
])
def test_adapter_instantiation(config, is_valid):
    # Instantiate the adapter

    if not is_valid:
        with pytest.raises(ValueError):
            PyMCAdapter(config)
    else:
        adapter = PyMCAdapter(config)
        assert adapter is not None
        assert adapter.config is not None

if __name__ == "__main__":
    pytest.main()