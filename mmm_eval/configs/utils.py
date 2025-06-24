import json
from pathlib import Path
from typing import Any


def save_config(config: dict[str, Any], save_path: str, file_name: str) -> None:
    """Save a config to a JSON file.

    Args:
        config: The config to save.
        save_path: The path to save the config to.
        file_name: The name of the file to save the config to.

    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(f"{save_path}/{file_name}.json", "w") as f:
        json.dump(config, f)


def load_config(config_path: str) -> dict[str, Any]:
    """Load config from JSON file if provided.

    Args:
        config_path: The path to the config file.

    Returns:
        The config.

    """
    config_path_obj = validate_path(config_path)
    if not config_path_obj.suffix.lower() == ".json":
        raise ValueError(f"Invalid config path: {config_path}. Must be a JSON file.")
    with open(config_path_obj) as f:
        return json.load(f)


def validate_path(path: str) -> Path:
    """Validate path is a valid file path.

    Args:
        path: The path to validate.

    Returns:
        The validated path.

    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Invalid path:{path}")
    return path_obj
