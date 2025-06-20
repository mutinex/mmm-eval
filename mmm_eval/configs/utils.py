from typing import Any, Optional, Dict
from pathlib import Path
import json

def save_config(config: dict[str, Any], save_path: str, file_name: str):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    json.dump(config, open(f"{save_path}/{file_name}.json", "w"))
    return config

def load_config(config_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load config from JSON file if provided."""
    if not config_path:
        return None
    config_path = validate_path(config_path)
    if not config_path.suffix.lower() == ".json":
        raise ValueError(f"Invalid config path: {config_path}. Must be a JSON file.")
    with open(config_path) as f:
        return json.load(f)
    
def validate_path(path: str) -> Path:
    """Validate path is a valid file path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Invalid path:{path}")
    return path