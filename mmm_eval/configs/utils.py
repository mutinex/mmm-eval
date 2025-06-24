from pathlib import Path


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
