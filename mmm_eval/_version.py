"""Version information for mmm-eval package."""

import tomllib
from pathlib import Path


def _get_version_from_pyproject() -> str:
    """Extract version from pyproject.toml file."""
    # Get the directory containing this file
    current_dir = Path(__file__).parent
    # Go up one level to find pyproject.toml
    pyproject_path = current_dir.parent / "pyproject.toml"
    
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")
    
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    # Extract version from poetry section
    if "tool" in data and "poetry" in data["tool"]:
        version = data["tool"]["poetry"].get("version")
        if version:
            return version
    
    raise ValueError("Version not found in pyproject.toml")


# Get the version
try:
    __version__ = _get_version_from_pyproject()
except (FileNotFoundError, ValueError) as e:
    # Fallback version if pyproject.toml cannot be read
    __version__ = "0.0.0"
    import warnings
    warnings.warn(f"Could not read version from pyproject.toml: {e}. Using fallback version {__version__}",
                  stacklevel=2) 