import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field, PlainValidator, ValidationInfo

from mmm_eval.configs.constants import ConfigConstants
from mmm_eval.configs.exceptions import InvalidConfigFormatError


def validate_response_column(v: str | None, info: ValidationInfo) -> str:
    """Validate and set response column default.

    Response column is optionally null. This function maps the revenue column
    to the response column if the response column is not provided.

    Args:
        v: The value to validate
        info: The validation info

    Returns:
        The validated value

    """
    if v is None:
        return info.data.get("revenue_column")
    return v


class BaseConfig(BaseModel, ABC):
    """Abstract base class for all framework configs.

    This class provides common functionality for config validation,
    serialization, and deserialization across different MMM frameworks.
    """

    # Common fields between all configs
    revenue_column: str = Field(..., description="Column containing the revenue variable")
    response_column: Annotated[str, PlainValidator(validate_response_column)] = Field(
        None, description="Column containing the response variable"
    )

    # Abstract methods that must be implemented by all configs
    @abstractmethod
    def save_model_object_to_json(self, save_path: str, file_name: str) -> "BaseConfig":
        """Save the config to a JSON file.

        Args:
            save_path: Directory to save the config to
            file_name: Name of the file (without extension)

        Returns:
            Self for method chaining

        """
        pass

    @classmethod
    @abstractmethod
    def load_model_config_from_json(cls, config_path: str) -> "BaseConfig":
        """Load a config from a JSON file.

        Args:
            config_path: Path to the JSON config file

        Returns:
            A config instance

        """
        pass

    @property
    @abstractmethod
    def date_column(self) -> str:
        """Return the date column."""
        pass

    @property
    @abstractmethod
    def channel_columns(self) -> list[str]:
        """Return the channel columns."""
        pass

    @property
    @abstractmethod
    def control_columns(self) -> list[str] | None:
        """Return the control columns."""
        pass

    # Common methods for all configs
    @classmethod
    def _validate_config_path(cls, config_path: str) -> Path:
        """Validate that the config path exists and is a JSON file.

        Args:
            config_path: Path to validate

        Returns:
            Path object if valid

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a JSON file

        """
        config_path_obj = Path(config_path)

        if not config_path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not config_path_obj.is_file():
            raise InvalidConfigFormatError(f"Config file is not a file: {config_path}")
        if config_path_obj.suffix.lower().lstrip(".") not in ConfigConstants.ValidConfigExtensions.all():
            raise InvalidConfigFormatError(
                f"Config file must be one of the following extensions: {ConfigConstants.ValidConfigExtensions.all()}"
            )

        return config_path_obj

    @classmethod
    def _load_json_file(cls, config_path: str) -> dict[str, Any]:
        """Load and parse a JSON file.

        Args:
            config_path: Path to the JSON file

        Returns:
            Parsed JSON data as a dictionary

        """
        config_path_obj = cls._validate_config_path(config_path)
        with open(config_path_obj) as f:
            return json.load(f)

    @classmethod
    def _save_json_file(cls, save_path: str, file_name: str, config_dict: dict[str, Any]) -> Path:
        """Save a dictionary to a JSON file.

        Args:
            save_path: Directory to save to
            file_name: Name of the file (without extension)
            config_dict: Dictionary to save

        Returns:
            Path to the saved file

        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        file_path = Path(save_path) / f"{file_name}.{ConfigConstants.ValidConfigExtensions.JSON}"
        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        return file_path
