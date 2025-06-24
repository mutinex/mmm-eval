from pathlib import Path
from typing import Any
import json

from mmm_eval.config.config_models import EvaluatorConfig
from mmm_eval.config.constants import ConfigConstants
from mmm_eval.data.constants import InputDataframeConstants


class ConfigLoader:
    """Loads and separates configuration for evaluator and framework components."""
    
    def __init__(self, config_path: str):
        """Initialize with path to JSON config file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        self.config_path = self._validate_path(config_path)
        self.raw_config = self._load_json_config()
        
        # Load and separate configurations
        self.internal_evaluator_config = self._extract_evaluator_config()
        self.framework_config = self._extract_framework_config()
    
    def _extract_evaluator_config(self) -> EvaluatorConfig:
        """Extract evaluator-specific configuration.
        
        Returns:
            Validated evaluator configuration
        """
        return EvaluatorConfig(
            response_column=self.raw_config.get("response_column", InputDataframeConstants.RESPONSE_COL),
            revenue_column=self.raw_config.get("revenue_column", InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL),
        )
    
    def _extract_framework_config(self) -> dict[str, Any]:
        """Extract framework-specific configuration.
        
        Returns:
            Framework configuration dictionary
        """
        evaluator_field_names = set(EvaluatorConfig.model_fields.keys())
        return {
            field_name: field_value 
            for field_name, field_value in self.raw_config.items()
            if field_name not in evaluator_field_names
        }
    
    def _validate_path(self, config_path: str) -> Path:
        """Validate config file path exists and is JSON.
        
        Args:
            config_path: Path to validate
            
        Returns:
            Validated Path object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not JSON
        """
        path_object = Path(config_path)
        if not path_object.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if path_object.suffix.lower().lstrip(".") not in ConfigConstants.ValidConfigExtensions.all():
            raise ValueError(f"Config file must be in {ConfigConstants.ValidConfigExtensions.all()}, got: {path_object.suffix}")
        
        return path_object
    
    def _load_json_config(self) -> dict[str, Any]:
        """Load JSON configuration from file.
        
        Returns:
            Raw configuration dictionary
            
        Raises:
            JSONDecodeError: If JSON is malformed
        """
        with open(self.config_path, "r") as config_file:
            return json.load(config_file)