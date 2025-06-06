"""
Configuration Management

Handles loading and merging configuration files for the evaluation framework.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from copy import deepcopy


class ConfigManager:
    """Manages configuration loading and merging."""
    
    def __init__(self, base_config_path: Optional[str] = None):
        """Initialize configuration manager."""
        if base_config_path is None:
            base_config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"
        
        self.base_config = self._load_config_file(base_config_path)
        self.current_config = deepcopy(self.base_config)
    
    def _load_config_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def load_custom_config(self, config_path: str):
        """Load and merge custom configuration."""
        custom_config = self._load_config_file(config_path)
        self.current_config = self._deep_merge(self.current_config, custom_config)
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with dictionary."""
        self.current_config = self._deep_merge(self.current_config, updates)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Example:
            config.get('evaluators.semantic.config.batch_size')
        """
        keys = key_path.split('.')
        value = self.current_config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Example:
            config.set('evaluators.semantic.config.batch_size', 10)
        """
        keys = key_path.split('.')
        target = self.current_config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        # Set the value
        target[keys[-1]] = value
    
    def get_evaluator_config(self, evaluator_name: str) -> Dict[str, Any]:
        """Get configuration for specific evaluator."""
        return self.get(f'evaluators.{evaluator_name}.config', {})
    
    def is_evaluator_enabled(self, evaluator_name: str) -> bool:
        """Check if evaluator is enabled."""
        return self.get(f'evaluators.{evaluator_name}.enabled', False)
    
    def get_enabled_evaluators(self) -> List[str]:
        """Get list of enabled evaluators."""
        evaluators = self.get('evaluators', {})
        return [name for name, config in evaluators.items() 
                if config.get('enabled', False)]
    
    def get_visualization_config(self, evaluator_name: str) -> List[Dict[str, Any]]:
        """Get visualization configuration for evaluator."""
        return self.get(f'visualizations.{evaluator_name}', [])
    
    def save_config(self, output_path: str):
        """Save current configuration to file."""
        output_path = Path(output_path)
        
        if output_path.suffix in ['.yaml', '.yml']:
            with open(output_path, 'w') as f:
                yaml.dump(self.current_config, f, default_flow_style=False)
        elif output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(self.current_config, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = deepcopy(base)
        
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Get current configuration as dictionary."""
        return deepcopy(self.current_config)