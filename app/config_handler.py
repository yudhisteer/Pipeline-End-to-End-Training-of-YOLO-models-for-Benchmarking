"""Configuration file handler for YAML operations."""
try:
    from ruamel.yaml import YAML
    HAS_RUAMEL = True
except ImportError:
    import yaml
    HAS_RUAMEL = False
import os
from typing import Dict, Any


class ConfigHandler:
    """Handles loading and saving YAML configuration files with format preservation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        if HAS_RUAMEL:
            self.yaml_handler = YAML()
            self.yaml_handler.preserve_quotes = True
            self.yaml_handler.default_flow_style = False
        self.config = self.load_config()
        self.original_content = self._read_raw_content()
    
    def _read_raw_content(self) -> str:
        """Read the raw file content for line-by-line editing."""
        try:
            with open(self.config_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            return ""
    
    def load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration file."""
        try:
            with open(self.config_path, 'r') as file:
                if HAS_RUAMEL:
                    return self.yaml_handler.load(file) or {}
                else:
                    return yaml.safe_load(file) or {}
        except FileNotFoundError:
            return {}
    
    def save_config(self) -> None:
        """Save the current configuration to the YAML file."""
        if HAS_RUAMEL:
            # Use ruamel.yaml for format-preserving save
            with open(self.config_path, 'w') as file:
                self.yaml_handler.dump(self.config, file)
        else:
            # Fallback to standard yaml
            with open(self.config_path, 'w') as file:
                yaml.safe_dump(self.config, file, default_flow_style=False)
    
    def update_parameter_surgically(self, path: list, new_value: Any) -> bool:
        """
        Update a specific parameter while preserving file formatting.
        Returns True if successful, False if fallback to full rewrite needed.
        """
        if not HAS_RUAMEL:
            # Fallback to standard approach
            self.set_nested_value(self.config, path, new_value)
            self.save_config()
            return False
        
        try:
            # Update the in-memory config
            self.set_nested_value(self.config, path, new_value)
            
            # Use ruamel.yaml to preserve formatting
            self.save_config()
            return True
            
        except Exception:
            # If surgical update fails, fall back to standard approach
            self.set_nested_value(self.config, path, new_value)
            self.save_config()
            return False
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """Get a specific section from the config."""
        return self.config.get(section_name, {})
    
    def update_section(self, section_name: str, updates: Dict[str, Any]) -> None:
        """Update a specific section in the config."""
        if section_name not in self.config:
            self.config[section_name] = {}
        self.config[section_name].update(updates)
    
    def get_all_sections(self) -> list:
        """Get all available sections in the config."""
        return list(self.config.keys())
    
    def set_nested_value(self, data, path, value):
        """Set a value in nested dictionary/list structure using a path."""
        current = data
        for key in path[:-1]:
            if isinstance(current, list):
                current = current[int(key)]
            else:
                current = current[key]
        
        final_key = path[-1]
        if isinstance(current, list):
            current[int(final_key)] = value
        else:
            current[final_key] = value
