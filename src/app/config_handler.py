"""Configuration file handler for YAML operations."""
import yaml
import os
from typing import Dict, Any


class ConfigHandler:
    """Handles loading and saving YAML configuration files."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file) or {}
        except FileNotFoundError:
            return {}
    
    def save_config(self) -> None:
        """Save the current configuration to the YAML file."""
        with open(self.config_path, 'w') as file:
            yaml.safe_dump(self.config, file, default_flow_style=False)
    
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
