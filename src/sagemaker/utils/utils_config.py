"""
Utility functions for the plastic bag detection pipeline.
"""

import os
import yaml
import json
import time
import math
import boto3
import argparse
import tarfile
from typing import Dict, Any, Tuple, List
from urllib.parse import urlparse
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

# initialize console
console = Console()


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        # Load .env if available to allow ${VAR} references inside YAML
        if load_dotenv is not None:
            try:
                load_dotenv()
            except Exception:
                pass

        with open(config_path, 'r') as file:
            raw_yaml = file.read()

        # Expand environment variables like ${VAR} within the YAML
        expanded_yaml = os.path.expandvars(raw_yaml)

        config = yaml.safe_load(expanded_yaml)
        
        if config is None:
            raise ValueError(f"Configuration file is empty: {config_path}")
            
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")


def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract data configuration section with defaults.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Data configuration with default values
    """
    return config.get('data', {})


def get_aws_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract AWS configuration section.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        AWS configuration dictionary
    """
    return config.get('aws', {})


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training configuration section.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Training configuration dictionary
    """
    return config.get('training', {})


def get_hyperparameters_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract hyperparameters configuration section.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Hyperparameters configuration dictionary
    """
    return config.get('training', {}).get('hyperparams', {})


def get_tuning_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract tuning configuration section.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Tuning configuration dictionary
    """
    return config.get('tuning', {})


def get_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract runtime configuration section.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Runtime configuration dictionary
    """
    return config.get('runtime', {})


def get_inference_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract inference configuration section.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Inference configuration dictionary
    """
    return config.get('inference', {})


def get_validation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract validation configuration section.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Validation configuration dictionary
    """
    return config.get('validation', {})



def load_training_config():
    """Load training configuration in SageMaker input directory"""
    
    # try to load config.yaml first
    config = None
    config_path = None
    
    # look for config in SageMaker input directory
    sagemaker_config_dir = os.environ.get("SM_CHANNEL_CONFIG", "/opt/ml/input/config")
    sagemaker_config_path = os.path.join(sagemaker_config_dir, "config.yaml")
    
    if os.path.exists(sagemaker_config_path):
        config_path = sagemaker_config_path
    elif os.path.exists("config.yaml"):
        config_path = "config.yaml"
    
    if config_path:
        try:
            config = load_config(config_path)
            print(f"Using config.yaml from: {config_path}")
            return config
        except Exception as e:
            print(f"Failed to load config.yaml: {e}!!!")

    return None


def load_registry_config():
    """Load registry configuration from config.yaml with fallbacks."""
    config = load_config()
    return config.get('registry', {})


def load_ground_truth_config():
    """Load ground truth configuration from config.yaml with fallbacks."""
    config = load_config()
    return config.get('ground_truth', {})



def get_inference_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get inference configuration"""
    return config.get('inference', {})



def get_deployment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get deployment configuration"""
    deployment = config.get('deployment', {})

    return {
        'job_name': deployment.get('job_name', {}),
        'lambda': deployment.get('lambda', {})
    }


def get_visualization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get visualization configuration"""
    return config.get('inference', {}).get('visualization', {})