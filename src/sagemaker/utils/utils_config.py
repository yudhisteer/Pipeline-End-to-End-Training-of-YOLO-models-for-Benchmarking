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
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
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
    """Load training configuration with priority: config.yaml > command-line args > defaults."""
    
    # Try to load config.yaml first
    config = None
    config_path = None
    
    # Look for config in SageMaker input directory
    sagemaker_config_dir = os.environ.get("SM_CHANNEL_CONFIG", "/opt/ml/input/config")
    sagemaker_config_path = os.path.join(sagemaker_config_dir, "config.yaml")
    
    if os.path.exists(sagemaker_config_path):
        config_path = sagemaker_config_path
    elif os.path.exists("config.yaml"):
        config_path = "config.yaml"
    
    if config_path:
        try:
            config = load_config(config_path)
            print(f"✅ Using config.yaml from: {config_path}")
            return config
        except Exception as e:
            print(f"⚠️ Failed to load config.yaml: {e}")
    
    print("⚠️ No config.yaml found, using command-line arguments")
    return None


def load_inference_config():
    """Load inference configuration from config.yaml with fallbacks."""
    try:
        config = load_config()
        aws_config = get_aws_config(config)
        data_config = get_data_config(config)
        inference_config = get_inference_config(config)
        
        # Extract pipeline configuration - model package group is defined here to avoid duplication
        pipeline_config = config.get('pipeline', {})
        
        # Extract registry and cloudwatch configurations
        registry_config = inference_config.get('registry', {})
        cloudwatch_config = inference_config.get('cloudwatch', {})
        temp_files_config = registry_config.get('temp_files', {})
        
        return {
            'model_package_group': pipeline_config.get('model_package_group_name', 'YOLOModelPackageGroup'),
            'metric_key': inference_config.get('metric_key', 'recall'),
            'aws_region': aws_config.get('region'),
            'bucket': aws_config.get('bucket'),
            'prefix': aws_config.get('prefix'),
            'confidence_threshold': inference_config.get('confidence_threshold', 0.25),
            'iou_threshold': inference_config.get('iou_threshold', 0.45),
            'output_dir': inference_config.get('output_dir', './output'),
            'dataset_dir': data_config.get('dataset_dir', 'dataset/yolo_dataset'),
            'max_image_size': inference_config.get('max_image_size', 4096),
            'supported_formats': inference_config.get('supported_formats', 
                                                    ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']),
            'visualization': inference_config.get('visualization', {
                'show_confidence': True,
                'show_class_names': True,
                'box_thickness': 2,
                'font_scale': 0.6,
                'save_visualizations': True,
                'figure_size': [12, 8]
            }),
            # Registry configuration
            'registry': {
                'max_results': registry_config.get('max_results', 50),
                'max_results_display': registry_config.get('max_results_display', 20),
                'evaluation_metrics_file': registry_config.get('evaluation_metrics_file', 'evaluation_metrics.json'),
                'training_job_patterns': registry_config.get('training_job_patterns', ['YOLOTraining', 'pipelines-']),
                'temp_files': {
                    'output_archive': temp_files_config.get('output_archive', 'temp_output.tar.gz'),
                    'output_extract_dir': temp_files_config.get('output_extract_dir', 'temp_output_dir'),
                    'model_archive': temp_files_config.get('model_archive', 'model.tar.gz'),
                    'model_extract_dir': temp_files_config.get('model_extract_dir', 'model_dir')
                }
            },
            # CloudWatch configuration
            'cloudwatch': {
                'namespace': cloudwatch_config.get('namespace', 'YOLOTrainingMetrics')
            }
        }
    except Exception as e:
        print(f"Could not load config.yaml: {e}. Using defaults.")
        return {
            'model_package_group': 'YOLOModelPackageGroup',
            'metric_key': 'recall',
            'aws_region': None,
            'bucket': None,
            'prefix': None,
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45,
            'output_dir': './output',
            'dataset_dir': 'dataset/yolo_dataset',
            'max_image_size': 4096,
            'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
            'visualization': {
                'show_confidence': True,
                'show_class_names': True,
                'box_thickness': 2,
                'font_scale': 0.6,
                'save_visualizations': True,
                'figure_size': [12, 8]
            },
            'registry': {
                'max_results': 50,
                'max_results_display': 20,
                'evaluation_metrics_file': 'evaluation_metrics.json',
                'training_job_patterns': ['YOLOTraining', 'pipelines-'],
                'temp_files': {
                    'output_archive': 'temp_output.tar.gz',
                    'output_extract_dir': 'temp_output_dir',
                    'model_archive': 'model.tar.gz',
                    'model_extract_dir': 'model_dir'
                }
            },
            'cloudwatch': {
                'namespace': 'YOLOTrainingMetrics'
            }
        }


def load_ground_truth_config():
    """Load ground truth configuration from config.yaml with fallbacks."""
    config = load_config()
    return config.get('ground_truth', {})