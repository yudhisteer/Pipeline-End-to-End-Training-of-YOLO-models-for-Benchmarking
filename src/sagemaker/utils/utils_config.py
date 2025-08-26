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
            'job_name': inference_config.get('job_name'),  # Add the job_name field
            'metric_key': inference_config.get('metric_key', 'recall'),
            'aws_region': aws_config.get('region'),
            'bucket': aws_config.get('bucket'),
            'prefix': aws_config.get('prefix'),
            'confidence_threshold': inference_config.get('confidence_threshold', 0.25),
            'iou_threshold': inference_config.get('iou_threshold', 0.45),
            'output_dir': inference_config.get('output_dir', './output'),
            'dataset_dir': data_config.get('dataset_dir', 'dataset/yolo-dataset'),
            'image_path': inference_config.get('test', {}).get('image_path', 'val/images/000000000885.jpg'),
            's3_bucket': inference_config.get('test', {}).get('s3_bucket', 'cyudhist-pipeline-yolo-503561429929'),
            's3_prefix': inference_config.get('test', {}).get('s3_bucket', 'yolo-pipeline/yolo-dataset/val/images'),
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
        raise Exception(f"Could not load config.yaml: {e}")


def load_ground_truth_config():
    """Load ground truth configuration from config.yaml with fallbacks."""
    config = load_config()
    return config.get('ground_truth', {})


def get_lambda_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get Lambda configuration from config.yaml"""
    
    # Extract Lambda configurations
    deployment_lambda = config.get('deployment', {}).get('lambda', {})
    inference_lambda = config.get('inference', {}).get('lambda', {})
    
    # Get environment variables from deployment (single source of truth)
    environment = deployment_lambda.get('environment', {})
    
    # Base Lambda configuration from deployment section
    base_config = {
        'function_name': deployment_lambda.get('function_name'),
        'stack_name': deployment_lambda.get('stack_name'),
        'memory_size': deployment_lambda.get('memory_size'),
        'timeout': deployment_lambda.get('timeout'),
        'architecture': deployment_lambda.get('architecture'),
        'package_type': deployment_lambda.get('package_type'),
        'api_gateway': deployment_lambda.get('api_gateway', {}),
        's3_bucket_access': deployment_lambda.get('s3_bucket_access', []),
        
        # Model configuration from environment variables
        'model_s3_bucket': environment.get('MODEL_S3_BUCKET'),
        'model_s3_key': environment.get('MODEL_S3_KEY'),
        'model_input_height': int(environment.get('MODEL_INPUT_HEIGHT', 640)),
        'model_input_width': int(environment.get('MODEL_INPUT_WIDTH', 640)),
        'max_detections': int(environment.get('MAX_DETECTIONS', 300)),
    }
    
    # Runtime configuration from inference section
    runtime_config = {
        'base_url': inference_lambda.get('base_url'),
        'local_base_url': inference_lambda.get('local_base_url'),
        'endpoints': inference_lambda.get('endpoints', {}),
        'request_timeout': inference_lambda.get('request_timeout'),
        'max_retries': inference_lambda.get('max_retries'),
        'retry_delay': inference_lambda.get('retry_delay'),
        'warm_up_requests': inference_lambda.get('warm_up_requests'),
        'batch_processing': inference_lambda.get('batch_processing'),
        'enable_cloudwatch_logs': inference_lambda.get('enable_cloudwatch_logs'),
        'log_level': inference_lambda.get('log_level'),
        'metrics_enabled': inference_lambda.get('metrics_enabled'),
    }
    
    # Inference parameters (with environment fallback)
    inference_params = {
        'confidence_threshold': config.get('inference', {}).get('confidence_threshold', float(environment.get('CONFIDENCE_THRESHOLD', 0.3))),
        'iou_threshold': config.get('inference', {}).get('iou_threshold', float(environment.get('IOU_THRESHOLD', 0.5))),
        'max_image_size': config.get('inference', {}).get('max_image_size', 640),
        'supported_formats': config.get('inference', {}).get('supported_formats', []),
        'output_dir': config.get('inference', {}).get('output_dir'),
    }
    
    # Test configuration
    test_config = {
        'image_path': config.get('inference', {}).get('test', {}).get('image_path'),
    }
    
    # Merge all configurations
    lambda_config = {**base_config, **runtime_config, **inference_params, **test_config}
    
    return lambda_config


def get_deployment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get deployment configuration"""
    deployment = config.get('deployment', {})
    
    return {
        'mode': deployment.get('mode'),
        'enabled': deployment.get('enabled'),
        'auto_delete': deployment.get('auto_delete'),
        'endpoint_name': deployment.get('endpoint_name'),
        'model_package_name': deployment.get('model_package_name'),
        'lambda': deployment.get('lambda', {}),
        'sagemaker': deployment.get('sagemaker', {})
    }


def get_visualization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get visualization configuration"""
    return config.get('inference', {}).get('visualization', {})