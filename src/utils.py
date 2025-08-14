"""
Utility functions for the plastic bag detection pipeline.
"""

import os
import yaml
import json
import time
import boto3
import argparse
from typing import Dict, Any


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
    return config.get('hyperparameters', {})


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


def generate_model_metrics(results, model_dir: str, config: Dict[str, Any] = None):
    """
    Generate validation metrics for SageMaker Model Registry.
    
    Args:
        results: YOLO training results
        model_dir: Directory to save metrics files
        config: Configuration dictionary (optional, will load default if None)
    """
    print("Generating model metrics for SageMaker Model Registry...")
    
    # Load config if not provided
    if config is None:
        try:
            config = load_config()
        except:
            config = {}
    
    # Get validation thresholds from config
    validation_config = get_validation_config(config)
    
    try:
        # Extract metrics from YOLO results
        # YOLO results typically contain validation metrics in the last epoch
        if hasattr(results, 'results_dict'):
            metrics_dict = results.results_dict
        else:
            # Fallback: extract from results object
            metrics_dict = {}
            if hasattr(results, 'metrics'):
                metrics_dict = results.metrics
    
        # YOLO-specific validation metrics for object detection
        validation_metrics = {
            "model_name": "YOLO",
            "validation_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "metrics": {
                # Primary YOLO metrics
                "mAP_0.5": float(getattr(results, 'maps', [0.0])[0]) if hasattr(results, 'maps') else 0.0,
                "mAP_0.5_0.95": float(getattr(results, 'map', 0.0)) if hasattr(results, 'map') else 0.0,
                "precision": float(getattr(results, 'mp', 0.0)) if hasattr(results, 'mp') else 0.0,
                "recall": float(getattr(results, 'mr', 0.0)) if hasattr(results, 'mr') else 0.0,
                
                # Training metrics
                "final_epoch": int(getattr(results, 'epoch', 0)) if hasattr(results, 'epoch') else 0,
                "best_fitness": float(getattr(results, 'fitness', 0.0)) if hasattr(results, 'fitness') else 0.0,
                
                # Loss metrics
                "box_loss": float(getattr(results, 'box_loss', 0.0)) if hasattr(results, 'box_loss') else 0.0,
                "cls_loss": float(getattr(results, 'cls_loss', 0.0)) if hasattr(results, 'cls_loss') else 0.0,
                "dfl_loss": float(getattr(results, 'dfl_loss', 0.0)) if hasattr(results, 'dfl_loss') else 0.0,
            },
            "model_info": {
                "parameters": getattr(results, 'model_params', 0),
                "model_size_mb": 0.0,  # Will be calculated below
                "inference_speed_ms": 0.0,  # Placeholder for inference speed
            }
        }
        
        # Calculate model size if model files exist
        model_files = ['best.pt', 'last.pt']
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                validation_metrics["model_info"]["model_size_mb"] = round(size_mb, 2)
                break
        
        # Save evaluation metrics
        evaluation_path = os.path.join(model_dir, "evaluation.json")
        with open(evaluation_path, 'w') as f:
            json.dump(validation_metrics, f, indent=2)
        print(f"Saved evaluation metrics to: {evaluation_path}")
        
        # Model constraints (thresholds for model approval) - from config
        model_constraints = {
            "model_quality_constraints": {
                "min_mAP_0.5": validation_config.get('min_mAP_0_5', 0.3),
                "min_precision": validation_config.get('min_precision', 0.5),
                "min_recall": validation_config.get('min_recall', 0.4),
                "max_model_size_mb": validation_config.get('max_model_size_mb', 500),
                "max_inference_time_ms": validation_config.get('max_inference_time_ms', 100)
            },
            "current_performance": {
                "mAP_0.5": validation_metrics["metrics"]["mAP_0.5"],
                "precision": validation_metrics["metrics"]["precision"],
                "recall": validation_metrics["metrics"]["recall"],
                "model_size_mb": validation_metrics["model_info"]["model_size_mb"]
            }
        }
        
        # Save model constraints
        constraints_path = os.path.join(model_dir, "constraints.json")
        with open(constraints_path, 'w') as f:
            json.dump(model_constraints, f, indent=2)
        print(f"Saved model constraints to: {constraints_path}")
        
        # Print summary
        print("Model Validation Summary:")
        print(f"  mAP@0.5: {validation_metrics['metrics']['mAP_0.5']:.3f}")
        print(f"  mAP@0.5:0.95: {validation_metrics['metrics']['mAP_0.5_0.95']:.3f}")
        print(f"  Precision: {validation_metrics['metrics']['precision']:.3f}")
        print(f"  Recall: {validation_metrics['metrics']['recall']:.3f}")
        print(f"  Model Size: {validation_metrics['model_info']['model_size_mb']:.1f} MB")
        
        # Check if model meets quality constraints
        constraints = model_constraints["model_quality_constraints"]
        current = model_constraints["current_performance"]
        
        meets_constraints = (
            current["mAP_0.5"] >= constraints["min_mAP_0.5"] and
            current["precision"] >= constraints["min_precision"] and
            current["recall"] >= constraints["min_recall"] and
            current["model_size_mb"] <= constraints["max_model_size_mb"]
        )
        
        print(f"\nModel Quality Assessment:")
        print(f"  Meets quality constraints: {'✅ YES' if meets_constraints else '❌ NO'}")
        if not meets_constraints:
            print("  Issues found:")
            if current["mAP_0.5"] < constraints["min_mAP_0.5"]:
                print(f"    - mAP@0.5 too low: {current['mAP_0.5']:.3f} < {constraints['min_mAP_0.5']}")
            if current["precision"] < constraints["min_precision"]:
                print(f"    - Precision too low: {current['precision']:.3f} < {constraints['min_precision']}")
            if current["recall"] < constraints["min_recall"]:
                print(f"    - Recall too low: {current['recall']:.3f} < {constraints['min_recall']}")
            if current["model_size_mb"] > constraints["max_model_size_mb"]:
                print(f"    - Model too large: {current['model_size_mb']:.1f} MB > {constraints['max_model_size_mb']} MB")
        
    except Exception as e:
        print(f"Warning: Could not generate complete metrics: {e}")
        # Create minimal metrics file
        minimal_metrics = {
            "model_name": "YOLO",
            "validation_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "status": "training_completed",
            "error": str(e)
        }
        
        evaluation_path = os.path.join(model_dir, "evaluation.json")
        with open(evaluation_path, 'w') as f:
            json.dump(minimal_metrics, f, indent=2)


def send_metrics_to_cloudwatch(recall: float, map_50: float, region: str = None) -> bool:
    """
    Send custom metrics to CloudWatch.
    
    Args:
        recall: Model recall value
        map_50: Model mAP@0.5 value  
        region: AWS region (defaults to environment or us-east-1)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Determine region
        aws_region = region or os.environ.get('AWS_REGION', 'us-east-1')
        
        # Create CloudWatch client
        cloudwatch = boto3.client('cloudwatch', region_name=aws_region)
        
        # Get job name for dimensioning
        job_name = os.environ.get('SM_JOB_NAME', 'UnknownJob')
        
        # Prepare metrics
        metrics = [
            {
                'MetricName': 'Recall',
                'Dimensions': [{'Name': 'TrainingJobName', 'Value': job_name}],
                'Value': recall,
                'Unit': 'None'
            },
            {
                'MetricName': 'mAP_0.5',
                'Dimensions': [{'Name': 'TrainingJobName', 'Value': job_name}],
                'Value': map_50,
                'Unit': 'None'
            }
        ]
        
        # Send metrics to CloudWatch
        cloudwatch.put_metric_data(
            Namespace='YOLOTrainingMetrics',
            MetricData=metrics
        )
        
        print(f"Sent metrics to CloudWatch: recall={recall:.4f}, mAP@0.5={map_50:.4f}")
        return True
        
    except Exception as e:
        print(f"Failed to send metrics to CloudWatch: {e}")
        return False


def save_metrics_for_pipeline(model_dir: str, output_dir: str) -> Dict[str, float]:
    """
    Extract and save metrics from YOLO training results for pipeline consumption.
    
    Args:
        model_dir: Directory containing training results
        output_dir: Directory to save pipeline metrics
        
    Returns:
        Dict containing extracted metrics
    """
    import pandas as pd
    
    # Look for results.csv in training output
    results_csv = os.path.join(model_dir, "results.csv")
    
    if not os.path.exists(results_csv):
        print(f"results.csv not found at {results_csv}")
        raise FileNotFoundError(f"results.csv not found")
    
    # Read results and get final epoch metrics
    df = pd.read_csv(results_csv)
    last_row = df.iloc[-1]  # Last epoch
    
    # Extract key metrics
    recall = float(last_row.get("metrics/recall(B)", 0))
    map_50 = float(last_row.get("metrics/mAP50(B)", 0))
    
    metrics = {
        "recall": recall,
        "map_50": map_50
    }
    
    # Save metrics for pipeline
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Pipeline metrics - recall: {recall:.4f}, mAP@0.5: {map_50:.4f}")
    print(f"Saved metrics to {metrics_path}")
    
    return metrics


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


def parse_training_args(config=None):
    """Parse training arguments with config.yaml taking priority."""
    parser = argparse.ArgumentParser(description='YOLO Training for SageMaker')
    
    # Get defaults from config or use hardcoded defaults
    def get_default(config_section, config_key, fallback):
        if config and config_section in config and config_key in config[config_section]:
            return config[config_section][config_key]
        return fallback
    
    # Core training parameters
    parser.add_argument('--model_name', type=str, 
                       default=get_default('hyperparameters', 'model_name', 'yolo11n.pt'))
    parser.add_argument('--epochs', type=int, 
                       default=get_default('training', 'epochs', 100))
    parser.add_argument('--batch_size', type=int, 
                       default=get_default('training', 'batch_size', 16))
    parser.add_argument('--image_size', type=int, 
                       default=get_default('training', 'image_size', 640))
    
    # Learning parameters
    parser.add_argument('--lr0', type=float, 
                       default=get_default('training', 'lr0', 0.01))
    parser.add_argument('--optimizer', type=str, 
                       default=get_default('training', 'optimizer', 'AdamW'))
    
    # Add other parameters with reasonable defaults
    parser.add_argument('--lrf', type=float, default=get_default('training', 'lrf', 0.1))
    parser.add_argument('--momentum', type=float, default=get_default('training', 'momentum', 0.937))
    parser.add_argument('--weight_decay', type=float, default=get_default('training', 'weight_decay', 0.0005))
    parser.add_argument('--warmup_epochs', type=float, default=get_default('training', 'warmup_epochs', 3.0))
    parser.add_argument('--cos_lr', type=lambda x: x.lower() == 'true', default=get_default('training', 'cos_lr', True))
    parser.add_argument('--patience', type=int, default=get_default('training', 'patience', 50))
    parser.add_argument('--amp', type=lambda x: x.lower() == 'true', default=get_default('training', 'amp', True))
    
    # Augmentation parameters
    parser.add_argument('--hsv_h', type=float, default=get_default('training', 'hsv_h', 0.015))
    parser.add_argument('--hsv_s', type=float, default=get_default('training', 'hsv_s', 0.7))
    parser.add_argument('--hsv_v', type=float, default=get_default('training', 'hsv_v', 0.4))
    parser.add_argument('--degrees', type=float, default=get_default('training', 'degrees', 10.0))
    parser.add_argument('--translate', type=float, default=get_default('training', 'translate', 0.1))
    parser.add_argument('--scale', type=float, default=get_default('training', 'scale', 0.5))
    parser.add_argument('--fliplr', type=float, default=get_default('training', 'fliplr', 0.5))
    parser.add_argument('--mosaic', type=float, default=get_default('training', 'mosaic', 1.0))
    parser.add_argument('--mixup', type=float, default=get_default('training', 'mixup', 0.1))
    
    # Loss weights
    parser.add_argument('--box', type=float, default=get_default('training', 'box', 7.5))
    parser.add_argument('--cls', type=float, default=get_default('training', 'cls', 0.5))
    parser.add_argument('--dfl', type=float, default=get_default('training', 'dfl', 1.5))
    parser.add_argument('--label_smoothing', type=float, default=get_default('training', 'label_smoothing', 0.1))
    parser.add_argument('--dropout', type=float, default=get_default('training', 'dropout', 0.1))
    
    return parser.parse_args()
