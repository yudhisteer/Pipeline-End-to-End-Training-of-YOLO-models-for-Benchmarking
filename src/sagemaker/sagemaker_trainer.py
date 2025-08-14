"""
SageMaker training utilities for YOLO models using PyTorch.
"""

import sagemaker
from sagemaker import get_execution_role
from sagemaker.tuner import (
    CategoricalParameter,
    ContinuousParameter,
    IntegerParameter,
    HyperparameterTuner,
)
from sagemaker.pytorch import PyTorch
import os
import sys
from typing import Dict, Optional, Tuple, Any
import boto3


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    load_config,
    get_data_config,
    get_aws_config,
    get_training_config,
    get_hyperparameters_config,
    get_tuning_config,
    get_runtime_config,
)



class SageMakerTrainer:
    """Handles SageMaker model training for YOLO with PyTorch."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SageMaker trainer with configuration.
        
        Args:
            config: Configuration dictionary loaded from YAML file
        """
        self.config = config or {}
        
        # Extract AWS configuration
        aws_config = get_aws_config(self.config)
        self.bucket = aws_config.get('bucket')
        self.prefix = aws_config.get('prefix')
        self.role_arn = aws_config.get('role_arn')
        region = aws_config.get('region')
        
        if not self.bucket:
            raise ValueError("Bucket must be specified in config.yaml")
            
        self.sess = sagemaker.Session()
        self.region = region or self.sess.boto_region_name
        
        # Handle role for local development
        if self.role_arn:
            self.role = self.role_arn
        else:
            try:
                self.role = get_execution_role()
            except:
                raise ValueError(
                    "Must provide role_arn when running locally. "
                    f"Use: --role-arn {self.role_arn}"
                )
        
        # S3 paths - read directly from config (YOLO expects a single training channel with data.yaml)
        data_config = get_data_config(self.config)
        # Prefer new complete dataset path, fallback to old train path for backward compatibility
        self.s3_train_data = (
            data_config.get('s3_dataset_prefix')
            or data_config.get('s3_train_prefix')
            or data_config.get('s3_train_path')
        )
        self.s3_output_location = f"s3://{self.bucket}/{self.prefix}/output"
        
        # Validate S3 paths
        if not self.s3_train_data:
            raise ValueError("s3_dataset_prefix (or s3_train_prefix) must be specified in config.yaml under 'data' section")
        
        # Training components
        self.estimator = None
        self.tuner = None
        
        print(f"Initialized SageMaker trainer for bucket: {self.bucket}, prefix: {self.prefix}")
        print(f"Output location: {self.s3_output_location}")
        print(f"S3 data paths from config:")
        print(f"  Training data: {self.s3_train_data}")
    
    def create_estimator(self) -> sagemaker.estimator.Estimator:
        """
        Create SageMaker PyTorch estimator for YOLO.
        
        Returns:
            Configured SageMaker PyTorch estimator
        """
        # Get training configuration
        training_config = get_training_config(self.config)
        instance_type = training_config.get('instance_type', 'ml.g4dn.xlarge')
        instance_count = training_config.get('instance_count', 1)
        volume_size = training_config.get('volume_size', 50)
        max_run = training_config.get('max_run', 360000)

        metric_definitions = [
            {"Name": "recall", "Regex": r"recall: ([0-9\.]+)"},
            {"Name": "mAP_0.5", "Regex": r"mAP@0\.5: ([0-9\.]+)"},
        ]

        # Create PyTorch estimator with minimal entry point
        self.estimator = PyTorch(
            entry_point="yolo_entrypoint.py",
            source_dir="src/sagemaker",
            role=self.role,
            framework_version="2.0",
            py_version="py310",
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size=volume_size,
            max_run=max_run,
            output_path=self.s3_output_location,
            sagemaker_session=self.sess,
            metric_definitions=metric_definitions,
            dependencies=["src/sagemaker/requirements.txt"],
        )

        print(f"Created PyTorch estimator with instance type: {instance_type}")
        return self.estimator
    
    def set_hyperparameters(self):
        """
        Set script hyperparameters for YOLO training using configuration.
        """
        if self.estimator is None:
            raise ValueError("Must create estimator first using create_estimator()")
        
        # Get hyperparameters from config
        hyperparams_config = get_hyperparameters_config(self.config)
        training_config = get_training_config(self.config)
        
        # Build hyperparameters that the training script understands
        hyperparams = {
            # Basic training parameters
            "model_name": hyperparams_config.get('model_name', 'yolo11n.pt'),
            "image_size": training_config.get('image_size', 640),
            "epochs": training_config.get('epochs', 100),
            "batch_size": training_config.get('batch_size', 16),
            
            # Learning rate & optimization
            "lr0": training_config.get('lr0', 0.01),
            "lrf": training_config.get('lrf', 0.1),
            "momentum": training_config.get('momentum', 0.937),
            "weight_decay": training_config.get('weight_decay', 0.0005),
            "warmup_epochs": training_config.get('warmup_epochs', 3.0),
            "cos_lr": training_config.get('cos_lr', True),
            "optimizer": training_config.get('optimizer', 'AdamW'),
            
            # Data augmentation
            "hsv_h": training_config.get('hsv_h', 0.015),
            "hsv_s": training_config.get('hsv_s', 0.7),
            "hsv_v": training_config.get('hsv_v', 0.4),
            "degrees": training_config.get('degrees', 10.0),
            "translate": training_config.get('translate', 0.1),
            "scale": training_config.get('scale', 0.5),
            "fliplr": training_config.get('fliplr', 0.5),
            "mosaic": training_config.get('mosaic', 1.0),
            "mixup": training_config.get('mixup', 0.1),
            
            # Loss function weights
            "box": training_config.get('box', 7.5),
            "cls": training_config.get('cls', 0.5),
            "dfl": training_config.get('dfl', 1.5),
            "label_smoothing": training_config.get('label_smoothing', 0.1),
            
            # Advanced training options
            "patience": training_config.get('patience', 50),
            "dropout": training_config.get('dropout', 0.1),
            "amp": training_config.get('amp', True),
        }
        
        self.estimator.set_hyperparameters(**hyperparams)
        
        print("Set script hyperparameters:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
    
    def create_hyperparameter_tuner(self) -> HyperparameterTuner:
        """
        Create hyperparameter tuner for automatic hyperparameter optimization.
        
        Returns:
            Configured hyperparameter tuner
        """
        if self.estimator is None:
            raise ValueError("Must create estimator first using create_estimator()")
        
        # Get tuning configuration
        tuning_config = get_tuning_config(self.config)
        max_jobs = tuning_config.get('max_jobs', 8)
        max_parallel_jobs = tuning_config.get('max_parallel_jobs', 1)
        # Must match a metric emitted by the training script and captured by metric_definitions
        objective_metric = tuning_config.get('objective_metric', 'mAP_0.5')
        objective_type = tuning_config.get('objective_type', 'Maximize')
        
        # Build hyperparameter ranges from config
        hyperparameter_ranges = {}
        ranges_config = tuning_config.get('hyperparameter_ranges', {})
        
        for param_name, param_config in ranges_config.items():
            if param_config['type'] == 'continuous':
                hyperparameter_ranges[param_name] = ContinuousParameter(
                    param_config['min'], param_config['max']
                )
            elif param_config['type'] == 'categorical':
                hyperparameter_ranges[param_name] = CategoricalParameter(
                    param_config['values']
                )
            elif param_config['type'] == 'integer':
                hyperparameter_ranges[param_name] = IntegerParameter(
                    param_config['min'], param_config['max']
                )
        # example output:
        # hyperparameter_ranges = {
        # 'learning_rate': ContinuousParameter(0.001, 0.1),
        # 'batch_size': IntegerParameter(8, 64),
        # 'optimizer': CategoricalParameter(['adam', 'sgd', 'rmsprop'])
        # }
        
        # Create tuner
        self.tuner = HyperparameterTuner(
            estimator=self.estimator,
            objective_metric_name=objective_metric,
            hyperparameter_ranges=hyperparameter_ranges,
            objective_type=objective_type,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs
        )
        
        print(f"Created hyperparameter tuner:")
        print(f"  Max jobs: {max_jobs}")
        print(f"  Max parallel jobs: {max_parallel_jobs}")
        print(f"  Objective: {objective_type} {objective_metric}")
        print(f"  Hyperparameter ranges: {list(hyperparameter_ranges.keys())}")
        
        return self.tuner
    
    def prepare_training_data(self) -> Dict[str, sagemaker.inputs.TrainingInput]:
        """
        Prepare training inputs for YOLO script.
        Since data.yaml is uploaded with training data, we only need the training channel.
        
        Returns:
            Dictionary of data channels for training
        """
        if not self.s3_train_data:
            raise ValueError("S3 data path not found. Please ensure s3_train_prefix is set in config.yaml under 'data'")

        train_data = sagemaker.inputs.TrainingInput(
            self.s3_train_data,
            distribution="FullyReplicated",
            s3_data_type="S3Prefix",
        )

        data_channels = {"training": train_data}

        print("Prepared training data channels:")
        print(f"  training: {self.s3_train_data}")

        return data_channels
    
    def start_training(self):
        """
        Start training job using configuration settings.
        """
        data_channels = self.prepare_training_data()
        
        # Get runtime configuration
        runtime_config = get_runtime_config(self.config)
        tuning_config = get_tuning_config(self.config)
        wait = runtime_config.get('wait_for_completion', True)
        logs = runtime_config.get('show_logs', True)
        use_tuner = tuning_config.get('enabled', False)
        
        if use_tuner:
            if self.tuner is None:
                raise ValueError("Must create tuner first using create_hyperparameter_tuner()")
            
            print("Starting hyperparameter tuning job...")
            self.tuner.fit(inputs=data_channels, wait=wait, logs=logs)
            
            if wait:
                print("Hyperparameter tuning completed!")
                # Get best training job
                best_training_job = self.tuner.best_training_job()
                print(f"Best training job: {best_training_job}")
        else:
            if self.estimator is None:
                raise ValueError("Must create estimator first using create_estimator()")
            
            print("Starting training job...")
            self.estimator.fit(inputs=data_channels, wait=wait, logs=logs)
            
            if wait:
                print("Training completed!")
    
    def get_model_artifacts(self) -> str:
        """
        Get the S3 path to trained model artifacts.
        
        Returns:
            S3 path to model artifacts
        """
        if self.tuner and hasattr(self.tuner, 'best_training_job'):
            # Get best model from hyperparameter tuning
            best_job = self.tuner.best_training_job()
            model_data = self.sess.describe_training_job(best_job)['ModelArtifacts']['S3ModelArtifacts']
        elif self.estimator and hasattr(self.estimator, 'model_data'):
            # Get model from regular training
            model_data = self.estimator.model_data
        else:
            raise ValueError("No trained model found. Complete training first.")
        
        print(f"Model artifacts location: {model_data}")
        return model_data


def main():
    """Configuration-driven SageMaker training. Assumes data already uploaded to S3."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train plastic bag detection model on SageMaker")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    print("Note: This script assumes training data (including data.yaml) is already uploaded to S3!")
    
    # Create trainer with configuration
    trainer = SageMakerTrainer(config)
    
    # Create estimator
    print("Creating SageMaker estimator...")
    trainer.create_estimator()
    
    # Set hyperparameters
    print("Setting hyperparameters...")
    trainer.set_hyperparameters()
    
    # Create tuner if enabled
    tuning_config = get_tuning_config(config)
    if tuning_config.get('enabled', False):  # Default to False for simplicity
        print("Creating hyperparameter tuner...")
        trainer.create_hyperparameter_tuner()
    
    # Start training
    print("Starting training job...")
    trainer.start_training()
    
    # Get model artifacts
    print("Getting model artifacts...")
    model_path = trainer.get_model_artifacts()
    print(f"Training complete! Model saved to: {model_path}")


if __name__ == "__main__":
    main()

    """
    Configuration-driven SageMaker training
    
    This script handles training only. Data upload is handled separately.
    
    Prerequisites:
    1. Upload training data to S3.
    2. Ensure config.yaml has correct AWS settings
    
    Usage:
    python src/sagemaker/sagemaker_trainer.py [--config CONFIG_FILE]
    
    Options:
      --config CONFIG_FILE       Configuration file path (default: config.yaml)
    """
    