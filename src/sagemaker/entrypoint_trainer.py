"""
Standalone SageMaker Training Job for YOLO Models.
Can be used independently for single training jobs or as part of a pipeline.
"""

import os
import boto3
import time
from datetime import datetime
from typing import Dict, Any, Optional

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline_context import PipelineSession

from utils.utils_config import (
    load_config,
    get_aws_config,
    get_data_config,
    get_training_config,
    get_hyperparameters_config,
)
from sagemaker_metrics import display_training_job_metrics


class YOLOSageMakerTrainer:
    """
    Standalone SageMaker trainer for YOLO models.
    Can be used for individual training jobs or integrated into pipelines.
    """
    
    def __init__(self, config_path: str = "config.yaml", sagemaker_session=None):
        """
        Initialize the YOLO SageMaker trainer.
        
        Args:
            config_path: Path to configuration file
            sagemaker_session: Optional SageMaker session (for pipeline integration).
                              If None, creates a default Session. For pipelines, pass PipelineSession.
        """
        self.config = load_config(config_path)
        
        # Extract configurations
        self.aws_config = get_aws_config(self.config)
        self.training_config = get_training_config(self.config)
        self.hyperparams_config = get_hyperparameters_config(self.config)
        self.runtime_config = self.config.get('runtime', {})
        
        self._setup_aws_components()
        
        # Setup timestamp for uniqueness
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._setup_naming()
        self._setup_s3_paths()
        
        # SageMaker session (can be provided for pipeline integration)
        self.sagemaker_session = sagemaker_session or sagemaker.Session()
        self.estimator = None
        
        print(f"Initialized YOLO SageMaker Trainer with timestamp: {self.timestamp}")
        print(f"Job name prefix: {self.job_name_prefix}")
    
    def _setup_aws_components(self):
        """Setup AWS session, region, and role."""
        # Get region
        self.region = self.aws_config.get('region') or boto3.Session().region_name
        
        # Get role ARN
        self.role_arn = self.aws_config.get('role_arn')
        if not self.role_arn:
            try:
                self.role_arn = sagemaker.get_execution_role()
            except Exception:
                raise ValueError(
                    "Role ARN must be specified in config.yaml under 'aws.role_arn' "
                    "when running outside SageMaker environment"
                )
        
        # Get bucket
        self.bucket = self.aws_config.get('bucket')
        if not self.bucket:
            raise ValueError("S3 bucket must be specified in config.yaml under 'aws.bucket'")
        
        self.prefix = self.aws_config.get('prefix', 'yolo-pipeline')
        
        print(f"AWS Setup - Region: {self.region}, Bucket: {self.bucket}, Prefix: {self.prefix}")
    
    def _setup_naming(self):
        """Setup naming for training jobs and resources."""
        training_config = self.config.get('training', {})
        
        # Base job name for training jobs
        self.job_name_prefix = training_config.get('job_name_prefix', 'yolo-training-object-detection')
        
        # CloudWatch log group
        self.training_log_group = f"/aws/sagemaker/TrainingJobs/{self.job_name_prefix}"
        
        print(f"Training naming setup:")
        print(f"  Job name prefix: {self.job_name_prefix}")
        print(f"  Log group: {self.training_log_group}")
    
    def _setup_s3_paths(self):
        """Setup S3 paths for training data and model outputs."""
        # Training data path
        self.s3_training_data = (
            self.training_config.get('s3_dataset_prefix') 
            or self.training_config.get('s3_train_prefix')
        )
        
        if not self.s3_training_data:
            raise ValueError(
                "Training data S3 path must be specified in config.yaml under "
                "'training.s3_dataset_prefix' or 'training.s3_train_prefix'"
            )
        
        # Model output path
        self.s3_model_output = f"s3://{self.bucket}/{self.prefix}/models/{self.timestamp}"
        
        # Code output path
        self.s3_code_output = f"s3://{self.bucket}/{self.prefix}/code/{self.timestamp}"
        
        print(f"S3 Paths configured:")
        print(f"  Training data: {self.s3_training_data}")
        print(f"  Model output: {self.s3_model_output}")
        print(f"  Code output: {self.s3_code_output}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the training dataset from S3.
        
        Returns:
            Dictionary with dataset information
        """
        try:
            from utils.utils_data import get_s3_dataset_info
            
            # Parse S3 URI to get bucket and prefix
            if self.s3_training_data.startswith("s3://"):
                bucket = self.s3_training_data.split("/")[2]
                base_prefix = "/".join(self.s3_training_data.split("/")[3:])
            else:
                bucket = self.bucket
                base_prefix = f"{self.prefix}/yolo-dataset"
            
            # Get dataset info using utility function
            dataset_info = get_s3_dataset_info(bucket, base_prefix)
            
            return dataset_info
            
        except Exception as e:
            print(f"Error getting dataset info: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'bucket': self.bucket,
                'base_prefix': f"{self.prefix}/yolo-dataset"
            }
    
    def display_dataset_info(self):
        """Display dataset information in a formatted way."""
        try:
            print("\n" + "="*50)
            print("DATASET INFORMATION")
            print("="*50)
            
            dataset_info = self.get_dataset_info()
            
            if dataset_info['status'] == 'found':
                print(f"   Dataset Summary:")
                print(f"   Training images: {dataset_info['train_images']}")
                print(f"   Validation images: {dataset_info['val_images']}")
                print(f"   Total images: {dataset_info['train_images'] + dataset_info['val_images']}")
                print(f"   Has data.yaml: {dataset_info['has_data_yaml']}")
            else:
                print(f"Dataset status: {dataset_info['status']}")
                if 'message' in dataset_info:
                    print(f"   Message: {dataset_info['message']}")
            
        except Exception as e:
            print(f"Could not display dataset information: {e}")
        
        print("="*50)
    
    def display_training_failure_logs(self, training_job_name: str):
        """Extract and display training job logs when training fails."""
        try:
            print("\n" + "="*60)
            print("TRAINING FAILURE DIAGNOSIS")
            print("="*60)
            
            print(f"Found failed training job: {training_job_name}")
            print("Fetching CloudWatch logs to diagnose failure...")
            print("-" * 60)
            
            # Get CloudWatch logs
            try:
                import boto3
                logs_client = boto3.client('logs', region_name=self.region)
                
                # Get the most recent log stream
                log_group = f"/aws/sagemaker/TrainingJobs/{training_job_name}"
                
                try:
                    # Get log streams, sorted by last event time
                    response = logs_client.describe_log_streams(
                        logGroupName=log_group,
                        orderBy='LastEventTime',
                        descending=True,
                        limit=1
                    )
                    
                    if response['logStreams']:
                        latest_stream = response['logStreams'][0]['logStreamName']
                        print(f"Latest log stream: {latest_stream}")
                        
                        # Get the last 50 log events (usually contains the error)
                        log_events = logs_client.get_log_events(
                            logGroupName=log_group,
                            logStreamName=latest_stream,
                            startFromHead=False,
                            limit=50
                        )
                        
                        print(f"\nLast 50 log events from training job:")
                        print("-" * 60)
                        
                        for event in reversed(log_events['events']):
                            timestamp = event['timestamp']
                            message = event['message']
                            print(f"[{timestamp}] {message}")
                        
                        print("-" * 60)
                        print(f"Full logs available at:")
                        print(f"  CloudWatch Console: https://{self.region}.console.aws.amazon.com/cloudwatch/home?region={self.region}#logsV2:log-groups/log-group/%2Faws%2Fsagemaker%2FTrainingJobs/log-events/{training_job_name}")
                        
                    else:
                        print(f"No log streams found for training job: {training_job_name}")
                        
                except logs_client.exceptions.ResourceNotFoundException:
                    print(f"Log group not found: {log_group}")
                    print("Training job may have failed before logs were created.")
                    
            except ImportError:
                print("boto3 not available. Cannot fetch CloudWatch logs.")
            except Exception as e:
                print(f"Error fetching CloudWatch logs: {e}")
                
        except Exception as e:
            print(f"Error displaying training failure logs: {e}")
            print("Check the SageMaker console manually for training job details.")
    
    def display_training_metrics(self, training_job_name: str):
        """Extract and display training metrics for a specific training job."""
        try:
            print("\n" + "="*60)
            print("EXTRACTING TRAINING METRICS")
            print("="*60)
            print(f"Found training job: {training_job_name}")
            print("Extracting metrics...")
            print("-" * 60)

            # Import here to avoid circular imports
            from sagemaker_metrics import display_training_job_metrics
            display_training_job_metrics(training_job_name, show_metrics=True)

        except ImportError as e:
            print(f"Could not import sagemaker_metrics: {e}")
            print("Please ensure sagemaker_metrics.py is available.")
        except Exception as e:
            print(f"Error displaying training metrics: {e}")
            print("You can manually check metrics using:")
            print("  python src/sagemaker/sagemaker_metrics.py [training_job_name]")
    
    def create_estimator(self, verbose: bool = True) -> PyTorch:
        """Create and configure the PyTorch estimator for YOLO training.
        
        Args:
            verbose: Whether to print hyperparameters configuration (default: True)
        """
        # Get training configuration
        instance_type = self.training_config.get('instance_type', 'ml.g4dn.xlarge')
        instance_count = self.training_config.get('instance_count', 1)
        volume_size = self.training_config.get('volume_size', 50)
        max_run = self.training_config.get('max_run', 86400)  # 24 hours default
        
        # Metric definitions for CloudWatch
        # TODO: parametrize for keypoint detection
        metric_definitions = [
            {
                "Name": "yolo:recall",
                "Regex": r"recall: ([0-9]*\.?[0-9]+)"
            },
            {
                "Name": "yolo:mAP_0.5",
                "Regex": r"mAP@0\.5: ([0-9\.]+)"
            },
            {
                "Name": "yolo:mAP_0.5_0.95",
                "Regex": r"mAP@0\.5:0\.95: ([0-9\.]+)"
            },
            {
                "Name": "yolo:precision",
                "Regex": r"precision: ([0-9\.]+)"
            },
            {
                "Name": "yolo:train_loss",
                "Regex": r"train/box_loss: ([0-9\.]+)"
            },
            {
                "Name": "yolo:val_loss",
                "Regex": r"val/box_loss: ([0-9\.]+)"
            }
        ]
        
        # Create PyTorch estimator
        self.estimator = PyTorch(
            source_dir="src/sagemaker",
            entry_point="entrypoint_training.py",
            role=self.role_arn,
            framework_version="2.0",
            py_version="py310",
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size=volume_size,
            max_run=max_run,
            output_path=self.s3_model_output,
            code_location=self.s3_code_output,
            sagemaker_session=self.sagemaker_session,
            metric_definitions=metric_definitions,
            dependencies=["src/sagemaker/dependencies/requirements.txt"],
            base_job_name=f"{self.job_name_prefix}-{self.timestamp}"
        )
        
        # Model name will be read from config.yaml in the entrypoint
        # No need to pass as hyperparameter since entrypoint reads config directly
        
        print(f"Created PyTorch estimator:")
        print(f"  Instance type: {instance_type}")
        print(f"  Instance count: {instance_count}")
        print(f"  Volume size: {volume_size} GB")
        print(f"  Max runtime: {max_run} seconds")
        
        # Display hyperparameters that will be used in training
        if verbose:
            if self.hyperparams_config:
                print(f"Training hyperparameters from config:")
                for key, value in self.hyperparams_config.items():
                    print(f"  {key}: {value}")
            else:
                print("  No hyperparameters configured - using YOLO defaults")
        
        return self.estimator

    def _prepare_basic_config_input(self) -> Optional[TrainingInput]:
        """
        Upload config.yaml to S3 and prepare it as a training input.
        
        Returns:
            TrainingInput for config.yaml, or None if config not found
        """
        # Check if config.yaml exists locally
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            print("config.yaml not found locally - training will use defaults")
            return None
        
        try:
            # Upload config.yaml to S3 (in a directory for TrainingInput)
            s3_config_key = f"{self.prefix}/config/{self.timestamp}/config.yaml"
            s3_config_dir = f"s3://{self.bucket}/{self.prefix}/config/{self.timestamp}/"
            
            s3_client = boto3.client('s3', region_name=self.region)
            s3_client.upload_file(config_path, self.bucket, s3_config_key)
            
            print(f"Uploaded config.yaml to: s3://{self.bucket}/{s3_config_key}")
            
            # Create TrainingInput for config directory
            config_input = TrainingInput(
                s3_data=s3_config_dir,
                distribution="FullyReplicated",
                s3_data_type="S3Prefix"
            )
            
            return config_input
            
        except Exception as e:
            print(f"Failed to upload config.yaml to S3: {e}")
            print("Training will use defaults")
            return None

    def prepare_training_inputs(self, execution_id: Optional[str] = None) -> Dict[str, TrainingInput]:
        """
        Prepare training inputs for the job.
        
        Args:
            execution_id: Not used anymore - kept for backward compatibility
            
        Returns:
            Dictionary of training inputs
        """
        # Prepare training data input
        training_input = TrainingInput(
            s3_data=self.s3_training_data,
            distribution="FullyReplicated",
            s3_data_type="S3Prefix"
        )
        
        # Upload config.yaml to S3 and create config input
        config_input = self._prepare_basic_config_input()
        
        # Build inputs dictionary
        inputs = {"training": training_input}
        if config_input:
            inputs["config"] = config_input
        
        print(f"Prepared training inputs:")
        print(f"  Training data: {self.s3_training_data}")
        if config_input:
            print(f"  Config input: Available")
        else:
            print(f"  Config input: Not available (using defaults)")
        
        return inputs
    
    def start_training_job(self, job_name: Optional[str] = None, wait: bool = False) -> sagemaker.estimator.Estimator:
        """
        Start a standalone training job.
        
        Args:
            job_name: Optional custom job name. If None, uses auto-generated name.
            wait: Whether to wait for training completion
            
        Returns:
            The estimator with the started training job
        """
        if self.estimator is None:
            self.create_estimator()
        
        # Prepare training inputs
        inputs = self.prepare_training_inputs(execution_id=None)
        
        # Set custom job name if provided
        if job_name:
            self.estimator.base_job_name = job_name
        
        print(f"Starting training job...")
        print(f"  Base job name: {self.estimator.base_job_name}")
        print(f"  Instance type: {self.estimator.instance_type}")
        print(f"  Training data: {self.s3_training_data}")
        
        # Start training
        self.estimator.fit(inputs, wait=wait)
        
        if wait:
            print(f"Training completed!")
            
            # Display metrics if enabled
            if self.runtime_config.get('display_metrics', True):
                training_job_name = self.estimator.latest_training_job.name
                self._display_training_metrics(training_job_name)
        else:
            print(f"Training job started. Monitor progress in SageMaker console.")
            print(f"  Job name: {self.estimator.latest_training_job.name}")
            print(f"  CloudWatch Logs: {self.training_log_group}")
        
        return self.estimator
    
    def _display_training_metrics(self, training_job_name: str):
        """Display training metrics for the completed job."""
        try:
            print("\n" + "="*60)
            print("EXTRACTING TRAINING METRICS")
            print("="*60)
            print(f"Training job: {training_job_name}")
            print("-" * 60)
            
            display_training_job_metrics(training_job_name, show_metrics=True)
            
        except ImportError as e:
            print(f"Could not import sagemaker_metrics: {e}")
            print("Please ensure sagemaker_metrics.py is available.")
        except Exception as e:
            print(f"Error displaying training metrics: {e}")
            print("You can manually check metrics using:")
            print(f"  python src/sagemaker/sagemaker_metrics.py {training_job_name}")
    
    def run_training(self, wait_for_completion: bool = None, job_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete training workflow: create estimator and start training.
        
        Args:
            wait_for_completion: Whether to wait for completion. If None, uses config setting.
            job_name: Optional custom job name
            
        Returns:
            Training job information
        """
        if wait_for_completion is None:
            wait_for_completion = self.runtime_config.get('wait_for_completion', False)
        
        print("="*60)
        print("Starting YOLO SageMaker Training Job")
        print("="*60)
        
        # Create estimator and start training
        estimator = self.start_training_job(job_name=job_name, wait=wait_for_completion)
        
        training_job_name = estimator.latest_training_job.name
        
        print("="*60)
        print("YOLO SageMaker Training Job Started")
        print("="*60)
        
        return {
            "training_job_name": training_job_name,
            "model_data": estimator.model_data if wait_for_completion else None,
            "timestamp": self.timestamp,
            "s3_model_output": self.s3_model_output
        }

    def get_config_s3_path(self) -> Optional[str]:
        """
        Get the current S3 path where config.yaml is stored.
        
        Returns:
            S3 path to config.yaml directory, or None if not uploaded
        """
        if hasattr(self, '_config_s3_path'):
            return self._config_s3_path
        return None
    
    def upload_config_with_execution_id(self, execution_id: str) -> bool:
        """
        Upload config.yaml to S3 with execution ID in the path for traceability.
        This method should be called after pipeline execution starts.
        
        Args:
            execution_id: The pipeline execution ID
            
        Returns:
            True if successful, False otherwise
        """
        # Check if config.yaml exists locally
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            print("config.yaml not found locally - cannot upload")
            return False
        
        try:
            # Create config directory name with timestamp and execution ID
            config_dir_name = f"{self.timestamp}-{execution_id}"
            
            # Upload config.yaml to S3 with execution ID in path
            s3_config_key = f"{self.prefix}/config/{config_dir_name}/config.yaml"
            s3_config_dir = f"s3://{self.bucket}/{self.prefix}/config/{config_dir_name}/"
            
            print(f"Uploading config.yaml with execution ID for traceability...")
            print(f"  Local path: {config_path}")
            print(f"  S3 location: s3://{self.bucket}/{s3_config_key}")
            
            s3_client = boto3.client('s3', region_name=self.region)
            s3_client.upload_file(config_path, self.bucket, s3_config_key)
            
            print(f"Successfully uploaded config.yaml to: s3://{self.bucket}/{s3_config_key}")
            print(f"  Execution ID: {execution_id}")
            print(f"  Timestamp: {self.timestamp}")
            
            return True
            
        except Exception as e:
            print(f"Failed to upload config.yaml with execution ID: {e}")
            return False




def main():
    """Main function for standalone execution."""
    config_path = os.environ.get("YOLO_CONFIG_PATH", "config.yaml")
    job_name = os.environ.get("YOLO_JOB_NAME", None)
    wait_for_completion = os.environ.get("YOLO_WAIT", "false").lower() == "true"
    
    try:
        print(f"Loading configuration from: {config_path}")
        trainer = YOLOSageMakerTrainer(config_path=config_path)
        
        result = trainer.run_training(
            wait_for_completion=wait_for_completion,
            job_name=job_name
        )
        
        print(f"\nTraining Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error running training: {str(e)}")
        raise


if __name__ == "__main__":
    main()


# Example usage:
"""
This script provides standalone SageMaker training for YOLO models.
It can be used independently or integrated into pipelines.

Usage Examples:

1. Run directly (uses config.yaml):
python src/sagemaker/sagemaker_trainer.py

2. Use with custom config and wait for completion:
export YOLO_CONFIG_PATH=/path/to/custom/config.yaml
export YOLO_WAIT=true
python src/sagemaker/sagemaker_trainer.py

3. Use with custom job name:
export YOLO_JOB_NAME=my-custom-training-job
python src/sagemaker/sagemaker_trainer.py

4. Programmatic usage:
from src.sagemaker.sagemaker_trainer import run_yolo_training

# Simple usage
result = run_yolo_training()

# Advanced usage
from src.sagemaker.sagemaker_trainer import YOLOSageMakerTrainer

trainer = YOLOSageMakerTrainer("config.yaml")
trainer.create_estimator()
estimator = trainer.start_training_job(wait=True)

5. Pipeline integration:
from sagemaker.workflow.pipeline_context import PipelineSession

pipeline_session = PipelineSession()
trainer = YOLOSageMakerTrainer("config.yaml", sagemaker_session=pipeline_session)
estimator = trainer.create_estimator()
# Use estimator in TrainingStep
"""