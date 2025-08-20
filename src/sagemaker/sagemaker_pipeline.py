"""
Comprehensive SageMaker Pipeline for YOLO Model Training and Registration.
Uses the standalone YOLOSageMakerTrainer for training functionality.
"""

import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, CacheConfig
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model_metrics import ModelMetrics, MetricsSource

from utils.utils_config import load_config, get_validation_config
from sagemaker_trainer import YOLOSageMakerTrainer
from sagemaker_metrics import display_training_job_metrics


class YOLOSageMakerPipeline:
    """
    SageMaker Pipeline for YOLO model training and registration.
    Uses YOLOSageMakerTrainer for training functionality.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the YOLO SageMaker pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        
        # Extract pipeline-specific configurations
        self.pipeline_config = self.config.get('pipeline', {})
        self.runtime_config = self.config.get('runtime', {})
        self.validation_config = get_validation_config(self.config)
        
        # Setup pipeline naming with timestamp for uniqueness
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._setup_pipeline_names()
        
        # Initialize pipeline components
        self.pipeline_session = PipelineSession()
        
        # Create trainer instance with pipeline session
        self.trainer = YOLOSageMakerTrainer(
            config_path=config_path, 
            sagemaker_session=self.pipeline_session
        )
        
        self.pipeline = None
        
        print(f"Initialized YOLO SageMaker Pipeline with timestamp: {self.timestamp}")
        print(f"Pipeline name: {self.pipeline_name}")
        print(f"Model package group: {self.model_package_group_name}")
    
    def _setup_pipeline_names(self):
        """Setup naming for pipeline components."""
        # Get custom names from config or use defaults
        self.pipeline_name = self.pipeline_config.get(
            'name', "yolo-training-pipeline-object-detection"
        )
        
        self.training_step_name = self.pipeline_config.get(
            'training_step_name', "YOLOTrainingStep-ObjectDetection"
        )
        
        self.registration_step_name = self.pipeline_config.get(
            'registration_step_name', "YOLOModelRegistrationStep-ObjectDetection"
        )
        
        self.model_package_group_name = self.pipeline_config.get(
            'model_package_group_name', "yolo-model-package-group-object-detection"
        )
        
        self.execution_name = f"yolo-pipeline-execution-object-detection-{self.timestamp}"
        
        print(f"Pipeline naming setup completed:")
        print(f"  Pipeline: {self.pipeline_name}")
        print(f"  Training Step: {self.training_step_name}")
        print(f"  Registration Step: {self.registration_step_name}")
        print(f"  Model Package Group: {self.model_package_group_name}")
    
    def create_training_step(self) -> TrainingStep:
        """
        Create the training step using YOLOSageMakerTrainer.
        
        Returns:
            Configured training step
        """
        # Create estimator using trainer
        estimator = self.trainer.create_estimator()
        
        # Prepare training inputs using trainer
        inputs = self.trainer.prepare_training_inputs()
        
        # Configure caching based on config
        cache_config = None
        if self.pipeline_config.get('enable_caching', False):
            cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")  # 1 hour cache
        
        # Create training step
        training_step = TrainingStep(
            name=self.training_step_name,
            estimator=estimator,
            inputs=inputs,
            cache_config=cache_config,
        )
        
        print(f"Created training step: {self.training_step_name}")
        print(f"  Training data input: {self.trainer.s3_training_data}")
        print(f"  Config input: {'Available' if 'config' in inputs else 'Not available'}")
        
        # Display hyperparameters for pipeline visibility
        print(f"Pipeline training hyperparameters:")
        if self.trainer.hyperparams_config:
            for key, value in self.trainer.hyperparams_config.items():
                print(f"  {key}: {value}")
        else:
            print("  No hyperparameters configured - using YOLO defaults")
        
        return training_step
    
    def create_model_registration_step(self, training_step: TrainingStep) -> RegisterModel:
        """
        Create the model registration step for the pipeline.
        
        Args:
            training_step: The training step that produces the model
            
        Returns:
            Configured model registration step
        """
        # Determine approval status based on config
        auto_approve = self.pipeline_config.get('auto_approve_models', False)
        approval_status = "Approved" if auto_approve else "PendingManualApproval"
        
        # Create model registration step
        registration_step = RegisterModel(
            name=self.registration_step_name,
            estimator=self.trainer.estimator,
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["application/x-onnx", "application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large", "ml.m5.xlarge", "ml.g4dn.xlarge"],
            transform_instances=["ml.m5.large", "ml.m5.xlarge"],
            model_package_group_name=self.model_package_group_name,
            approval_status=approval_status,
            model_metrics=self._create_model_metrics()
        )
        
        print(f"Created model registration step: {self.registration_step_name}")
        print(f"  Model package group: {self.model_package_group_name}")
        print(f"  Approval status: {approval_status}")
        
        return registration_step
    
    def _create_model_metrics(self):
        """
        Create model metrics for YOLO model validation and registration.
        
        Returns:
            ModelMetrics object with YOLO-specific validation metrics
        """
        model_statistics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=f"{self.trainer.s3_model_output}/evaluation.json",
                content_type="application/json"
            ),
            model_constraints=MetricsSource(
                s3_uri=f"{self.trainer.s3_model_output}/constraints.json",
                content_type="application/json"
            )
        )
        
        # TODO: can we parametrize for keypoint detection?
        print("Model metrics configured:")
        print("  - mAP@0.5 (Primary detection metric)")
        print("  - mAP@0.5:0.95 (COCO evaluation metric)")
        print("  - Precision/Recall (Classification metrics)")
        print("  - Speed metrics (Inference performance)")
        print("  - Model size (Memory footprint)")
        
        return model_statistics
    
    def create_pipeline(self) -> Pipeline:
        """
        Create the complete pipeline with training and registration steps.
        
        Returns:
            Configured pipeline
        """
        # Create training step
        training_step = self.create_training_step()
        
        # Create model registration step
        registration_step = self.create_model_registration_step(training_step)
        
        # Create pipeline
        self.pipeline = Pipeline(
            name=self.pipeline_name,
            steps=[training_step, registration_step],
            sagemaker_session=self.pipeline_session,
        )
        
        print(f"Created complete pipeline: {self.pipeline_name}")
        print(f"  Steps: {len(self.pipeline.steps)}")
        print(f"  1. {training_step.name}")
        print(f"  2. {registration_step.name}")
        
        return self.pipeline
    
    def upsert_pipeline(self) -> Dict[str, Any]:
        """
        Create or update the pipeline in SageMaker.
        
        Returns:
            Pipeline response from SageMaker
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be created first using create_pipeline()")
        
        print(f"Upserting pipeline: {self.pipeline_name}")
        
        # Upsert (create or update) the pipeline
        response = self.pipeline.upsert(role_arn=self.trainer.role_arn)
        
        print(f"Pipeline upserted successfully!")
        print(f"  Pipeline ARN: {response.get('PipelineArn', 'N/A')}")
        
        return response
    
    def start_execution(self, execution_display_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Start pipeline execution.
        
        Args:
            execution_display_name: Optional custom execution name
            
        Returns:
            Execution object
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be created first using create_pipeline()")
        
        # Set default execution name if not provided
        if execution_display_name is None:
            execution_display_name = self.execution_name
        
        print(f"Starting pipeline execution: {execution_display_name}")
        
        # Start execution
        execution = self.pipeline.start(
            execution_display_name=execution_display_name,
            execution_description="YOLO model training execution"
        )
        
        print(f"Pipeline execution started successfully!")
        print(f"  Execution ARN: {execution.arn}")
        print(f"\nMonitoring Information:")
        print(f"  Pipeline Name: {self.pipeline_name}")
        print(f"  Execution Name: {execution_display_name}")
        print(f"  Region: {self.trainer.region}")
        print(f"  CloudWatch Logs: {self.trainer.training_log_group}")
        
        return execution
    
    def run_pipeline(self, wait_for_completion: bool = False) -> Dict[str, Any]:
        """
        Complete pipeline workflow: create, upsert, and execute.
        
        Args:
            wait_for_completion: Whether to wait for pipeline completion
            
        Returns:
            Execution information
        """
        print("="*60)
        print("Starting YOLO SageMaker Pipeline Workflow")
        print("="*60)
        
        # Create pipeline
        self.create_pipeline()
        
        # Upsert pipeline
        self.upsert_pipeline()
        
        # Start execution
        execution = self.start_execution()
        
        if wait_for_completion:
            print(f"\nWaiting for pipeline execution to complete...")
            print(f"You can monitor progress at:")
            print(f"  SageMaker Console: https://{self.trainer.region}.console.aws.amazon.com/sagemaker/home?region={self.trainer.region}#/pipelines/{self.pipeline_name}/executions/{self.execution_name}")
            
            # Monitor execution with detailed logging
            self._monitor_execution_with_logs(execution)
            
            print(f"Pipeline execution completed!")
            
            # Get final execution status
            status = execution.describe()
            final_status = status.get('PipelineExecutionStatus', 'Unknown')
            print(f"Final status: {final_status}")
            
            # Show step results
            self._print_execution_summary(execution)
            
            # Extract and display training metrics if enabled and execution succeeded
            if self.runtime_config.get('display_metrics', True) and final_status == 'Succeeded':
                self._display_training_metrics_from_execution(execution)
            elif final_status == 'Failed':
                print(f"\nPipeline execution failed. Skipping metrics extraction.")
                print(f"Extracting training job logs to diagnose the failure...")
                self._display_training_failure_logs_from_execution(execution)
        else:
            print(f"\nPipeline execution started. Monitor progress in SageMaker console.")
            print(f"  SageMaker Console: https://{self.trainer.region}.console.aws.amazon.com/sagemaker/home?region={self.trainer.region}#/pipelines/{self.pipeline_name}/executions/{self.execution_name}")
        
        print("="*60)
        print("YOLO SageMaker Pipeline Workflow Complete")
        print("="*60)
        
        return {
            "pipeline_name": self.pipeline_name,
            "execution_arn": execution.arn,
            "timestamp": self.timestamp,
            "model_package_group": self.model_package_group_name
        }
    
    def _monitor_execution_with_logs(self, execution):
        """Monitor pipeline execution with detailed logging."""
        print("="*50)
        print("PIPELINE EXECUTION MONITORING")
        print("="*50)
        
        last_status = None
        step_statuses = {}
        
        while True:
            try:
                # Get current execution status
                current_status = execution.describe()
                execution_status = current_status.get('PipelineExecutionStatus', 'Unknown')
                
                # Print execution status change
                if execution_status != last_status:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{timestamp}] Pipeline Status: {execution_status}")
                    last_status = execution_status
                
                # Get step statuses
                try:
                    steps = execution.list_steps()
                    for step in steps:
                        step_name = step.get('StepName', 'Unknown')
                        step_status = step.get('StepStatus', 'Unknown')
                        
                        # Print step status change
                        if step_name not in step_statuses or step_statuses[step_name] != step_status:
                            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                            print(f"[{timestamp}] Step '{step_name}': {step_status}")
                            
                            # Show additional details for training steps
                            if step_status == 'Executing' and 'train' in step_name.lower():
                                training_job_arn = step.get('Metadata', {}).get('TrainingJob', {}).get('Arn', '')
                                if training_job_arn:
                                    job_name = training_job_arn.split('/')[-1]
                                    print(f"    Training Job: {job_name}")
                                    print(f"    CloudWatch Logs: https://{self.trainer.region}.console.aws.amazon.com/cloudwatch/home?region={self.trainer.region}#logsV2:log-groups/log-group/%2Faws%2Fsagemaker%2FTrainingJobs/log-events/{job_name}")
                            
                            step_statuses[step_name] = step_status
                
                except Exception as e:
                    print(f"Warning: Could not get step details: {e}")
                
                # Check if execution is complete
                if execution_status in ['Succeeded', 'Failed', 'Stopped']:
                    break
                    
                # Wait before next check
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                print(f"\nMonitoring interrupted by user. Pipeline continues running.")
                print(f"Check SageMaker console for status updates.")
                break
            except Exception as e:
                print(f"Error monitoring execution: {e}")
                time.sleep(30)
                
        print("="*50)
    
    def _print_execution_summary(self, execution):
        """Print detailed execution summary."""
        print("\n" + "="*50)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*50)
        
        try:
            # Get execution details
            status = execution.describe()
            execution_status = status.get('PipelineExecutionStatus', 'Unknown')
            start_time = status.get('CreationTime', 'Unknown')
            end_time = status.get('LastModifiedTime', 'Unknown')
            
            print(f"Execution Status: {execution_status}")
            print(f"Start Time: {start_time}")
            print(f"End Time: {end_time}")
            
            # Get step details
            steps = execution.list_steps()
            print(f"\nStep Results:")
            print("-" * 30)
            
            for step in steps:
                step_name = step.get('StepName', 'Unknown')
                step_status = step.get('StepStatus', 'Unknown')
                step_start = step.get('StartTime', 'N/A')
                step_end = step.get('EndTime', 'N/A')
                
                print(f"Step: {step_name}")
                print(f"  Status: {step_status}")
                print(f"  Start: {step_start}")
                print(f"  End: {step_end}")
                
                # Show training job details if available
                metadata = step.get('Metadata', {})
                if 'TrainingJob' in metadata:
                    training_job = metadata['TrainingJob']
                    training_job_arn = training_job.get('Arn', '')
                    if training_job_arn:
                        job_name = training_job_arn.split('/')[-1]
                        print(f"  Training Job: {job_name}")
                
                # Show model registration details if available
                if 'RegisterModel' in metadata:
                    model_package_arn = metadata['RegisterModel'].get('Arn', '')
                    if model_package_arn:
                        print(f"  Model Package: {model_package_arn}")
                
                print()
                
        except Exception as e:
            print(f"Error getting execution summary: {e}")
        
        print("="*50)
        
        # Display dataset information from trainer
        self.trainer.display_dataset_info()
    
    def _get_training_job_name_from_execution(self, execution):
        """Extract training job name from pipeline execution."""
        try:
            steps = execution.list_steps()
            for step in steps:
                if 'train' in step.get('StepName', '').lower():
                    metadata = step.get('Metadata', {})
                    if 'TrainingJob' in metadata:
                        training_job_arn = metadata['TrainingJob'].get('Arn', '')
                        if training_job_arn:
                            return training_job_arn.split('/')[-1]
        except Exception as e:
            print(f"Warning: Could not extract training job name from pipeline: {e}")
        return None

    def _display_training_failure_logs_from_execution(self, execution):
        """Extract training job name and delegate to trainer for log analysis."""
        training_job_name = self._get_training_job_name_from_execution(execution)
        if training_job_name:
            self.trainer.display_training_failure_logs(training_job_name)
        else:
            print("Could not find training job name from pipeline execution.")
            print("Check the SageMaker console for training job details.")

    def _display_training_metrics_from_execution(self, execution):
        """Extract training job name and delegate to trainer for metrics display."""
        training_job_name = self._get_training_job_name_from_execution(execution)
        if training_job_name:
            self.trainer.display_training_metrics(training_job_name)
        else:
            print("Could not find training job name from pipeline execution.")
            print("You can manually check metrics using:")
            print("  python src/sagemaker/sagemaker_metrics.py [training_job_name]")




def main():
    """Main function for standalone execution of the complete YOLO pipeline workflow."""
    config_path = os.environ.get("YOLO_CONFIG_PATH", "config.yaml")
    
    try:
        print(f"Loading configuration from: {config_path}")
        pipeline = YOLOSageMakerPipeline(config_path=config_path)
        
        # Get runtime configuration for execution behavior
        runtime_config = pipeline.config.get('runtime', {})
        pipeline_config = pipeline.config.get('pipeline', {})
        
        # Check if this is a dry run
        dry_run = pipeline_config.get('dry_run', False)
        
        # Check if we should wait for completion
        wait_for_completion = runtime_config.get('wait_for_completion', False)
        
        if dry_run:
            print("Dry run mode enabled in config: Creating pipeline without execution")
            pipeline.create_pipeline()
            pipeline.upsert_pipeline()
            print("Pipeline created successfully! Set 'pipeline.dry_run: false' in config to execute.")
            result = {
                "pipeline_name": pipeline.pipeline_name,
                "timestamp": pipeline.timestamp,
                "model_package_group": pipeline.model_package_group_name,
                "status": "dry_run_completed"
            }
        else:
            # Run complete pipeline
            result = pipeline.run_pipeline(wait_for_completion=wait_for_completion)
        
        print(f"\nPipeline Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
    
    
    # Example usage:
    """
    This script creates and executes a complete SageMaker pipeline that:
    1. Trains a YOLO model using YOLOSageMakerTrainer
    2. Registers the trained model in the SageMaker Model Registry
    3. Can be run standalone with proper configuration

    The pipeline uses the refactored YOLOSageMakerTrainer for all training functionality,
    making the code more modular and maintainable.

    Usage Examples:

    1. Run pipeline directly (uses config.yaml):
    python src/sagemaker/sagemaker_pipeline.py

    2. Use with custom config path (via environment variable):
    export YOLO_CONFIG_PATH=/path/to/custom/config.yaml
    python src/sagemaker/sagemaker_pipeline.py

    3. Advanced usage - create pipeline object for more control:
    from src.sagemaker.sagemaker_pipeline import YOLOSageMakerPipeline

    pipeline = YOLOSageMakerPipeline("config.yaml")
    pipeline.create_pipeline()
    pipeline.upsert_pipeline()
    execution = pipeline.start_execution()

    4. Integration with standalone trainer:
    from src.sagemaker.sagemaker_trainer import run_yolo_training

    # Run just training (no pipeline)
    result = run_yolo_training()

    # Or run full pipeline using this script
    python src/sagemaker/sagemaker_pipeline.py
    """