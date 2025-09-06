import os
import time
import json
import boto3
import logging
import warnings
from datetime import datetime
from typing import Dict, Any, Optional

from rich import print
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, CacheConfig
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model_metrics import ModelMetrics, MetricsSource

from utils.utils_config import load_config, get_validation_config
from utils.utils_train import validate_model_quality, update_model_package_approval, check_and_display_quality_gates
from entrypoint_trainer import YOLOSageMakerTrainer
from sagemaker_metrics import display_training_job_metrics

logging.getLogger('sagemaker').setLevel(logging.ERROR)
logging.getLogger('sagemaker.workflow.utilities').setLevel(logging.ERROR)
logging.getLogger('sagemaker.workflow._utils').setLevel(logging.ERROR)
logging.getLogger('sagemaker.estimator').setLevel(logging.ERROR)
logging.getLogger('sagemaker.image_uris').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='sagemaker.workflow.steps')


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
        
        # extract pipeline-specific configurations
        self.pipeline_config = self.config.get('pipeline', {})
        self.runtime_config = self.config.get('runtime', {})
        self.registry_config = self.config.get('registry', {})
        self.validation_config = get_validation_config(self.config)
        
        # setup pipeline naming with timestamp for uniqueness
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._setup_pipeline_names()
        
        # initialize pipeline components
        self.pipeline_session = PipelineSession()
        
        # create trainer instance with pipeline session
        self.trainer = YOLOSageMakerTrainer(
            config_path=config_path, 
            sagemaker_session=self.pipeline_session
        )
        
        self.pipeline = None
        
    
    def _setup_pipeline_names(self):
        """Setup naming for pipeline components."""

        self.pipeline_name = self.pipeline_config.get('name')
        
        self.training_step_name = self.pipeline_config.get('training_step_name')
        
        self.registration_step_name = self.pipeline_config.get('registration_step_name')
        
        self.model_package_group_name = self.pipeline_config.get('model_package_group_name')
        
        self.execution_name = self.pipeline_config.get('execution_name')
        
    
    def create_training_step(self) -> TrainingStep:
        """
        Create the training step using YOLOSageMakerTrainer.
        
        Returns:
            Configured training step
        """
        # create estimator using trainer
        estimator = self.trainer.create_estimator()
        
        # prepare training inputs using trainer
        inputs = self.trainer.prepare_training_inputs(execution_id=None)
        
        # configure caching
        cache_config = None
        if self.pipeline_config.get('enable_caching'): 
            cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")  # 1 hour cache
        
        # Create training step
        training_step = TrainingStep(
            name=self.training_step_name,
            estimator=estimator,
            inputs=inputs,
            cache_config=cache_config,
        )
        
        
        return training_step
    
    def create_model_registration_step(self, training_step: TrainingStep) -> RegisterModel:
        """
        Create the model registration step for the pipeline.
        
        Args:
            training_step: The training step that produces the model
            
        Returns:
            Configured model registration step
        """
        # All models start as pending - approval happens post-training based on actual metrics
        approval_status = "PendingManualApproval"
        approval_reason = "Pending quality gate validation after training completes"
        
        # Create model metrics using results.csv from training output
        # This provides evaluation metrics in SageMaker Model Registry UI while quality gates continue to use S3 results.csv
        model_metrics = self._create_model_metrics(training_step)
        
        # create model registration step
        registration_step = RegisterModel(
            name=self.registration_step_name,
            estimator=self.trainer.estimator,
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["application/x-onnx", "application/json"],
            response_types=["application/json"],
            inference_instances=self.config.get('registry', {}).get('model_package', {}).get('inference_instances'),
            transform_instances=self.config.get('registry', {}).get('model_package', {}).get('transform_instances'),
            model_package_group_name=self.model_package_group_name,
            approval_status=approval_status,
            model_metrics=model_metrics  # Include evaluation metrics for SageMaker Model Registry UI
        )
        
        return registration_step
    
    def _create_model_metrics(self, training_step: TrainingStep) -> ModelMetrics:
        """
        Create ModelMetrics object for SageMaker Model Registry.
        
        Args:
            training_step: The training step that produces model artifacts
            
        Returns:
            ModelMetrics object with evaluation metrics from results.csv
        """
        # Use Join to properly concatenate pipeline properties with strings
        from sagemaker.workflow.functions import Join
        
        # Create S3 URI for results.csv using Join function for pipeline properties
        metrics_s3_uri = Join(
            on="",
            values=[training_step.properties.ModelArtifacts.S3ModelArtifacts, "/results.csv"]
        )
        
        # Create model metrics with reference to results.csv
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=metrics_s3_uri,
                content_type="text/csv"
            )
        )
        
        return model_metrics
    
    def validate_model_quality_post_training(self, training_job_name: str) -> tuple[bool, dict]:
        """
        Validate model quality against quality gates after training is complete.
        This can be called after pipeline execution to check actual metrics.
        
        Args:
            training_job_name: Name of the completed training job
            
        Returns:
            Tuple of (passes_gates: bool, validation_report: dict)
        """
            # Get approval configuration
        approval_config = self.registry_config.get('approval', {})
        quality_gates = approval_config.get('quality_gates', {})
        require_all = approval_config.get('require_all_thresholds')
        
        # Use utility function for validation
        return validate_model_quality(training_job_name, quality_gates, require_all)
    
    def update_model_approval_after_training(self, execution) -> dict:
        """
        Method to check quality gates and update model approval after training.
        Call this after pipeline execution completes.
        
        Args:
            execution: SageMaker pipeline execution object
            
        Returns:
            dict: Approval update results
        """
        try:
            # Get training job name from execution
            steps = execution.list_steps()
            training_job_name = None
            
            for step in steps:
                if step['StepName'] == self.training_step_name:
                    if 'Metadata' in step and 'TrainingJob' in step['Metadata']:
                        training_job_name = step['Metadata']['TrainingJob']['Arn'].split('/')[-1]
                        break
            
            if not training_job_name:
                return {'error': 'Could not find training job name from execution'}
            
            # Validate quality gates
            passes_gates, validation_report = self.validate_model_quality_post_training(training_job_name)
            
            # Get approval configuration
            approval_config = self.registry_config.get('approval', {})
            model_package_group = self.registry_config.get('model_package_group_name', 'yolo-model-group')
            
            # determine approval action
            if passes_gates:
                # model passes and has auto approve
                if approval_config.get('auto_approve_on_quality_gates'):
                    # Automatically approve the model
                    approval_description = f"Automatically approved - passed all quality gates: {validation_report['gates_passed']}"
                    try:
                        approval_result = update_model_package_approval(
                            training_job_name=training_job_name,
                            model_package_group_name=model_package_group,
                            approval_status='Approved',
                            approval_description=approval_description
                        )
                        validation_report['approval_result'] = approval_result
                        validation_report['action_taken'] = 'AUTO_APPROVED'
                        validation_report['action_needed'] = 'Model automatically approved - ready for deployment'
                    except Exception as e:
                        print(f"Failed to update model package approval: {e}")
                        validation_report['action_taken'] = 'AUTO_APPROVE_FAILED'
                        validation_report['action_needed'] = f'Failed to auto-approve: {str(e)} - Manual approval required'
                        validation_report['approval_error'] = str(e)
                else:
                    # model passes gates but requires manual approval
                    validation_report['recommendation'] = 'APPROVE'
                    validation_report['action_needed'] = f'Model passed quality gates - ready for manual approval'
                    validation_report['action_taken'] = 'PENDING_MANUAL_APPROVAL'
            else:
                # model failed quality gates and auto reject is true
                if approval_config.get('auto_reject_on_failure'):
                    approval_description = f"Automatically rejected - failed quality gates: {validation_report['gates_passed']}"
                    try:
                        approval_result = update_model_package_approval(
                            training_job_name=training_job_name,
                            model_package_group_name=model_package_group,
                            approval_status='Rejected',
                            approval_description=approval_description
                        )
                        validation_report['approval_result'] = approval_result
                        validation_report['action_taken'] = 'AUTO_REJECTED'
                        validation_report['action_needed'] = 'Model automatically rejected due to failed quality gates'
                    except Exception as e:
                        print(f"Failed to update model package approval: {e}")
                        validation_report['action_taken'] = 'AUTO_REJECT_FAILED'
                        validation_report['action_needed'] = f'Failed to auto-reject: {str(e)} - Manual rejection required'
                        validation_report['approval_error'] = str(e)
                else:
                    # model failed and auto reject is false
                    validation_report['recommendation'] = 'REJECT'
                    validation_report['action_needed'] = f'Model failed quality gates - manual review recommended'
                    validation_report['action_taken'] = 'PENDING_MANUAL_REVIEW'
            
            return validation_report
            
        except Exception as e:
            return {'error': f'Failed to update model approval: {e}'}
    
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
        
        
        return self.pipeline
    
    def upsert_pipeline(self) -> Dict[str, Any]:
        """
        Create or update the pipeline in SageMaker.
        
        Returns:
            Pipeline response from SageMaker
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be created first using create_pipeline()")
        
        # Upsert (create or update) the pipeline
        response = self.pipeline.upsert(role_arn=self.trainer.role_arn)
        
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
        
        # Start execution
        execution = self.pipeline.start(
            execution_display_name=execution_display_name,
            execution_description="YOLO model training execution"
        )
        
        # Extract execution ID from ARN and update config path for traceability
        try:
            execution_id = execution.arn.split('/')[-1]
            # upload config with execution ID
            self.trainer.upload_config_with_execution_id(execution_id)
        except Exception as e:
            print(f"Warning: Could not extract execution ID: {e}")
        
        return execution
    
    def run_pipeline(self, wait_for_completion: bool = False) -> Dict[str, Any]:
        """
        Complete pipeline workflow: create, upsert, and execute.
        
        Args:
            wait_for_completion: Whether to wait for pipeline completion
            
        Returns:
            Execution information
        """
        print("\n" + "="*60)
        print("STARTING PIPELINE EXECUTION")
        print("="*60)
        
        # Create pipeline
        self.create_pipeline()
        
        # Upsert pipeline
        self.upsert_pipeline()
        
        # Start execution
        execution = self.start_execution()
        
        if wait_for_completion:
            print(f"Waiting for pipeline execution to complete...")
            
            # Monitor execution with detailed logging
            self._monitor_execution_with_logs(execution)
            
            # Get final execution status
            status = execution.describe()
            final_status = status.get('PipelineExecutionStatus', 'Unknown')
            
            # Show step results
            self._print_execution_summary(execution)
            
            # Extract and display training metrics if enabled and execution succeeded
            if self.runtime_config.get('display_metrics', True) and final_status == 'Succeeded':
                self._display_training_metrics_from_execution(execution)
            elif final_status == 'Failed':
                print(f"\nPipeline execution failed. Skipping metrics extraction.")
                self._display_training_failure_logs_from_execution(execution)
        else:
            print(f"Wait for completion is false. Pipeline execution started. Monitor progress in SageMaker console.")
            return {
                "pipeline_name": self.pipeline_name,
                "execution_arn": execution.arn,
                "timestamp": self.timestamp,
                "model_package_group": self.model_package_group_name,
                "training_job_name": None,  # Not available when not waiting for completion
                "execution": execution
            }
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*60)
        
        # Extract training job name for result
        training_job_name = self._get_training_job_name_from_execution(execution)
        
        return {
            "pipeline_name": self.pipeline_name,
            "execution_arn": execution.arn,
            "timestamp": self.timestamp,
            "model_package_group": self.model_package_group_name,
            "training_job_name": training_job_name,
            "execution": execution  # Add execution object for quality gate checking
        }
    
    def _monitor_execution_with_logs(self, execution):
        """Monitor pipeline execution with detailed logging."""
        print("="*60)
        print("PIPELINE EXECUTION MONITORING")
        print("="*60)
        
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
                
        print("="*60)
    
    def _print_execution_summary(self, execution):
        """Print detailed execution summary."""
        print("\n" + "="*60)
        print("EXECUTION SUMMARY")
        print("="*60)
        
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
        
        print("="*60)
        
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
        """Extract training job name and display metrics directly."""
        training_job_name = self._get_training_job_name_from_execution(execution)
        if training_job_name:
            display_training_job_metrics(training_job_name, show_metrics=True)
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
        
        # Display pipeline configuration summary
        print(f"\n" + "="*60)
        print("PIPELINE CONFIGURATION")
        print("="*60)
        print(f"Pipeline Name: {pipeline.pipeline_name}")
        print(f"Training Step: {pipeline.training_step_name}")
        print(f"Registration Step: {pipeline.registration_step_name}")
        print(f"Model Package Group: {pipeline.model_package_group_name}")
        print(f"Execution Name: {pipeline.execution_name}")
        print("="*60)
        
        # Get runtime configuration for execution behavior
        runtime_config = pipeline.config.get('runtime', {})
        pipeline_config = pipeline.config.get('pipeline', {})
        dry_run = pipeline_config.get('dry_run')
        wait_for_completion = runtime_config.get('wait_for_completion')
        
        # Display approval configuration
        approval_config = pipeline.registry_config.get('approval', {})
        quality_gates = approval_config.get('quality_gates', {})
        
        print(f"\n" + "="*60)
        print("MODEL APPROVAL CONFIGURATION")
        print("="*60)
        print(f"Auto-approve on quality gates: {approval_config.get('auto_approve_on_quality_gates', False)}")
        print(f"Auto-reject on failure: {approval_config.get('auto_reject_on_failure', False)}")
        print(f"Require all thresholds: {approval_config.get('require_all_thresholds', True)}")
        
        if quality_gates:
            print(f"\nActive Quality Gates:")
            for gate_name, threshold in quality_gates.items():
                print(f"  {gate_name}: {threshold}")
        else:
            print("\nNo quality gates configured - all models will require manual approval")
        print("="*60)
        
        if dry_run:
            print("Dry run mode enabled in config: Creating pipeline without execution")
            pipeline.create_pipeline()
            pipeline.upsert_pipeline()
            print("Pipeline created successfully! Set 'pipeline.dry_run: false' in config to execute.")
            result = {
                "pipeline_name": pipeline.pipeline_name,
                "timestamp": pipeline.timestamp,
                "model_package_group": pipeline.model_package_group_name,
                "training_job_name": None,  # Not available in dry run mode
                "status": "dry_run_completed"
            }
        else:
            # Run complete pipeline
            result = pipeline.run_pipeline(wait_for_completion=wait_for_completion)
            
            # Only show results and quality gates if we waited for completion
            if not wait_for_completion:
                return  # Exit early, don't print final results
            
            # Check quality gates if using quality_gated strategy and pipeline completed
            approval_strategy = pipeline_config.get('approval_strategy', 'quality_gated')
            if approval_strategy == 'quality_gated':
                # Get execution object and check its actual status
                execution = result.get('execution')
                if execution:
                    approval_result = check_and_display_quality_gates(pipeline, execution, pipeline_config)
                    result['quality_gates'] = approval_result
                else:
                    print("Could not access execution for quality gate checking")
        
        print(f"\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print("Job name: ", result.get('training_job_name'))
        for key, value in result.items():
            if key not in ['quality_gates', 'execution']:  # Don't print complex objects
                print(f"{key}: {value}")

        print("="*60)
    
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