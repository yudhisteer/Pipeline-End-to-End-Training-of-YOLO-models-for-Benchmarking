"""
Comprehensive SageMaker Pipeline for YOLO Model Training and Registration.
Uses the standalone YOLOSageMakerTrainer for training functionality.
"""

import os
import time
import json
import boto3
from datetime import datetime
from typing import Dict, Any, Optional

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, CacheConfig
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model_metrics import ModelMetrics, MetricsSource

from utils.utils_config import load_config, get_validation_config
from entrypoint_trainer import YOLOSageMakerTrainer
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
        inputs = self.trainer.prepare_training_inputs(execution_id=None)
        
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
        # Determine approval status based on quality gates and config
        meets_quality_gates, gate_reason = self._check_quality_gates()
        approval_strategy = self.pipeline_config.get('approval_strategy', 'quality_gated')
        
        if approval_strategy == "always":
            approval_status = "Approved"
            approval_reason = "Always approve strategy enabled"
        elif approval_strategy == "never":
            approval_status = "PendingManualApproval"
            approval_reason = "Never approve strategy - manual review required"
        else:  # quality_gated
            if meets_quality_gates:
                approval_status = "Approved"
                approval_reason = f"Quality gates passed: {gate_reason}"
            else:
                approval_status = "PendingManualApproval" 
                approval_reason = f"Quality gates check: {gate_reason}"
        
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
        print(f"  Approval strategy: {approval_strategy}")
        print(f"  Approval status: {approval_status}")
        print(f"  Approval reason: {approval_reason}")
        
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
    
    def _check_quality_gates(self) -> tuple[bool, str]:
        """
        Check if the model meets quality gates for auto-approval.
        
        Returns:
            Tuple of (meets_gates: bool, reason: str)
        """
        try:
            # Get approval configuration
            approval_config = self.config.get('approval', {})
            quality_gates = approval_config.get('quality_gates', {})
            
            if not quality_gates:
                print("Warning: No quality gates configured, defaulting to manual approval")
                return False, "No quality gates configured"
            
            # Try to load the evaluation metrics from the training output
            evaluation_path = f"{self.trainer.s3_model_output}/evaluation.json"
            
            # For now, we'll implement a basic check
            # In a full implementation, you'd download and parse the evaluation.json
            print(f"Quality gates configured:")
            for gate, threshold in quality_gates.items():
                print(f"  {gate}: {threshold}")
            
            # Since we can't easily access the training metrics here during pipeline creation,
            # we'll return based on the approval strategy
            approval_strategy = self.pipeline_config.get('approval_strategy', 'quality_gated')
            
            if approval_strategy == "always":
                return True, "Always approve strategy"
            elif approval_strategy == "never":
                return False, "Never approve strategy"
            else:  # quality_gated
                # For quality_gated, register as pending and validate after training
                return False, "Quality gates will be checked after training completes"
                    
        except Exception as e:
            print(f"Error checking quality gates: {e}")
            return False, f"Error checking quality gates: {e}"
    
    def validate_model_quality_post_training(self, training_job_name: str) -> tuple[bool, dict]:
        """
        Validate model quality against quality gates after training is complete.
        This can be called after pipeline execution to check actual metrics.
        
        Args:
            training_job_name: Name of the completed training job
            
        Returns:
            Tuple of (passes_gates: bool, validation_report: dict)
        """
        try:
            # Get approval configuration
            approval_config = self.config.get('approval', {})
            quality_gates = approval_config.get('quality_gates', {})
            require_all = approval_config.get('require_all_thresholds', True)
            
            # Get training job details to find S3 output path
            sagemaker_client = boto3.client('sagemaker')
            job_details = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
            output_path = job_details['OutputDataConfig']['S3OutputPath']
            
            # Construct path to model.tar.gz (not evaluation.json directly)
            s3_model_path = f"{output_path}/{training_job_name}/output/model.tar.gz"
            
            # Download and extract model.tar.gz to find evaluation.json
            s3_client = boto3.client('s3')
            # Parse S3 path
            bucket = output_path.split('/')[2]
            key = '/'.join(output_path.split('/')[3:]) + f"/{training_job_name}/output/model.tar.gz"
            
            try:
                print(f"Attempting to download model.tar.gz from s3://{bucket}/{key}")
                
                # Download tar.gz file
                import tempfile
                import tarfile
                with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tar_tmp_file:
                    s3_client.download_fileobj(bucket, key, tar_tmp_file)
                    tar_tmp_file.flush()
                    
                    # Extract evaluation.json from tar.gz
                    with tarfile.open(tar_tmp_file.name, 'r:gz') as tar:
                        # Look for evaluation.json in the tar file
                        evaluation_member = None
                        for member in tar.getmembers():
                            if member.name.endswith('evaluation.json') or member.name == 'evaluation.json':
                                evaluation_member = member
                                break
                        
                        if evaluation_member is None:
                            raise FileNotFoundError("evaluation.json not found in the tar.gz file")
                        
                        print(f"Found {evaluation_member.name} in tar.gz")
                        
                        # Extract and read evaluation.json
                        extracted_file = tar.extractfile(evaluation_member)
                        if extracted_file is None:
                            raise RuntimeError(f"Could not extract {evaluation_member.name}")
                        
                        evaluation_data = json.loads(extracted_file.read().decode('utf-8'))
                        metrics = evaluation_data.get('metrics', {})
                        model_info = evaluation_data.get('model_info', {})
                        print(f"Successfully loaded evaluation data with metrics: {list(metrics.keys())}")
                    
                    # Clean up temp file
                    import os
                    os.unlink(tar_tmp_file.name)
                
                # Check each quality gate dynamically
                validation_results = {}
                passes_count = 0
                total_gates = len(quality_gates)
                
                # Mapping of gate names to metric extraction logic
                gate_mappings = {
                    "min_mAP_0_5": lambda: (metrics.get('mAP_0.5', 0.0), ">="),
                    "min_mAP_0_5_0_95": lambda: (metrics.get('mAP_0.5_0.95', 0.0), ">="), 
                    "min_precision": lambda: (metrics.get('precision', 0.0), ">="),
                    "min_recall": lambda: (metrics.get('recall', 0.0), ">="),
                    "min_f1_score": lambda: (metrics.get('f1_score', 0.0), ">="),
                    "max_model_size_mb": lambda: (model_info.get('model_size_mb', float('inf')), "<="),
                    "max_inference_time_ms": lambda: (model_info.get('inference_time_ms', 100), "<="),
                    "min_speed_fps": lambda: (model_info.get('speed_fps', 0.0), ">="),
                }
                
                for gate_name, threshold in quality_gates.items():
                    if gate_name in gate_mappings:
                        actual_value, operator = gate_mappings[gate_name]()
                        
                        if operator == ">=":
                            passes = actual_value >= threshold
                        elif operator == "<=":
                            passes = actual_value <= threshold
                        else:
                            passes = False
                    else:
                        # Unknown gate - skip with warning
                        print(f"Warning: Unknown quality gate '{gate_name}' - skipping")
                        actual_value = None
                        passes = True  # Don't fail on unknown gates
                        total_gates -= 1  # Don't count unknown gates
                    
                    validation_results[gate_name] = {
                        'threshold': threshold,
                        'actual': actual_value,
                        'passes': passes
                    }
                    
                    if passes:
                        passes_count += 1
                
                # Determine overall pass/fail
                if require_all:
                    overall_pass = passes_count == total_gates
                else:
                    overall_pass = passes_count > 0
                
                validation_report = {
                    'training_job_name': training_job_name,
                    'overall_pass': overall_pass,
                    'strategy': 'require_all' if require_all else 'require_any',
                    'gates_passed': f"{passes_count}/{total_gates}",
                    'detailed_results': validation_results,
                    'evaluation_path': s3_model_path
                }
                
                return overall_pass, validation_report
                
            except Exception as e:
                print(f"Failed to extract evaluation.json from model.tar.gz: {e}")
                print(f"Tried downloading: s3://{bucket}/{key}")
                return False, {
                    'error': f"Could not load evaluation metrics from model.tar.gz: {e}",
                    'training_job_name': training_job_name,
                    'attempted_path': f"s3://{bucket}/{key}"
                }
                
        except Exception as e:
            return False, {
                'error': f"Validation failed: {e}",
                'training_job_name': training_job_name
            }
    
    def update_model_approval_after_training(self, execution) -> dict:
        """
        Simple method to check quality gates and update model approval after training.
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
            
            if passes_gates:
                # Update model package to approved status
                # Find the registered model package
                approval_config = self.config.get('approval', {})
                if approval_config.get('allow_manual_override', True):
                    # For simplicity, just return the validation report
                    # In practice, you'd update the model package status here
                    validation_report['recommendation'] = 'APPROVE'
                    validation_report['action_needed'] = f'Manually approve model package for training job: {training_job_name}'
                else:
                    validation_report['recommendation'] = 'REJECT'
                    validation_report['action_needed'] = f'Model failed quality gates - manual override disabled'
            else:
                validation_report['recommendation'] = 'REJECT'
                validation_report['action_needed'] = f'Model failed quality gates - keep as pending manual approval'
            
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
        print(f"Execution ARN: {execution.arn}")
        
        # Extract execution ID from ARN and update config path for traceability
        try:
            execution_id = execution.arn.split('/')[-1]
            print(f"Execution ID: {execution_id}")
            
            # upload config with execution ID
            if self.trainer.upload_config_with_execution_id(execution_id):
                print(f"Config uploaded successfully!")
            else:
                print(f"Config upload failed")
                
        except Exception as e:
            print(f"Warning: Could not extract execution ID: {e}")
        
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
            "model_package_group": self.model_package_group_name,
            "execution": execution  # Add execution object for quality gate checking
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
            
            # Check quality gates if using quality_gated strategy and pipeline completed
            approval_strategy = pipeline_config.get('approval_strategy', 'quality_gated')
            if approval_strategy == 'quality_gated' and wait_for_completion:
                # Get execution object and check its actual status
                execution = result.get('execution')
                if execution:
                    try:
                        execution_status = execution.describe().get('PipelineExecutionStatus', 'Unknown')
                        if execution_status == 'Succeeded':
                            print(f"\nChecking quality gates for approval...")
                            approval_result = pipeline.update_model_approval_after_training(execution)
                            print(f"\nQuality Gate Results:")
                            if approval_result.get('overall_pass'):
                                print(f"Status: PASSED")
                                print(f"Gates passed: {approval_result.get('gates_passed', 'N/A')}")
                                print(f"Recommendation: {approval_result.get('recommendation', 'N/A')}")
                            else:
                                print(f"Status: FAILED")
                                print(f"Gates passed: {approval_result.get('gates_passed', 'N/A')}")
                                print(f"Recommendation: {approval_result.get('recommendation', 'N/A')}")
                            
                            # Display quality gate comparison table
                            detailed_results = approval_result.get('detailed_results', {})
                            if detailed_results:
                                print("\nQuality Gate Comparison:")
                                print("-" * 70)
                                print(f"{'Gate':<20} {'Required':<12} {'Actual':<12} {'Status':<10} {'Gap':<10}")
                                print("-" * 70)
                                
                                for gate_name, gate_info in detailed_results.items():
                                    threshold = gate_info.get('threshold', 'N/A')
                                    actual = gate_info.get('actual', 'N/A')
                                    passes = gate_info.get('passes', False)
                                    
                                    # Format values
                                    if isinstance(threshold, (int, float)) and isinstance(actual, (int, float)):
                                        threshold_str = f"{threshold:.3f}"
                                        actual_str = f"{actual:.3f}"
                                        gap = actual - threshold if gate_name.startswith('min_') else threshold - actual
                                        gap_str = f"{gap:+.3f}"
                                    else:
                                        threshold_str = str(threshold)
                                        actual_str = str(actual)
                                        gap_str = "N/A"
                                    
                                    status_str = "✅ PASS" if passes else "❌ FAIL"
                                    display_name = gate_name.replace('min_', '').replace('_', ' ').title()
                                    
                                    print(f"{display_name:<20} {threshold_str:<12} {actual_str:<12} {status_str:<10} {gap_str:<10}")
                                
                                print("-" * 70)
                            
                            print(f"Action needed: {approval_result.get('action_needed', 'N/A')}")
                            result['quality_gates'] = approval_result
                        else:
                            print(f"\nPipeline status is {execution_status} - skipping quality gate checks")
                    except Exception as e:
                        print(f"Quality gate checking failed: {e}")
                else:
                    print("Could not access execution for quality gate checking")
        
        print(f"\nPipeline Results:")
        for key, value in result.items():
            if key != 'quality_gates':  # Don't print the full quality gates dict
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