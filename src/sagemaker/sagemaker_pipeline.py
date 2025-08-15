"""
Comprehensive SageMaker Pipeline for YOLO Model Training and Registration.
"""

import os
import boto3
import time
from datetime import datetime
from typing import Dict, Any, Optional


import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, CacheConfig
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model_metrics import ModelMetrics, MetricsSource


from utils.utils_config import (
    load_config,
    get_aws_config,
    get_data_config,
    get_training_config,
    get_hyperparameters_config,
    get_validation_config
)
from sagemaker_metrics import display_training_job_metrics


class YOLOSageMakerPipeline:
    
    def __init__(self, config_path: str = "config.yaml"):

        self.config = load_config(config_path)
        
        # Extract configurations
        self.aws_config = get_aws_config(self.config)
        self.data_config = get_data_config(self.config)
        self.training_config = get_training_config(self.config)
        self.hyperparams_config = get_hyperparameters_config(self.config)
        self.pipeline_config = self.config.get('pipeline', {})
        self.runtime_config = self.config.get('runtime', {})
        self.validation_config = get_validation_config(self.config)
        
        self._setup_aws_components()
        
        # setup pipeline naming with timestamp for uniqueness
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._setup_pipeline_names()
        
        self._setup_s3_paths()
        
        # initialize pipeline components
        self.pipeline_session = PipelineSession()
        self.estimator = None
        self.pipeline = None
        
        print(f"Initialized YOLO SageMaker Pipeline with timestamp: {self.timestamp}")
        print(f"Pipeline name: {self.pipeline_name}")
        print(f"Model package group: {self.model_package_group_name}")
    
    def _setup_aws_components(self):
        """
        "Setup AWS session, region, and role.
        TODO: this is being reproduced in all other files. maybe create another class for it
        
        """
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
    
    def _setup_pipeline_names(self):
        # Get custom names from config or use defaults
        pipeline_config = self.config.get('pipeline', {})
        
        self.pipeline_name = pipeline_config.get('name', "yolo-training-pipeline-object-detection")
        
        self.training_step_name = pipeline_config.get('training_step_name', "YOLOTrainingStep-ObjectDetection")
        
        self.registration_step_name = pipeline_config.get('registration_step_name', "YOLOModelRegistrationStep-ObjectDetection")
        
        self.model_package_group_name = pipeline_config.get('model_package_group_name', "yolo-model-package-group-object-detection")
        
        self.execution_name = f"yolo-pipeline-execution-object-detection-{self.timestamp}"
        
        # CloudWatch log group names
        self.training_log_group = f"/aws/sagemaker/TrainingJobs/yolo-training-object-detection"
        
        print(f"Pipeline naming setup completed:")
        print(f"  Pipeline: {self.pipeline_name}")
        print(f"  Training Step: {self.training_step_name}")
        print(f"  Registration Step: {self.registration_step_name}")
        print(f"  Model Package Group: {self.model_package_group_name}")
    
    def _setup_s3_paths(self):
        """Setup S3 paths for training data and model outputs."""
        # training data path
        self.s3_training_data = (
            self.data_config.get('s3_dataset_prefix') 
            or self.data_config.get('s3_train_prefix')
        )
        
        if not self.s3_training_data:
            raise ValueError(
                "Training data S3 path must be specified in config.yaml under "
                "'data.s3_dataset_prefix' or 'data.s3_train_prefix'"
            )
        
        # model output path
        self.s3_model_output = f"s3://{self.bucket}/{self.prefix}/models/{self.timestamp}"
        
        # code output path
        self.s3_code_output = f"s3://{self.bucket}/{self.prefix}/code/{self.timestamp}"
        
        print(f"S3 Paths configured:")
        print(f"  Training data: {self.s3_training_data}")
        print(f"  Model output: {self.s3_model_output}")
        print(f"  Code output: {self.s3_code_output}")
    
    def create_estimator(self) -> PyTorch:
        # get training configuration
        instance_type = self.training_config.get('instance_type', 'ml.g4dn.xlarge')
        instance_count = self.training_config.get('instance_count', 1)
        volume_size = self.training_config.get('volume_size', 50)
        max_run = self.training_config.get('max_run', 86400)  # 24 hours default
        
        # metric definitions for CloudWatch
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
        
        # create PyTorch estimator
        self.estimator = PyTorch(
            entry_point="yolo_entrypoint.py",
            source_dir="src/sagemaker",
            role=self.role_arn,
            framework_version="2.0",
            py_version="py310",
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size=volume_size,
            max_run=max_run,
            output_path=self.s3_model_output,
            code_location=self.s3_code_output,
            sagemaker_session=self.pipeline_session,
            metric_definitions=metric_definitions,
            dependencies=["src/sagemaker/dependencies/requirements.txt"],
            base_job_name=f"yolo-training-{self.timestamp}"
        )
        
        # set hyperparameters from config
        self._set_hyperparameters()
        
        print(f"Created PyTorch estimator:")
        print(f"  Instance type: {instance_type}")
        print(f"  Instance count: {instance_count}")
        print(f"  Volume size: {volume_size} GB")
        print(f"  Max runtime: {max_run} seconds")
        
        return self.estimator
    
    def _set_hyperparameters(self):
        #TODO: has to parameterize for keypoint detection
        """Set hyperparameters for the training job from configuration."""
        # Build hyperparameters from config
        hyperparams = {
            "model_name": self.hyperparams_config.get('model_name', 'yolo11n.pt'),
            "image_size": self.training_config.get('image_size', 640),
            "epochs": self.training_config.get('epochs', 100),
            "batch_size": self.training_config.get('batch_size', 16),
            "lr0": self.training_config.get('lr0', 0.01),
            "lrf": self.training_config.get('lrf', 0.1),
            "momentum": self.training_config.get('momentum', 0.937),
            "weight_decay": self.training_config.get('weight_decay', 0.0005),
            "warmup_epochs": self.training_config.get('warmup_epochs', 3.0),
            "cos_lr": self.training_config.get('cos_lr', True),
            "optimizer": self.training_config.get('optimizer', 'AdamW'),
            "hsv_h": self.training_config.get('hsv_h', 0.015),
            "hsv_s": self.training_config.get('hsv_s', 0.7),
            "hsv_v": self.training_config.get('hsv_v', 0.4),
            "degrees": self.training_config.get('degrees', 10.0),
            "translate": self.training_config.get('translate', 0.1),
            "scale": self.training_config.get('scale', 0.5),
            "fliplr": self.training_config.get('fliplr', 0.5),
            "mosaic": self.training_config.get('mosaic', 1.0),
            "mixup": self.training_config.get('mixup', 0.1),
            "box": self.training_config.get('box', 7.5),
            "cls": self.training_config.get('cls', 0.5),
            "dfl": self.training_config.get('dfl', 1.5),
            "label_smoothing": self.training_config.get('label_smoothing', 0.1),
            "patience": self.training_config.get('patience', 50),
            "dropout": self.training_config.get('dropout', 0.1),
            "amp": self.training_config.get('amp', True),
        }

        # Total Loss = 7.5 × Box Loss + 0.5 × Class Loss + 1.5 × DFL Loss
        
        self.estimator.set_hyperparameters(**hyperparams)
        
        print("Set hyperparameters for training:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
    
    def _prepare_config_input(self) -> TrainingInput:
        """
        Upload config.yaml to S3 and prepare it as a training input.
        
        Returns:
            TrainingInput for config.yaml, or None if config not found

        NOTE: We need to upload config.yaml each time we run pipeline as we have may have params changes
        """

        # Check if config.yaml exists locally
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            print("config.yaml not found locally - training will use command-line args only")
            return None
        
        try:
            # upload config.yaml to S3 (in a directory for TrainingInput)
            s3_config_key = f"{self.prefix}/config/{self.timestamp}/config.yaml"
            s3_config_dir = f"s3://{self.bucket}/{self.prefix}/config/{self.timestamp}/"
            
            s3_client = boto3.client('s3', region_name=self.region)
            s3_client.upload_file(config_path, self.bucket, s3_config_key)
            
            print(f"Uploaded config.yaml to: s3://{self.bucket}/{s3_config_key}")
            
            # create TrainingInput for config directory
            config_input = TrainingInput(
                s3_data=s3_config_dir,
                distribution="FullyReplicated",
                s3_data_type="S3Prefix"
            )
            
            return config_input
            
        except Exception as e:
            print(f"Failed to upload config.yaml to S3: {e}")
            print("Training will use command-line args only")
            return None
    
    def create_training_step(self) -> TrainingStep:
        if self.estimator is None:
            raise ValueError("Estimator must be created first using create_estimator()")
        
        # prepare training inputs
        training_input = TrainingInput(
            s3_data=self.s3_training_data,
            distribution="FullyReplicated",
            s3_data_type="S3Prefix"
        )
        
        # upload config.yaml to S3 and create config input
        config_input = self._prepare_config_input()
        
        # configure caching based on config
        cache_config = None
        if self.pipeline_config.get('enable_caching', False):
            from sagemaker.workflow.pipeline_context import PipelineSession
            cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")  # 1 hour cache
        
        # create training step with both data and config inputs
        inputs = {"training": training_input}
        if config_input:
            inputs["config"] = config_input
        
        training_step = TrainingStep(
            name=self.training_step_name,
            estimator=self.estimator,
            inputs=inputs,
            cache_config=cache_config,
        )
        
        print(f"Created training step: {self.training_step_name}")
        print(f"  Training data input: {self.s3_training_data}")
        if config_input:
            print(f"  Config input: Available")
        else:
            print(f"  Config input: Not available (using command-line args)")
        
        return training_step
    
    def create_model_registration_step(self, training_step: TrainingStep) -> RegisterModel:
        """
        Create the model registration step for the pipeline.
        
        Args:
            training_step: The training step that produces the model
            
        Returns:
            Configured model registration step
        """
        # determine approval status based on config
        auto_approve = self.pipeline_config.get('auto_approve_models', False)
        approval_status = "Approved" if auto_approve else "PendingManualApproval"
        
        # create model registration step
        registration_step = RegisterModel(
            name=self.registration_step_name,
            estimator=self.estimator,
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["application/x-onnx", "application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large", "ml.m5.xlarge", "ml.g4dn.xlarge"], #TODO: need to check this
            transform_instances=["ml.m5.large", "ml.m5.xlarge"], #TODO: need to check this
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
        
        These metrics help evaluate model performance and are used by SageMaker
        Model Registry for model comparison and approval workflows.
        
        Returns:
            ModelMetrics object with YOLO-specific validation metrics
        """
        
        model_statistics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=f"{self.s3_model_output}/evaluation.json",  # Validation results
                content_type="application/json"
            ),
            model_constraints=MetricsSource(
                s3_uri=f"{self.s3_model_output}/constraints.json",  # Model constraints
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
        # create estimator
        self.create_estimator()
        
        # create training step
        training_step = self.create_training_step()
        
        # create model registration step
        registration_step = self.create_model_registration_step(training_step)
        
        # create pipeline
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
        
        # upsert (create or update) the pipeline
        response = self.pipeline.upsert(role_arn=self.role_arn)
        
        print(f"Pipeline upserted successfully!")
        print(f"  Pipeline ARN: {response.get('PipelineArn', 'N/A')}")
        
        return response
    
    def start_execution(self, execution_display_name: Optional[str] = None) -> Dict[str, Any]:

        if self.pipeline is None:
            raise ValueError("Pipeline must be created first using create_pipeline()")
        
        # set default execution name if not provided
        if execution_display_name is None:
            execution_display_name = self.execution_name
        
        print(f"Starting pipeline execution: {execution_display_name}")
        
        # start execution
        execution = self.pipeline.start(
            execution_display_name=execution_display_name,
            execution_description="YOLO model training execution"
        )
        
        print(f"Pipeline execution started successfully!")
        print(f"  Execution ARN: {execution.arn}")
        print(f"\nMonitoring Information:")
        print(f"  Pipeline Name: {self.pipeline_name}")
        print(f"  Execution Name: {execution_display_name}")
        print(f"  Region: {self.region}")
        print(f"  CloudWatch Logs: {self.training_log_group}")
        
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
        
        # create pipeline
        self.create_pipeline()
        
        # upsert pipeline
        self.upsert_pipeline()
        
        # start execution
        execution = self.start_execution()
        
        if wait_for_completion:
            print(f"\nWaiting for pipeline execution to complete...")
            print(f"You can monitor progress at:")
            print(f"  SageMaker Console: https://{self.region}.console.aws.amazon.com/sagemaker/home?region={self.region}#/pipelines/{self.pipeline_name}/executions/{self.execution_name}")
            
            # monitor execution with detailed logging
            self._monitor_execution_with_logs(execution)
            
            print(f"Pipeline execution completed!")
            
            # get final execution status
            status = execution.describe()
            final_status = status.get('PipelineExecutionStatus', 'Unknown')
            print(f"Final status: {final_status}")
            
            # show step results
            self._print_execution_summary(execution)
            
            # get final execution status
            status = execution.describe()
            execution_status = status.get('PipelineExecutionStatus', 'Unknown')
            
            # extract and display training metrics if enabled and execution succeeded
            if self.runtime_config.get('display_metrics', True) and execution_status == 'Succeeded':
                self._display_training_metrics(execution)
            elif execution_status == 'Failed':
                print(f"\nPipeline execution failed. Skipping metrics extraction.")
                print(f"Extracting training job logs to diagnose the failure...")
                self._display_training_failure_logs(execution)
        else:
            print(f"\nPipeline execution started. Monitor progress in SageMaker console.")
            print(f"  SageMaker Console: https://{self.region}.console.aws.amazon.com/sagemaker/home?region={self.region}#/pipelines/{self.pipeline_name}/executions/{self.execution_name}")
        
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

        print("="*50)
        print("PIPELINE EXECUTION MONITORING")
        print("="*50)
        
        last_status = None
        step_statuses = {}
        
        while True:
            try:
                # get current execution status
                current_status = execution.describe()
                execution_status = current_status.get('PipelineExecutionStatus', 'Unknown')
                
                # print execution status change
                if execution_status != last_status:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{timestamp}] Pipeline Status: {execution_status}")
                    last_status = execution_status
                
                # get step statuses
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
                                    print(f"    CloudWatch Logs: https://{self.region}.console.aws.amazon.com/cloudwatch/home?region={self.region}#logsV2:log-groups/log-group/%2Faws%2Fsagemaker%2FTrainingJobs/log-events/{job_name}")
                            
                            step_statuses[step_name] = step_status
                
                except Exception as e:
                    print(f"Warning: Could not get step details: {e}")
                
                # check if execution is complete
                if execution_status in ['Succeeded', 'Failed', 'Stopped']:
                    break
                    
                # wait before next check
                time.sleep(30)  # check every 30 seconds
                
            except KeyboardInterrupt:
                print(f"\nMonitoring interrupted by user. Pipeline continues running.")
                print(f"Check SageMaker console for status updates.")
                break
            except Exception as e:
                print(f"Error monitoring execution: {e}")
                time.sleep(30)
                
        print("="*50)
    
    def _print_execution_summary(self, execution):
        print("\n" + "="*50)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*50)
        
        try:
            # get execution details
            status = execution.describe()
            execution_status = status.get('PipelineExecutionStatus', 'Unknown')
            start_time = status.get('CreationTime', 'Unknown')
            end_time = status.get('LastModifiedTime', 'Unknown')
            
            print(f"Execution Status: {execution_status}")
            print(f"Start Time: {start_time}")
            print(f"End Time: {end_time}")
            
            # get step details
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
                
                # show training job details if available
                metadata = step.get('Metadata', {})
                if 'TrainingJob' in metadata:
                    training_job = metadata['TrainingJob']
                    training_job_arn = training_job.get('Arn', '')
                    if training_job_arn:
                        job_name = training_job_arn.split('/')[-1]
                        print(f"  Training Job: {job_name}")
                
                # show model registration details if available
                if 'RegisterModel' in metadata:
                    model_package_arn = metadata['RegisterModel'].get('Arn', '')
                    if model_package_arn:
                        print(f"  Model Package: {model_package_arn}")
                
                print()
                
        except Exception as e:
            print(f"Error getting execution summary: {e}")
        
        print("="*50)
    
    def _display_training_failure_logs(self, execution):
        """
        Extract and display training job logs when pipeline execution fails.
        This helps diagnose why the training step failed.
        """
        try:
            print("\n" + "="*60)
            print("TRAINING FAILURE DIAGNOSIS")
            print("="*60)
            
            # get the training job name from the execution steps
            training_job_name = None
            try:
                steps = execution.list_steps()
                for step in steps:
                    if 'train' in step.get('StepName', '').lower():
                        metadata = step.get('Metadata', {})
                        if 'TrainingJob' in metadata:
                            training_job_arn = metadata['TrainingJob'].get('Arn', '')
                            if training_job_arn:
                                training_job_name = training_job_arn.split('/')[-1]
                                break
            except Exception as e:
                print(f"Warning: Could not extract training job name from pipeline: {e}")
            
            if training_job_name:
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
                
            else:
                print("Could not find training job name from pipeline execution.")
                print("Check the SageMaker console for training job details.")
        
        except Exception as e:
            print(f"Error displaying training failure logs: {e}")
            print("Check the SageMaker console manually for training job details.")

    def _display_training_metrics(self, execution):
        try:
            print("\n" + "="*60)
            print("EXTRACTING TRAINING METRICS")
            print("="*60)
            
            # get the training job name from the execution steps
            training_job_name = None
            try:
                steps = execution.list_steps()
                for step in steps:
                    if 'train' in step.get('StepName', '').lower():
                        metadata = step.get('Metadata', {})
                        if 'TrainingJob' in metadata:
                            training_job_arn = metadata['TrainingJob'].get('Arn', '')
                            if training_job_arn:
                                training_job_name = training_job_arn.split('/')[-1]
                                break
            except Exception as e:
                print(f"Warning: Could not extract training job name from pipeline: {e}")
            
            if training_job_name:
                print(f"Found training job: {training_job_name}")
                print("Extracting metrics...")
                print("-" * 60)

                display_training_job_metrics(training_job_name)

            else:
                print("Could not find training job name from pipeline execution.")
                print("You can manually check metrics using:")
                print("  python src/sagemaker/sagemaker_metrics.py [training_job_name]")
        
        except ImportError as e:
            print(f"Could not import sagemaker_metrics: {e}")
            print("Please ensure sagemaker_metrics.py is available.")
        except Exception as e:
            print(f"Error displaying training metrics: {e}")
            print("You can manually check metrics using:")
            print("  python src/sagemaker/sagemaker_metrics.py [training_job_name]")



def run_yolo_pipeline(config_path: str = "config.yaml") -> Dict[str, Any]:

    print(f"Loading configuration from: {config_path}")
    pipeline = YOLOSageMakerPipeline(config_path=config_path)
    
    # get runtime configuration for execution behavior
    runtime_config = pipeline.config.get('runtime', {})
    pipeline_config = pipeline.config.get('pipeline', {})
    
    # check if this is a dry run
    dry_run = pipeline_config.get('dry_run', False)
    
    # check if we should wait for completion
    wait_for_completion = runtime_config.get('wait_for_completion', False)
    
    if dry_run:
        print("Dry run mode enabled in config: Creating pipeline without execution")
        pipeline.create_pipeline()
        pipeline.upsert_pipeline()
        print("Pipeline created successfully! Set 'pipeline.dry_run: false' in config to execute.")
        return {
            "pipeline_name": pipeline.pipeline_name,
            "timestamp": pipeline.timestamp,
            "model_package_group": pipeline.model_package_group_name,
            "status": "dry_run_completed"
        }
    else:
        # run complete pipeline
        result = pipeline.run_pipeline(wait_for_completion=wait_for_completion)
        return result


def main():

    config_path = os.environ.get("YOLO_CONFIG_PATH", "config.yaml")
    
    try:
        result = run_yolo_pipeline(config_path)
        
        print(f"\nPipeline Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
    
    
    # Example:
    """
    This script creates and executes a complete SageMaker pipeline that:
    1. Trains a YOLO model using the configured dataset
    2. Registers the trained model in the SageMaker Model Registry
    3. Can be run standalone with proper configuration

    The pipeline integrates with the existing sagemaker_trainer.py and yolo_entrypoint.py
    components for a complete end-to-end training workflow.

    Usage Examples:

    1. Run directly (uses config.yaml):
    python src/sagemaker/sagemaker_pipeline.py

    2. Use with custom config path (via environment variable):
    export YOLO_CONFIG_PATH=/path/to/custom/config.yaml
    python src/sagemaker/sagemaker_pipeline.py

    # Simple usage - just run with default config
    from src.sagemaker.sagemaker_pipeline import run_yolo_pipeline
    result = run_yolo_pipeline()

    # Advanced usage - create pipeline object for more control
    from src.sagemaker.sagemaker_pipeline import YOLOSageMakerPipeline

    pipeline = YOLOSageMakerPipeline("config.yaml")
    pipeline.create_pipeline()
    pipeline.upsert_pipeline()
    execution = pipeline.start_execution()
    """
