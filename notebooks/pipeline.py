import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline_context import PipelineSession

# Please set your account_id
account_id = "503561429929"

# SageMaker session
region = boto3.Session().region_name
# role = f"arn:aws:iam::{account_id}:role/service-role/AmazonSageMaker-ExecutionRole-20250810T153553"
role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole-Pipeline-YOLO"

pipeline_session = PipelineSession()

# Pipeline Parameters
train_data_s3 = f"s3://cyudhist-pipeline-yolo/"  
model_output_s3 = f"s3://cyudhist-yolo-pipeline-models-{account_id}/"
config_s3 = f"s3://cyudhist-yolo-pipeline-scripts/config.yaml"

# Show Metrics for SageMaker console
metric_definitions = [
    {
        "Name": "recall",
        "Regex": r"recall: ([0-9\.]+)"
    },
    {
        "Name": "mAP_0.5",
        "Regex": r"mAP@0\.5: ([0-9\.]+)"
    }
]

# Trainning Estimator（PyTorch）
estimator = PyTorch(
    entry_point="train_yolo.py",
    source_dir="notebooks",
    role=role,
    framework_version="2.0",
    py_version="py310",
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    output_path=model_output_s3,
    sagemaker_session=pipeline_session,
    metric_definitions=metric_definitions,
    dependencies=["notebooks/requirements.txt"]
)

# Step1: trainning
train_step = TrainingStep(
    name="TrainYOLOStep",
    estimator=estimator,
    inputs={
        "training": TrainingInput(train_data_s3),
        "config": TrainingInput(config_s3)
    }
)

# Step2: Register in the Model Registry
register_step = RegisterModel(
    name="RegisterYOLOModelStep",
    estimator=estimator,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/x-onnx"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large", "ml.m5.xlarge"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="YOLOModelPackageGroup"
)

# Pipeline definition
pipeline = Pipeline(
    name="YOLOTrainingPipeline",
    steps=[train_step, register_step],
    sagemaker_session=pipeline_session
)

if __name__ == "__main__":
    # Pipeline Creation & Execution
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print(f"Started pipeline execution: {execution.arn}")
