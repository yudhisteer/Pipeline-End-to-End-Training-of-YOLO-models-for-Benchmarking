"""
Minimal SageMaker Pipeline for YOLO evaluation.
"""

import os
from datetime import datetime
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

class MinimalYOLOEvaluationPipeline:
    """Minimal YOLO evaluation pipeline."""
    
    def __init__(self, 
                 bucket: str,
                 role_arn: str,
                 region: str = "us-east-1",
                 prefix: str = "yolo-eval"):
        
        self.bucket = bucket
        self.role_arn = role_arn
        self.region = region
        self.prefix = prefix
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Pipeline components
        self.pipeline_session = PipelineSession()
        self.pipeline_name = f"yolo-eval-{self.timestamp}"
        
        print(f"Minimal pipeline initialized: {self.pipeline_name}")
    
    def create_pipeline(self, 
                       test_data_s3: str,
                       trained_model_s3: str) -> Pipeline:
        """Create minimal evaluation pipeline."""
        
        # Create processor
        processor = ScriptProcessor(
            image_uri=self._get_image_uri(),
            role=self.role_arn,
            instance_type="ml.m5.large",
            instance_count=1,
            volume_size_in_gb=20,
            max_runtime_in_seconds=1800,  # 30 minutes
            base_job_name=f"yolo-eval-{self.timestamp}",
            sagemaker_session=self.pipeline_session,
            command=["python"]
        )
        
        # Define inputs - use different destinations to avoid conflicts
        inputs = [
            ProcessingInput(
                source=test_data_s3,
                destination="/opt/ml/processing/test_data",
                input_name="test_data"
            ),
            ProcessingInput(
                source=trained_model_s3,
                destination="/opt/ml/processing/trained_model",
                input_name="trained_model"
            )
        ]
        
        # Define output
        output_s3 = f"s3://{self.bucket}/{self.prefix}/results/{self.timestamp}"
        outputs = [
            ProcessingOutput(
                source="/opt/ml/processing/evaluation",
                destination=output_s3,
                output_name="results"
            )
        ]
        
        # Create processing step
        eval_step = ProcessingStep(
            name="YOLOEvaluation",
            processor=processor,
            inputs=inputs,
            outputs=outputs,
            code="src/sagemaker/entrypoint_evaluation.py"  # Make sure this file exists
        )
        
        # Create pipeline
        self.pipeline = Pipeline(
            name=self.pipeline_name,
            steps=[eval_step],
            sagemaker_session=self.pipeline_session,
        )
        
        print(f"Created pipeline with inputs:")
        print(f"  Test data: {test_data_s3}")
        print(f"  Trained model: {trained_model_s3}")
        print(f"  Output: {output_s3}")
        
        return self.pipeline
    
    def _get_image_uri(self) -> str:
        """Get container image."""
        from sagemaker import image_uris
        
        return image_uris.retrieve(
            framework="pytorch",
            region=self.region,
            version="2.1",
            py_version="py310",
            instance_type="ml.m5.large",
            accelerator_type=None,
            image_scope="training"
        )
    
    def run_evaluation(self,
                      test_data_s3: str,
                      trained_model_s3: str,
                      wait: bool = False) -> dict:
        """Run the evaluation pipeline."""
        
        print("=" * 50)
        print("STARTING MINIMAL YOLO EVALUATION")
        print("=" * 50)
        
        # Create pipeline
        pipeline = self.create_pipeline(test_data_s3, trained_model_s3)
        
        # Upsert pipeline
        print("Creating pipeline...")
        pipeline.upsert(role_arn=self.role_arn)
        
        # Start execution
        print("Starting execution...")
        execution = pipeline.start(
            execution_display_name=f"eval-{self.timestamp}"
        )
        
        print(f"Execution started: {execution.arn}")
        
        if wait:
            print("Waiting for completion...")
            while True:
                status = execution.describe()
                exec_status = status.get('PipelineExecutionStatus', 'Unknown')
                print(f"Status: {exec_status}")
                
                if exec_status in ['Succeeded', 'Failed', 'Stopped']:
                    break
                
                import time
                time.sleep(30)
        
        result = {
            'pipeline_name': self.pipeline_name,
            'execution_arn': execution.arn,
            'output_location': f"s3://{self.bucket}/{self.prefix}/results/{self.timestamp}",
            'timestamp': self.timestamp
        }
        
        print("=" * 50)
        print("PIPELINE SUBMITTED")
        print("=" * 50)
        for k, v in result.items():
            print(f"{k}: {v}")
        
        return result

def main():
    """Main function."""
    
    # Get parameters from environment
    bucket = os.environ.get("SAGEMAKER_BUCKET")
    role_arn = os.environ.get("SAGEMAKER_ROLE")
    test_data_s3 = os.environ.get("TEST_DATA_S3")
    trained_model_s3 = os.environ.get("TRAINED_MODEL_S3")
    wait = os.environ.get("WAIT", "false").lower() == "true"
    
    if not all([bucket, role_arn, test_data_s3, trained_model_s3]):
        print("Missing required environment variables:")
        print("  SAGEMAKER_BUCKET")
        print("  SAGEMAKER_ROLE") 
        print("  TEST_DATA_S3")
        print("  TRAINED_MODEL_S3")
        return
    
    # Create and run pipeline
    pipeline = MinimalYOLOEvaluationPipeline(
        bucket=bucket,
        role_arn=role_arn
    )
    
    result = pipeline.run_evaluation(
        test_data_s3=test_data_s3,
        trained_model_s3=trained_model_s3,
        wait=wait
    )
    
    return result

if __name__ == "__main__":
    main()