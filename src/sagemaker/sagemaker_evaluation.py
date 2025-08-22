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
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load config
        import yaml
        from dotenv import load_dotenv
        load_dotenv()
        
        # Load and expand config
        with open(config_path, 'r') as f:
            config_content = f.read()

        config_content = os.path.expandvars(config_content)
        self.config = yaml.safe_load(config_content)
        
        # Extract values from config
        aws_config = self.config.get('aws', {})
        eval_config = self.config.get('evaluation', {})
        
        self.bucket = aws_config.get('bucket')
        if not self.bucket:
            # Try to get default bucket from SageMaker
            try:
                import sagemaker
                self.bucket = sagemaker.Session().default_bucket()
                print(f"Using SageMaker default bucket: {self.bucket}")
            except Exception as e:
                print(f"Warning: Could not get SageMaker default bucket: {e}")
                self.bucket = "yolo-evaluation-bucket"  # fallback
                print(f"Using fallback bucket: {self.bucket}")
        
        self.role_arn = aws_config.get('role_arn')
        if not self.role_arn:
            # Try to get execution role from SageMaker
            try:
                import sagemaker
                self.role_arn = sagemaker.get_execution_role()
                print(f"Using SageMaker execution role: {self.role_arn}")
            except Exception as e:
                print(f"Warning: Could not get SageMaker execution role: {e}")
                print("Please set ROLE_ARN environment variable or specify in config.yaml")
        
        self.region = aws_config.get('region', 'us-east-1')
        self.prefix = aws_config.get('prefix', 'yolo-eval')
        
        # Pipeline settings
        pipeline_config = self.config.get('evaluation_pipeline', {})
        self.pipeline_name = pipeline_config.get('name', 'yolo-evaluation-pipeline')
        
        # Evaluation parameters
        self.test_data_s3 = eval_config.get('s3_test_dataset')
        self.trained_model = eval_config.get('trained_model')
        self.instance_type = eval_config.get('instance_type', 'ml.m5.large')
        
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.pipeline_session = PipelineSession()
        
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
        
        # Add config file if it exists locally
        if os.path.exists("config.yaml"):
            inputs.append(
                ProcessingInput(
                    source="config.yaml",  # local file
                    destination="/opt/ml/processing/input_config.yaml",
                    input_name="config"
                )
            )
            print("✓ Config file will be uploaded")
        else:
            print("⚠️ Local config.yaml not found - will use defaults in processing job")
        
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
            code="src/sagemaker/entrypoint_evaluation.py"
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
    config_path = "config.yaml"  # or from command line arg
    
    try:
        pipeline = MinimalYOLOEvaluationPipeline(config_path)
        
        # Validate required config values
        if not pipeline.test_data_s3:
            raise ValueError("test_data_s3 not found in config. Please check the 'evaluation.s3_test_dataset' field.")
        if not pipeline.trained_model:
            raise ValueError("trained_model not found in config. Please check the 'evaluation.trained_model' field.")
        
        print(f"Using test data: {pipeline.test_data_s3}")
        print(f"Using trained model: {pipeline.trained_model}")
        
        result = pipeline.run_evaluation(
            test_data_s3=pipeline.test_data_s3,
            trained_model_s3=pipeline.trained_model,
            wait=False
        )
        
        return result
        
    except FileNotFoundError:
        print("config.yaml not found. Please create configuration file.")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()