"""
This module handles model deployment and endpoint management.
"""

import sagemaker
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorchModel
import boto3
import os
import sys
import argparse
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel

from utils.utils_config import (
    load_config,
    get_aws_config,
    get_inference_config
)
from sagemaker_metrics import display_training_job_metrics


class SageMakerDeployment:
    """Handles SageMaker model deployment and endpoint management."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize SageMaker deployment with configuration.
        
        Args:
            config: Configuration dictionary from YAML file
        """
        self.config = config or load_config()
        
        # Get AWS and inference configuration
        aws_config = get_aws_config(self.config)
        inference_config = get_inference_config(self.config)
        
        # Model data will be set either from training job or config fallback
        self.model_data = inference_config.get('model_data')
        # Note: model_data can be None - will be set later from training job
        
        self.region = aws_config.get('region')
        self.role_arn = aws_config.get('role_arn') or os.getenv('ROLE_ARN')
        
        self.sess = sagemaker.Session()
        self.region = self.region or self.sess.boto_region_name
        
        # Handle role for local development
        if self.role_arn:
            self.role = self.role_arn
        else:
            try:
                self.role = get_execution_role()
            except:
                raise ValueError(
                    "Must provide role_arn in config.yaml or environment variable. "
                    f"Use: --role-arn {self.role_arn}"
                )
        
        # Store inference configuration
        self.inference_config = inference_config
        
        # Model and endpoint
        self.model = None
        self.endpoint = None
        self.endpoint_name = None
        
        print(f"Initialized deployment manager with model: {self.model_data or 'TBD from training job'}")
    
    def create_model(self) -> PyTorchModel:
        """
        Create SageMaker PyTorch model from trained artifacts.
        
        Returns:
            SageMaker PyTorch model object
        """
        if not self.model_data:
            raise ValueError("model_data must be set before creating model. Call get_model_data_from_job() or provide model_data in config.")

        # Create PyTorch model with custom inference code
        self.model = PyTorchModel(
            model_data=self.model_data,
            role=self.role,
            framework_version="2.0",
            py_version="py310",
            entry_point="inference.py",
            source_dir="src/sagemaker",
            dependencies=["src/sagemaker/dependencies/requirements.txt"],
            sagemaker_session=self.sess,
            env={
                'SAGEMAKER_MODEL_SERVER_TIMEOUT': '3600',
                'SAGEMAKER_MODEL_SERVER_WORKERS': '1',
                'MMS_DEFAULT_RESPONSE_TIMEOUT': '900',
                'SAGEMAKER_ENABLE_CLOUDWATCH_METRICS': 'true'
            }
        )
        
        print("Created PyTorch model with custom YOLO inference code")
        return self.model
    
    def deploy_endpoint(self, 
        endpoint_name: str = None, 
        instance_type: str = None,
        instance_count: int = None
        ) -> str:
        """
        Deploy model to SageMaker endpoint using configuration.
        
        Args:
            endpoint_name: Name for the endpoint
            instance_type: EC2 instance type for hosting
            instance_count: Number of instances
            
        Returns:
            Endpoint name
        """
        if self.model is None:
            self.create_model()
        
        # Get endpoint configuration
        endpoint_config = self.inference_config.get('endpoint', {})
        endpoint_name = endpoint_name or endpoint_config.get('default_endpoint_name', 'plastic-bag-detection-endpoint')
        instance_type = instance_type or endpoint_config.get('instance_type', 'ml.m4.xlarge')
        instance_count = instance_count or endpoint_config.get('instance_count', 1)
        
        print(f"Deploying endpoint: {endpoint_name}")
        print(f"Instance type: {instance_type}")
        print(f"Instance count: {instance_count}")
        
        self.endpoint_name = endpoint_name
        try:
            self.endpoint = self.model.deploy(
                initial_instance_count=instance_count,
                instance_type=instance_type,
                endpoint_name=endpoint_name
            )
            print(f"Endpoint deployed successfully: {endpoint_name}")
            return endpoint_name
        except Exception as e:
            # Check for specific duplicate endpoint config error
            if "already existing endpoint configuration" in str(e):
                print("Error: Endpoint configuration already exists")
                print("Solution: Use a different endpoint name or delete the existing configuration")
            else:
                print(f"Failed to deploy endpoint: {e}")
            # Reset endpoint name since deployment failed
            self.endpoint_name = None
            self.endpoint = None
            return None
    
    def delete_endpoint(self, endpoint_name: str = None):
        """Delete the deployed endpoint to save costs."""
        endpoint_to_delete = endpoint_name or self.endpoint_name
        
        if not endpoint_to_delete:
            print("No endpoint specified to delete")
            return
        
        print(f"Deleting endpoint: {endpoint_to_delete}")
        
        try:
            if self.endpoint and endpoint_to_delete == self.endpoint_name:
                self.endpoint.delete_endpoint()
                print("Endpoint deleted successfully")
            else:
                client = boto3.client('sagemaker', region_name=self.region)
                client.delete_endpoint(EndpointName=endpoint_to_delete)
                print("Endpoint deleted successfully using SageMaker client")
                
        except Exception as e:
            print(f"Error deleting endpoint: {e}")
        
        if endpoint_to_delete == self.endpoint_name:
            self.endpoint = None
            self.endpoint_name = None
    
    def get_model_data_from_job(self, job_name: str) -> str:
        """Get model artifacts S3 URI from a specific training job."""
        try:
            client = boto3.client('sagemaker', region_name=self.region)
            response = client.describe_training_job(TrainingJobName=job_name)
            model_artifacts = response['ModelArtifacts']['S3ModelArtifacts']
            print(f"Found model artifacts for job '{job_name}': {model_artifacts}")
            return model_artifacts
        except Exception as e:
            print(f"Error getting model data from job '{job_name}': {e}")
            raise ValueError(f"Could not find model artifacts for training job: {job_name}")
    
    def deploy_from_job(self, 
        job_name: str,
        endpoint_name: str = None, 
        instance_type: str = None,
        instance_count: int = None
        ) -> str:
        """Deploy model from a specific training job."""
        print(f"Deploying model from training job: {job_name}")
        
        # Get model data from the specific job
        self.model_data = self.get_model_data_from_job(job_name)
        
        # Create model and deploy
        self.create_model()
        return self.deploy_endpoint(endpoint_name, instance_type, instance_count)
    
    def endpoint_exists(self, endpoint_name: str) -> bool:
        """Check if an endpoint exists and is in service."""
        try:
            client = boto3.client('sagemaker', region_name=self.region)
            response = client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            print(f"Endpoint {endpoint_name} status: {status}")
            return status == 'InService'
        except client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                print(f"Endpoint {endpoint_name} does not exist")
                return False
            else:
                print(f"Error checking endpoint {endpoint_name}: {e}")
                return False


# Rest of the deployment functions remain the same...
def deploy_model(job_name: str = None, config: Dict[str, Any] = None) -> SageMakerDeployment:
    """Deploy model from training job or config."""
    console = Console()
    
    deployment_config = config.get('deployment', {})
    final_job_name = job_name or deployment_config.get('default_job_name')
    
    endpoint_name = deployment_config.get('endpoint_name', 'yolo-endpoint-object-detection_v0')
    instance_type = deployment_config.get('instance_type', 'ml.m5.large')
    instance_count = deployment_config.get('initial_instance_count', 1)
    auto_delete = deployment_config.get('auto_delete', False)
    
    deployment = SageMakerDeployment(config=config)
    
    if deployment.endpoint_exists(endpoint_name):
        console.print(f"[yellow]Endpoint '{endpoint_name}' already exists and is in service.[/yellow]")
        return deployment
    
    if final_job_name:
        try:
            deployed_endpoint = deployment.deploy_from_job(
                job_name=final_job_name,
                endpoint_name=endpoint_name,
                instance_type=instance_type,
                instance_count=instance_count
            )
            
            if deployed_endpoint:
                console.print(f"[green]Endpoint '{deployed_endpoint}' deployed successfully![/green]")
                if auto_delete:
                    console.print("\n[yellow]Auto-delete enabled - deleting endpoint...[/yellow]")
                    deployment.delete_endpoint()
            else:
                console.print("[red]Deployment failed. Check error messages above.[/red]")
                
        except Exception as e:
            console.print(f"[red]Deployment failed: {e}[/red]")
    
    return deployment


def main():
    """SageMaker model deployment tool."""
    parser = argparse.ArgumentParser(description="SageMaker Model Deployment Tool")
    parser.add_argument('job_name', nargs='?', help='Training job name to deploy from')
    args = parser.parse_args()
    
    console = Console()
    try:
        config = load_config()
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)
    
    deployment = deploy_model(job_name=args.job_name, config=config)
    return deployment


if __name__ == "__main__":
    main()