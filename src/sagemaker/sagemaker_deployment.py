"""
This module handles model deployment and endpoint management.
"""

import sagemaker
from sagemaker import get_execution_role, image_uris
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
    
    def create_model(self) -> sagemaker.model.Model:
        """
        Create SageMaker model from trained artifacts.
        
        Returns:
            SageMaker model object
        """
        if not self.model_data:
            raise ValueError("model_data must be set before creating model. Call get_model_data_from_job() or provide model_data in config.")
        
        # Get deployment configuration for instance type
        deployment_config = self.config.get('deployment', {})
        instance_type = deployment_config.get('instance_type', 'ml.m5.large')
        
        # Get the PyTorch container image (same as training)
        training_image = image_uris.retrieve(
            region=self.region,
            framework="pytorch",
            version="2.0",
            py_version="py310",
            image_scope="inference",
            instance_type=instance_type
        )
        
        print(f"Using inference image: {training_image}")
        print(f"Using model data: {self.model_data}")
        
        # Create model
        self.model = sagemaker.model.Model(
            image_uri=training_image,
            model_data=self.model_data,
            role=self.role,
            sagemaker_session=self.sess
        )
        
        print("Created SageMaker model")
        return self.model
    
    def deploy_endpoint(self, 
        endpoint_name: str = None, 
        instance_type: str = None,
        instance_count: int = None
        ) -> str:
        """
        Deploy model to SageMaker endpoint using configuration.
        
        Args:
            endpoint_name: Name for the endpoint (uses config default if None)
            instance_type: EC2 instance type for hosting (uses config default if None)
            instance_count: Number of instances (uses config default if None)
            
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
            # Don't re-raise the exception to avoid traceback
            return None
    
    def delete_endpoint(self, endpoint_name: str = None):
        """
        Delete the deployed endpoint to save costs.
        
        Args:
            endpoint_name: Name of endpoint to delete (uses current endpoint if None)
        """
        endpoint_to_delete = endpoint_name or self.endpoint_name
        
        if not endpoint_to_delete:
            print("No endpoint specified to delete")
            return
        
        print(f"Deleting endpoint: {endpoint_to_delete}")
        
        try:
            # Try using the endpoint object first if available
            if self.endpoint and endpoint_to_delete == self.endpoint_name:
                self.endpoint.delete_endpoint()
                print("Endpoint deleted successfully")
            else:
                # Use SageMaker client directly
                client = boto3.client('sagemaker', region_name=self.region)
                client.delete_endpoint(EndpointName=endpoint_to_delete)
                print("Endpoint deleted successfully using SageMaker client")
                
        except Exception as e:
            print(f"Error deleting endpoint: {e}")
            # Try to delete using SageMaker client as fallback
            try:
                client = boto3.client('sagemaker', region_name=self.region)
                client.delete_endpoint(EndpointName=endpoint_to_delete)
                print("Endpoint deleted successfully using SageMaker client fallback")
            except Exception as e2:
                print(f"Failed to delete endpoint using client fallback: {e2}")
        
        # Clear local references if this was our endpoint
        if endpoint_to_delete == self.endpoint_name:
            self.endpoint = None
            self.endpoint_name = None
    
    def get_model_data_from_job(self, job_name: str) -> str:
        """
        Get model artifacts S3 URI from a specific training job.
        
        Args:
            job_name: Name of the training job
            
        Returns:
            S3 URI of model artifacts
        """
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
        """
        Deploy model from a specific training job.
        
        Args:
            job_name: Name of the training job to deploy from
            endpoint_name: Name for the endpoint (uses config default if None)
            instance_type: EC2 instance type for hosting (uses config default if None)
            instance_count: Number of instances (uses config default if None)
            
        Returns:
            Endpoint name
        """
        print(f"Deploying model from training job: {job_name}")
        
        # Get model data from the specific job
        self.model_data = self.get_model_data_from_job(job_name)
        
        # Create model and deploy
        self.create_model()
        return self.deploy_endpoint(endpoint_name, instance_type, instance_count)
    
    def list_endpoints(self) -> list:
        """
        List all available SageMaker endpoints.
        
        Returns:
            List of endpoint names
        """
        try:
            client = boto3.client('sagemaker', region_name=self.region)
            response = client.list_endpoints()
            endpoints = [ep['EndpointName'] for ep in response['Endpoints']]
            print(f"Available endpoints: {endpoints}")
            return endpoints
        except Exception as e:
            print(f"Error listing endpoints: {e}")
            return []
    
    def endpoint_exists(self, endpoint_name: str) -> bool:
        """
        Check if an endpoint exists and is in service.
        
        Args:
            endpoint_name: Name of endpoint to check
            
        Returns:
            True if endpoint exists and is InService
        """
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
        except Exception as e:
            print(f"Error checking endpoint {endpoint_name}: {e}")
            return False





def deploy_model(job_name: str = None, config: Dict[str, Any] = None) -> SageMakerDeployment:
    """
    Unified deployment function that can deploy from:
    1. Specific job name (provided as argument)
    2. Default job name from config
    3. Static model_data from config (fallback)
    
    Args:
        job_name: Training job name to deploy from (optional)
        config: Configuration dictionary
    """
    console = Console()
    
    # Get deployment configuration
    deployment_config = config.get('deployment', {})
    
    # Determine job name to use
    final_job_name = job_name or deployment_config.get('default_job_name')
    
    # Get deployment settings from config
    endpoint_name = deployment_config.get('endpoint_name', 'yolo-endpoint-object-detection_v0')
    instance_type = deployment_config.get('instance_type', 'ml.m5.large')
    instance_count = deployment_config.get('initial_instance_count', 1)
    auto_delete = deployment_config.get('auto_delete', False)
    
    if final_job_name:
        # Deploy from training job (specific or default)
        console.print(Panel(
            f"[bold blue]Deploying Model from Training Job[/bold blue]\n"
            f"[dim]Job: {final_job_name}[/dim]\n"
            f"[dim]Source: {'Command argument' if job_name else 'Config default'}[/dim]",
            title="[bold green]Model Deployment[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))
        
        # Create deployment manager
        deployment = SageMakerDeployment(config=config)
        
        console.print(f"[cyan]Deployment Configuration:[/cyan]")
        console.print(f"  • Training Job: {final_job_name}")
        console.print(f"  • Endpoint name: {endpoint_name}")
        console.print(f"  • Instance type: {instance_type}")
        console.print(f"  • Instance count: {instance_count}")
        console.print()
        
        # Check if endpoint already exists
        if deployment.endpoint_exists(endpoint_name):
            console.print(f"[yellow]Endpoint '{endpoint_name}' already exists and is in service.[/yellow]")
            console.print("[dim]Skipping deployment. Use inference client to make predictions.[/dim]")
            return deployment
        
        try:
            # Deploy from training job
            deployed_endpoint = deployment.deploy_from_job(
                job_name=final_job_name,
                endpoint_name=endpoint_name,
                instance_type=instance_type,
                instance_count=instance_count
            )
            
            if deployed_endpoint:
                console.print(f"[green]Endpoint '{deployed_endpoint}' deployed successfully![/green]")
                
                # Handle auto-delete
                if auto_delete:
                    console.print("\n[yellow]Auto-delete enabled - deleting endpoint...[/yellow]")
                    deployment.delete_endpoint()
                else:
                    console.print(f"[dim]Remember to delete the endpoint when done to avoid charges.[/dim]")
            else:
                console.print("[red]Deployment failed. Check error messages above.[/red]")
                
        except Exception as e:
            console.print(f"[red]Deployment failed: {e}[/red]")
        
        return deployment
    
    else:
        # Fallback: Deploy using static model_data from config
        console.print(Panel(
            "[bold blue]Configuration-driven Model Deployment[/bold blue]\n"
            "[dim]Using static model_data from config.yaml[/dim]\n"
            "[yellow]No job name provided and no default_job_name in config[/yellow]",
            title="[bold green]Model Deployment[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))
        
        # Create deployment manager
        deployment = SageMakerDeployment(config=config)
        
        console.print(f"[cyan]Deployment Configuration:[/cyan]") 
        console.print(f"  • Endpoint name: {endpoint_name}")
        console.print(f"  • Instance type: {instance_type}")
        console.print(f"  • Instance count: {instance_count}")
        console.print()
        
        # Check if endpoint already exists
        if deployment.endpoint_exists(endpoint_name):
            console.print(f"[yellow]Endpoint '{endpoint_name}' already exists and is in service.[/yellow]")
            console.print("[dim]Skipping deployment. Use inference client to make predictions.[/dim]")
            return deployment
        
        # Deploy endpoint using static config
        deployed_endpoint = deployment.deploy_endpoint(
            endpoint_name=endpoint_name,
            instance_type=instance_type,
            instance_count=instance_count
        )
        
        # Check if deployment was successful
        if deployed_endpoint is None:
            console.print("[red]Endpoint deployment failed. Check the error messages above.[/red]")
            return deployment
        
        console.print(f"[green]Endpoint '{deployed_endpoint}' deployed successfully![/green]")
        
        # Handle auto-delete
        if auto_delete:
            console.print("\n[yellow]Auto-delete enabled - deleting endpoint...[/yellow]")
            deployment.delete_endpoint()
        else:
            console.print(f"[dim]Endpoint is still running. Remember to delete when done to avoid charges.[/dim]")
        
        return deployment


def main():
    """Enhanced SageMaker model deployment with job-specific support."""
    parser = argparse.ArgumentParser(
        description="SageMaker Model Deployment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
        Examples:
        # Deploy model from specific training job
        python src/sagemaker/sagemaker_deployment.py pipelines-bpblsxjiotwz-YOLOTrainingStep-Obj-S6s6poOKsB
        
        # Deploy using default_job_name from config.yaml (if set)
        python src/sagemaker/sagemaker_deployment.py
        
        # List recent training jobs
        python src/sagemaker/sagemaker_deployment.py --list-jobs
        
        # Show metrics for all recent jobs
        python src/sagemaker/sagemaker_deployment.py --metrics
        
        # Show metrics for specific job
        python src/sagemaker/sagemaker_deployment.py --metrics pipelines-bpblsxjiotwz-YOLOTrainingStep-Obj-S6s6poOKsB
        
        # List top 5 jobs
        python src/sagemaker/sagemaker_deployment.py --list-jobs --max-results 5
        """
    )
    
    parser.add_argument(
        'job_name',
        nargs='?',
        help='Training job name to deploy from'
    )
    
    parser.add_argument(
        '--list-jobs',
        action='store_true',
        help='List recent training jobs'
    )
    
    parser.add_argument(
        '--metrics',
        nargs='?',
        const='',
        help='Show metrics for jobs (optionally specify job name)'
    )
    
    parser.add_argument(
        '--max-results',
        type=int,
        default=10,
        help='Maximum number of jobs to show (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    console = Console()
    try:
        config = load_config()
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)
    
    # Handle different command modes
    if args.list_jobs:
        # List jobs without metrics
        display_training_job_metrics(
            training_job_name=None, 
            show_metrics=False, 
            max_results=args.max_results
        )
        
    elif args.metrics is not None:
        # Show metrics for specific job or all jobs
        job_name = args.metrics if args.metrics else None
        display_training_job_metrics(
            training_job_name=job_name, 
            show_metrics=True, 
            max_results=args.max_results
        )
        
    else:
        # Deploy model (either from job_name argument or default config)
        deployment = deploy_model(job_name=args.job_name, config=config)
        return deployment


if __name__ == "__main__":
    main()
