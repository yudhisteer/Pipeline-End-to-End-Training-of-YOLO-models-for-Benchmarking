import boto3
import tarfile
import os
import json
import math
import shutil


from utils.utils_config import load_inference_config
from utils.utils_exceptions import ModelLoadError
from utils.utils_metrics import list_all_training_jobs_with_metrics 
from utils.utils_s3 import parse_s3_uri, object_exists
from utils.utils_metrics import list_specific_job_with_metrics



def list_registry_models():
    """
    List models in the SageMaker Model Registry with their metrics and details.
    """
    try:
        config = load_inference_config()
        MODEL_GROUP = config['model_package_group']
        region = config['aws_region']
        if region:
            sm_client = boto3.client("sagemaker", region_name=region)
            s3_client = boto3.client("s3", region_name=region)
        else:
            sm_client = boto3.client("sagemaker")
            s3_client = boto3.client("s3")
        
        print(f"Models in Registry: {MODEL_GROUP}")
        print("=" * 80)
        
        config = load_inference_config()
        search_args = {
            "ModelPackageGroupName": MODEL_GROUP,
            "SortBy": "CreationTime",
            "SortOrder": "Descending",
            "MaxResults": config['registry']['max_results_display']
        }
        
        try:
            response = sm_client.list_model_packages(**search_args)
            
            if not response["ModelPackageSummaryList"]:
                print("No models found in the registry.")
                return
            
            for i, package in enumerate(response["ModelPackageSummaryList"], 1):
                try:
                    description = sm_client.describe_model_package(
                        ModelPackageName=package["ModelPackageArn"]
                    )
                    
                    model_s3_uri = description["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
                    bucket, key = parse_s3_uri(model_s3_uri)
                    path_parts = key.split("/")
                    
                    # extract training job name from path
                    training_job = "unknown"
                    training_patterns = config['registry']['training_job_patterns']
                    for part in path_parts:
                        if any(pattern in part for pattern in training_patterns):
                            training_job = part
                            break
                    if training_job == "unknown" and len(path_parts) >= 3:
                        training_job = path_parts[2] if len(path_parts) > 2 else path_parts[0]
                    
                    creation_time = package["CreationTime"].strftime('%Y-%m-%d %H:%M:%S')
                    status = package["ModelPackageStatus"]
                    approval_status = package.get("ModelApprovalStatus", "PendingManualApproval")
                    
                    print(f"\n{i}. Model Package")
                    print(f"   ARN: {package['ModelPackageArn'].split('/')[-1]}")
                    print(f"   Training Job: {training_job}")
                    print(f"   Status: {status}")
                    print(f"   Approval: {approval_status}")
                    print(f"   Created: {creation_time}")
                    
                    # get metrics using the existing function
                    print(f"Metrics for this model:")
                    try:
                        list_specific_job_with_metrics(training_job)
                    except Exception as metrics_error:
                        print(f"Could not fetch metrics: {str(metrics_error)[:100]}...")
                    
                    print(f"S3 Location: s3://{bucket}/{key}")
                    print("-" * 80)
                    
                except Exception as pkg_error:
                    print(f"Error processing package: {pkg_error}")
                    
        except sm_client.exceptions.ResourceNotFound:
            print(f"Model package group '{MODEL_GROUP}' not found.")
            print("Check the model package group name in your config.yaml")
            
    except Exception as e:
        print(f"Error listing models: {e}")
        print("Check your AWS credentials and permissions")



def get_model_from_registry(training_job_name: str = None) -> str:
    """
    Find and download a model from SageMaker Model Registry.
    
    Args:
        training_job_name: Optional specific training job name to load.
                          If None, finds the best performing model by metric.
    
    Returns:
        Path to the downloaded ONNX model file
        
    Raises:
        ModelLoadError: If no suitable model is found or download fails
    """
    if training_job_name:
        print(f"Searching for model from training job: {training_job_name}")
    else:
        print("Searching for best model in SageMaker Model Registry...")
    
    try:
        config = load_inference_config()
        MODEL_GROUP = config['model_package_group']
        METRIC_KEY = config['metric_key']
        
        # use region from config if available
        region = config['aws_region']
        if region:
            sm_client = boto3.client("sagemaker", region_name=region)
            s3_client = boto3.client("s3", region_name=region)
        else:
            sm_client = boto3.client("sagemaker")
            s3_client = boto3.client("s3")
        
        config = load_inference_config()
        
        # search for model packages
        search_args = {
            "ModelPackageGroupName": MODEL_GROUP,
            "SortBy": "CreationTime", 
            "SortOrder": "Descending",
            "MaxResults": config['registry']['max_results']
        }
        
        best_model = {"metric": -math.inf, "model_s3": None, "arn": None}
        
        # iterate through model packages to find the specific training job
        for package in sm_client.list_model_packages(**search_args)["ModelPackageSummaryList"]:
            try:
                # get model package details
                description = sm_client.describe_model_package(
                    ModelPackageName=package["ModelPackageArn"]
                )
                model_s3_uri = description["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
                
                # extract training job name from S3 path
                bucket, key = parse_s3_uri(model_s3_uri)
                path_parts = key.split("/")
                if len(path_parts) < 3:
                    print(f"Unexpected path format: {model_s3_uri}")
                    continue
                    
                current_training_job = path_parts[0]
                
                # Check if this is the requested training job
                if training_job_name:
                    matches = (current_training_job == training_job_name or 
                             any(training_job_name in part for part in path_parts))
                    if not matches:
                        continue
                    
                    # Found the requested training job! Use it immediately
                    print(f"Found requested training job model: {training_job_name}")
                    best_model = {
                        "metric": 0,  # Not needed for specific job
                        "model_s3": model_s3_uri,
                        "arn": package["ModelPackageArn"]
                    }
                    break  # Exit loop immediately
                
                else:
                    # For best model search (when no specific job requested)
                    # This code path is not used when training_job_name is provided
                    pass
                    
            except Exception as e:
                print(f"Error processing model package {package.get('ModelPackageArn', 'unknown')}: {e}")
                continue
        
        if not best_model["model_s3"]:
            if training_job_name:
                error_msg = f"Training job '{training_job_name}' not found in the Model Registry."
                error_msg += "\n\nPossible reasons:"
                error_msg += "\n- Training job name is incorrect or misspelled"
                error_msg += "\n- Model hasn't been registered in the Model Registry yet"
                error_msg += "\n- Model was registered with a different name"
                error_msg += "\n- Insufficient permissions to access the model"
                error_msg += "\n\nUse 'python sagemaker_inference.py --list-jobs' to see available training jobs"
            else:
                error_msg = "No suitable model found in registry"
                # Only list all jobs when searching for best model, not specific job
                available_jobs = list_all_training_jobs_with_metrics()
                if available_jobs:
                    error_msg += f"\nAvailable training jobs: {', '.join(available_jobs[:5])}"
                    if len(available_jobs) > 5:
                        error_msg += f" (and {len(available_jobs) - 5} more)"
            
            raise ModelLoadError(error_msg)
        
        print(f"Selected best model: {best_model['arn']} with {METRIC_KEY} = {best_model['metric']}")
        
        # download the best model
        bucket, key = parse_s3_uri(best_model["model_s3"])
        model_tar_path = config['registry']['temp_files']['model_archive']
        s3_client.download_file(bucket, key, model_tar_path)
        
        # extract the model
        model_extract_dir = config['registry']['temp_files']['model_extract_dir']
        
        # Create training job-specific directory
        if training_job_name:
            job_specific_dir = os.path.join(model_extract_dir, training_job_name)
        else:
            job_specific_dir = os.path.join(model_extract_dir, "best_model")
        
        os.makedirs(job_specific_dir, exist_ok=True)
        
        with tarfile.open(model_tar_path, "r:gz") as tar:
            tar.extractall(job_specific_dir)
        
        # find the ONNX model file
        onnx_path = None
        for root, dirs, files in os.walk(job_specific_dir):
            for file in files:
                if file.endswith(".onnx"):
                    onnx_path = os.path.join(root, file)
                    break
            if onnx_path:
                break
        
        if not onnx_path:
            raise ModelLoadError("No ONNX model file found in downloaded model")
        
        # Clean up only the temporary zip file
        print("Cleaning up temporary files...")
        try:
            # Remove only the tar.gz file
            if os.path.exists(model_tar_path):
                os.remove(model_tar_path)
                print(f"Removed temporary file: {model_tar_path}")
            
            # Keep the extraction directory - it contains the model we need
            print(f"Keeping extracted model in: {job_specific_dir}")
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up temporary files: {cleanup_error}")
        
        print(f"Model loaded successfully from: {onnx_path}")
        return onnx_path
        
    except Exception as e:
        print(f"Failed to load model from registry: {e}")
        raise ModelLoadError(f"Model loading failed: {e}")

