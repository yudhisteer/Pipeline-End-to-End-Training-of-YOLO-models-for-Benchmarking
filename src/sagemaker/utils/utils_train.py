import boto3
import tarfile
import os
import json
import math
import shutil
import tempfile


from utils.utils_config import load_registry_config
from utils.utils_exceptions import ModelLoadError
from utils.utils_metrics import list_all_training_jobs_with_metrics 
from utils.utils_s3 import parse_s3_uri, object_exists
from utils.utils_metrics import list_specific_job_with_metrics



def list_registry_models():
    # This func may not work.
    """
    List models in the SageMaker Model Registry with their metrics and details.
    """
    try:
        config = load_registry_config()
        MODEL_GROUP = config.get("model_package_group_name")
        region = config['aws_region']
        if region:
            sm_client = boto3.client("sagemaker", region_name=region)
            s3_client = boto3.client("s3", region_name=region)
        else:
            sm_client = boto3.client("sagemaker")
            s3_client = boto3.client("s3")
        
        print(f"Models in Registry: {MODEL_GROUP}")
        print("=" * 80)
        
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
    #TODO: this func may not work
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
        config = load_registry_config()
        MODEL_GROUP = config.get("model_package_group_name")
        METRIC_KEY = config.get("metric_key")
        
        # use region from config if available
        region = config.get("aws_region")
        if region:
            sm_client = boto3.client("sagemaker", region_name=region)
            s3_client = boto3.client("s3", region_name=region)
        else:
            sm_client = boto3.client("sagemaker")
            s3_client = boto3.client("s3")
        
        # search for model packages
        search_args = {
            "ModelPackageGroupName": MODEL_GROUP,
            "SortBy": "CreationTime", 
            "SortOrder": "Descending",
            "MaxResults": config.get("max_results")
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
        model_tar_path = config.get("temp_files").get("model_archive")
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


def extract_evaluation_from_training_job(training_job_name: str) -> dict:
    """
    Extract evaluation.json from a SageMaker training job's model.tar.gz output.
    
    Args:
        training_job_name: Name of the completed training job
        
    Returns:
        Dictionary containing evaluation data with 'metrics' and 'model_info' keys
        
    Raises:
        FileNotFoundError: If evaluation.json is not found in the model archive
        RuntimeError: If extraction fails
        Exception: For other S3 or SageMaker API errors
    """
    # Get training job details to find S3 output path
    sagemaker_client = boto3.client('sagemaker')
    job_details = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
    output_path = job_details['OutputDataConfig']['S3OutputPath']
    
    # Construct path to model.tar.gz
    s3_model_path = f"{output_path}/{training_job_name}/output/model.tar.gz"
    
    # Download and extract model.tar.gz to find evaluation.json
    s3_client = boto3.client('s3')
    # Parse S3 path
    bucket = output_path.split('/')[2]
    key = '/'.join(output_path.split('/')[3:]) + f"/{training_job_name}/output/model.tar.gz"
    
    # Download tar.gz file
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
            
            # Extract and read evaluation.json
            extracted_file = tar.extractfile(evaluation_member)
            if extracted_file is None:
                raise RuntimeError(f"Could not extract {evaluation_member.name}")
            
            evaluation_data = json.loads(extracted_file.read().decode('utf-8'))
        
        # Clean up temp file
        os.unlink(tar_tmp_file.name)
    
    return evaluation_data


def validate_model_quality(training_job_name: str, quality_gates: dict, require_all: bool = True) -> tuple[bool, dict]:
    """
    Validate model quality against quality gates after training is complete.
    
    Args:
        training_job_name: Name of the completed training job
        quality_gates: Dictionary of quality gate names and their threshold values
        require_all: If True, all gates must pass. If False, at least one gate must pass.
        
    Returns:
        Tuple of (passes_gates: bool, validation_report: dict)
        
    Raises:
        Exception: For errors in extracting evaluation data or processing quality gates
    """
    try:
        # Extract metrics directly from S3 (same method as the display table)
        # Get training job details to extract S3 URI
        sagemaker_client = boto3.client('sagemaker')
        job_details = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
        s3_uri = job_details["ModelArtifacts"]["S3ModelArtifacts"]
        
        # Import extract_metrics_from_s3 and Console
        from rich.console import Console
        from .utils_metrics import extract_metrics_from_s3
        
        # Use the same extraction method as the display table
        console = Console()
        s3_metrics = extract_metrics_from_s3(s3_uri, console)
        
        if s3_metrics:
            # Convert S3 metrics to the format expected by quality gates
            metrics = {
                "mAP_0_5": float(s3_metrics.get("mAP50", 0.0)),
                "mAP_0_5_0_95": float(s3_metrics.get("mAP50-95", 0.0)), 
                "precision": float(s3_metrics.get("precision", 0.0)),
                "recall": float(s3_metrics.get("recall", 0.0)),
            }
            model_info = {"model_size_mb": 0.0}  # Placeholder
        else:
            # Fallback to evaluation.json approach
            evaluation_data = extract_evaluation_from_training_job(training_job_name)
            
            # Handle nested structure from generate_model_metrics function
            nested_metrics = evaluation_data.get('metrics', {})
            metrics = nested_metrics if nested_metrics else evaluation_data
            model_info = evaluation_data.get('model_info', {})
        
        # Check each quality gate dynamically
        validation_results = {}
        passes_count = 0
        total_gates = len(quality_gates)
        
        # Mapping of gate names to metric extraction logic
        gate_mappings = {
            "min_mAP_0_5": lambda: (metrics.get('mAP_0_5', 0.0), ">="),
            "min_mAP_0_5_0_95": lambda: (metrics.get('mAP_0_5_0_95', 0.0), ">="), 
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
                passes = None  # do not record unknown gates
                total_gates -= 1  # do not count unknown gates
            
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
            'evaluation_source': 'extracted_from_model_archive'
        }
        
        return overall_pass, validation_report
        
    except Exception as e:
        print(f"Failed to validate model quality: {e}")
        return False, {
            'error': f"Could not validate model quality: {e}",
            'training_job_name': training_job_name
        }


def check_and_display_quality_gates(pipeline, execution, pipeline_config: dict) -> dict:
    """
    Check quality gates and display results in a formatted table.
    
    Args:
        pipeline: The YOLOSageMakerPipeline instance
        execution: The pipeline execution object
        pipeline_config: Configuration dictionary containing approval strategy
        
    Returns:
        dict: Quality gate results including overall_pass, detailed_results, etc.
    """
    try:
        execution_status = execution.describe().get('PipelineExecutionStatus', 'Unknown')
        if execution_status != 'Succeeded':
            print(f"\nPipeline status is {execution_status} - skipping quality gate checks")
            return {'overall_pass': False, 'reason': f'Pipeline status: {execution_status}'}
        
        print(f"\nChecking quality gates for approval...")
        approval_result = pipeline.update_model_approval_after_training(execution)
        
        # Display basic results
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
                
                status_str = "PASS" if passes else "FAIL"
                
                # Create proper display names for gates
                display_name_map = {
                    'min_mAP_0_5': 'mAP50',
                    'min_mAP_0_5_0_95': 'mAP50-95',
                    'min_precision': 'Precision',
                    'min_recall': 'Recall',
                    'min_f1_score': 'F1 Score',
                    'max_model_size_mb': 'Model Size (MB)',
                    'max_inference_time_ms': 'Inference Time (ms)',
                    'min_speed_fps': 'Speed (FPS)'
                }
                display_name = display_name_map.get(gate_name, gate_name.replace('min_', '').replace('max_', '').replace('_', ' ').title())
                
                print(f"{display_name:<20} {threshold_str:<12} {actual_str:<12} {status_str:<10} {gap_str:<10}")
            
            print("-" * 70)
        
        print(f"Action needed: {approval_result.get('action_needed', 'N/A')}")
        return approval_result
        
    except Exception as e:
        print(f"Quality gate checking failed: {e}")
        return {'overall_pass': False, 'error': str(e)}


def update_model_package_approval(
    training_job_name: str, 
    model_package_group_name: str, 
    approval_status: str, 
    approval_description: str = None
    ) -> dict:
    """
    Update the approval status of a model package in SageMaker Model Registry.
    
    Args:
        training_job_name: Name of the training job to find the associated model package
        model_package_group_name: Name of the model package group
        approval_status: New approval status ('Approved', 'Rejected', or 'PendingManualApproval')
        approval_description: Optional description for the approval decision
        
    Returns:
        dict: Result of the approval update operation
        
    Raises:
        Exception: For errors in finding or updating the model package
    """
    try:
        sagemaker_client = boto3.client('sagemaker')
        
        # Find the model package associated with this training job
        print(f"Searching for model package from training job: {training_job_name}")
        
        # List model packages in the group, sorted by creation time (newest first)
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=50  # Search recent packages
        )
        
        target_model_package_arn = None
        
        # Look for model package that matches this training job
        for package in response['ModelPackageSummaryList']:
            try:
                # Get detailed info about the model package
                package_details = sagemaker_client.describe_model_package(
                    ModelPackageName=package['ModelPackageArn']
                )
                
                # Check if this model package came from our training job
                model_data_url = package_details['InferenceSpecification']['Containers'][0]['ModelDataUrl']
                
                # Extract training job name from S3 path
                if training_job_name in model_data_url:
                    target_model_package_arn = package['ModelPackageArn']
                    print(f"Found matching model package: {target_model_package_arn}")
                    break
                    
            except Exception as e:
                print(f"Error checking model package {package['ModelPackageArn']}: {e}")
                continue
        
        if not target_model_package_arn:
            raise Exception(f"No model package found for training job: {training_job_name}")
        
        # Update the model package approval status
        update_args = {
            'ModelPackageName': target_model_package_arn,
            'ModelApprovalStatus': approval_status
        }
        
        if approval_description:
            update_args['ApprovalDescription'] = approval_description
        
        print(f"Updating model package approval status to: {approval_status}")
        sagemaker_client.update_model_package(**update_args)
        
        return {
            'success': True,
            'model_package_arn': target_model_package_arn,
            'training_job_name': training_job_name,
            'approval_status': approval_status,
            'approval_description': approval_description,
            'message': f'Model package approval status updated to {approval_status}'
        }
        
    except Exception as e:
        error_msg = f"Failed to update model package approval: {e}"
        print(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'training_job_name': training_job_name,
            'requested_status': approval_status
        }

