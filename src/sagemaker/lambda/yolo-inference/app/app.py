import json
import boto3
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import base64
import os
import tempfile
import tarfile
import yaml
from urllib.parse import urlparse



def get_latest_successful_training_job() -> str:
    """
    Get the latest successful training job from SageMaker.
    
    Returns:
        str: Latest successful training job name
    """
    try:
        sm = boto3.client("sagemaker")
        
        # Get recent training jobs, sorted by creation time (newest first)
        response = sm.list_training_jobs(
            SortBy="CreationTime", 
            SortOrder="Descending", 
            MaxResults=50  # Look at more jobs to find a successful one
        )
        
        # Find the first completed job
        for job in response["TrainingJobSummaries"]:
            if job["TrainingJobStatus"] == "Completed":
                job_name = job["TrainingJobName"]
                print(f"Found latest successful training job: {job_name}")
                return job_name
        
        raise ValueError("No successful training jobs found")
        
    except Exception as e:
        raise Exception(f"Error retrieving latest training job: {e}")


def load_deployment_config() -> str:
    """
    Load deployment configuration from config.yaml and return the training job name.
    If job_name is empty, gets the latest successful training job from SageMaker.
    
    Returns:
        str: Training job name for model loading
    """
    try:
        config_path = 'config.yaml'  # config.yaml is copied to container root by Dockerfile
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        # extract job name from deployment config
        deployment = config.get('deployment', {})
        job_name = deployment.get('job_name')
        
        if not job_name or job_name.strip() == '':
            job_name = get_latest_successful_training_job()
            print(f"No job name provided. Getting latest successful training job: {job_name}")
        else:
            print(f"Loaded job name from config: {job_name}")
            
        return job_name
        
    except FileNotFoundError:
        raise Exception("config.yaml not found!")
    except Exception as e:
        raise Exception(f"Error loading config.yaml: {e}")



def get_model_artifacts_path(job_name: str) -> str:
    """
    Get the S3 model artifacts path for a specific SageMaker training job.
    
    Args:
        job_name: Name of the SageMaker training job
        
    Returns:
        str: S3 path to model artifacts (model.tar.gz)
        
    Raises:
        ValueError: If job not found or missing model artifacts
    """
    try:
        # Get training job details from SageMaker
        sm = boto3.client("sagemaker")
        job_details = sm.describe_training_job(TrainingJobName=job_name)
        
        # Extract model artifacts path
        model_artifacts = job_details.get("ModelArtifacts", {})
        s3_model_path = model_artifacts.get("S3ModelArtifacts")
        
        if not s3_model_path:
            raise ValueError(f"Training job '{job_name}' does not have S3ModelArtifacts")
            
        print(f"Found model artifacts at: {s3_model_path}")
        return s3_model_path
        
    except Exception as e:
        error_msg = f"Failed to get model artifacts for training job '{job_name}': {e}"
        raise ValueError(error_msg)





# Load training job name from configuration
job_name = load_deployment_config()

# Initialize S3 client
s3_client = boto3.client('s3')

# Global variables for model (loaded once per container)
model_session = None


def load_model(job_name: str):
    """Load ONNX model from S3 tar.gz file"""
    global model_session
    
    if model_session is None:
        print("Loading ONNX model from S3 tar.gz...")
        model_path = get_model_artifacts_path(job_name)
        # Parse S3 path to get bucket and key
        parsed_url = urlparse(model_path)
        model_s3_bucket = parsed_url.netloc
        model_s3_key = parsed_url.path.lstrip('/')
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tar_tmp_file:
            # Download tar.gz file from S3
            print(f"Downloading {model_s3_bucket}/{model_s3_key}")
            s3_client.download_fileobj(
                model_s3_bucket,
                model_s3_key, 
                tar_tmp_file
            )
            tar_tmp_file.flush()
            
            # Extract best.onnx from tar.gz
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as onnx_tmp_file:
                try:
                    with tarfile.open(tar_tmp_file.name, 'r:gz') as tar:
                        # Look for best.onnx in the tar file
                        best_onnx_member = None
                        for member in tar.getmembers():
                            if member.name.endswith('best.onnx') or member.name == 'best.onnx':
                                best_onnx_member = member
                                break
                        
                        if best_onnx_member is None:
                            raise FileNotFoundError("best.onnx not found in the tar.gz file")
                        
                        print(f"Found {best_onnx_member.name} in tar.gz")
                        
                        # Extract best.onnx to temporary file
                        extracted_file = tar.extractfile(best_onnx_member)
                        if extracted_file is None:
                            raise RuntimeError(f"Could not extract {best_onnx_member.name}")
                        
                        onnx_tmp_file.write(extracted_file.read())
                        onnx_tmp_file.flush()
                        
                        # Load ONNX model
                        model_session = ort.InferenceSession(
                            onnx_tmp_file.name,
                            providers=['CPUExecutionProvider']  # Lambda only supports CPU
                        )
                        
                finally:
                    # Clean up temp files
                    os.unlink(tar_tmp_file.name)
                    os.unlink(onnx_tmp_file.name)
            
        print("Model loaded successfully!")
        print(f"Model inputs: {[input.name for input in model_session.get_inputs()]}")
        print(f"Model outputs: {[output.name for output in model_session.get_outputs()]}")
    
    return model_session

def preprocess_image(image_data: str, target_size: tuple[int, int] = (640, 640)) -> np.ndarray:
    """Preprocess image for YOLO inference"""
    # Convert base64 to PIL Image if needed
    if isinstance(image_data, str):
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
    else:
        image = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Change from HWC to CHW format (channels first)
    img_array = img_array.transpose(2, 0, 1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def postprocess_predictions(predictions: np.ndarray, confidence_threshold: float) -> list[dict]:
    """Post-processing of YOLO predictions"""
    detections = []
    
    if len(predictions) > 0:
        pred = predictions[0]  # Get first output
        
        # Print shape for debugging
        print(f"Prediction shape: {pred.shape}")
        
        # Handle different YOLO output formats
        if len(pred.shape) == 3:
            batch_size, features, num_anchors = pred.shape
            
            if features == 5:
                # Format: [batch, 5, 8400] - need to transpose to [batch, 8400, 5]
                print("Transposing prediction from (batch, 5, anchors) to (batch, anchors, 5)")
                pred = pred.transpose(0, 2, 1)  # [1, 8400, 5]
            
            batch_pred = pred[0]  # Get first batch: [8400, 5]
        else:
            batch_pred = pred
        
        print(f"Processing {batch_pred.shape[0]} anchor predictions")
        print(f"Each prediction has {batch_pred.shape[1]} features")
        
        # Debug: Show some sample values
        print(f"Sample prediction values:")
        print(f"  First prediction: {batch_pred[0]}")
        print(f"  Min values: {np.min(batch_pred, axis=0)}")
        print(f"  Max values: {np.max(batch_pred, axis=0)}")
        print(f"  Mean values: {np.mean(batch_pred, axis=0)}")
        
        # Now batch_pred should be [num_anchors, features]
        # Process only anchors with reasonable objectness scores for performance
        valid_detections = 0
        processed_count = 0
        
        for i, detection in enumerate(batch_pred):
            processed_count += 1
            if processed_count > 8400:  # Safety limit
                break
                
            if len(detection) >= 5:
                # Extract coordinates and objectness score
                x_center, y_center, width, height, objectness = detection[:5]
                
                # Convert to scalar values
                x_center = float(x_center)
                y_center = float(y_center) 
                width = float(width)
                height = float(height)
                objectness = float(objectness)
                
                # Skip very low objectness predictions early for performance
                if objectness < 0.01:  # Very low threshold for objectness
                    continue
                    
                # Get class probabilities (if available)
                class_probs = detection[5:] if len(detection) > 5 else np.array([1.0])
                
                # For YOLO, often the final confidence is just objectness
                # Let's try both approaches
                if len(class_probs) > 0 and len(class_probs) > 1:
                    # Multi-class case
                    max_class_prob = float(np.max(class_probs))
                    max_class_idx = int(np.argmax(class_probs))
                    # Method 1: objectness * max_class_prob
                    final_confidence_v1 = objectness * max_class_prob
                    # Method 2: just objectness (common in some YOLO versions)
                    final_confidence_v2 = objectness
                    
                    # Use the higher of the two for now
                    final_confidence = max(final_confidence_v1, final_confidence_v2)
                else:
                    # Single class or no class probs
                    max_class_prob = 1.0
                    max_class_idx = 0
                    final_confidence = objectness
                
                # Apply confidence threshold
                if final_confidence > confidence_threshold:
                    valid_detections += 1
                    
                    # Convert center format to corner format
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    detections.append({
                        'bbox_center': [x_center, y_center, width, height],
                        'bbox_corners': [x1, y1, x2, y2],
                        'objectness': objectness,
                        'class_confidence': max_class_prob,
                        'final_confidence': final_confidence,
                        'class_id': max_class_idx,
                        'class_scores': class_probs.tolist() if len(class_probs) > 0 else [],
                        'detection_index': i  # For debugging
                    })
        
        print(f"Processed {processed_count} predictions")
        print(f"Found {len(detections)} detections above threshold {confidence_threshold}")
    
    return detections

def process_batch_images(body: dict, model: ort.InferenceSession) -> dict:
    """Process multiple images in batch"""

    # Get parameters from body
    confidence_threshold = body.get('confidence_threshold')
    chunk_size = body.get('chunk_size')
    images = body.get('images', [])
    
    if not images:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'No images provided in batch'})
        }
    
    print(f"Processing batch of {len(images)} images in chunks of {chunk_size}")
    
    results = []
    total_processing_time = 0
    
    # Process images in chunks
    for chunk_idx in range(0, len(images), chunk_size):
        chunk = images[chunk_idx:chunk_idx + chunk_size]
        chunk_results = process_image_chunk(chunk, model, confidence_threshold)
        results.extend(chunk_results)
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'results': results,
            'batch_size': len(images),
            'chunks_processed': (len(images) + chunk_size - 1) // chunk_size,
            'total_detections': sum(len(r.get('detections', [])) for r in results)
        })
    }

def process_image_chunk(image_chunk: list[dict], model: ort.InferenceSession, confidence_threshold: float) -> list[dict]:
    """Process a chunk of images"""
    chunk_results = []
    print(f"DEBUG: Starting batch processing")
    
    for img_info in image_chunk:
        try:
            # Get image data
            if 'image_base64' in img_info:
                image_data = img_info['image_base64']
            elif 's3_bucket' in img_info and 's3_key' in img_info:
                response = s3_client.get_object(Bucket=img_info['s3_bucket'], Key=img_info['s3_key'])
                image_data = response['Body'].read()
            else:
                chunk_results.append({
                    'image_id': img_info.get('image_id', 'unknown'),
                    'status': 'error',
                    'error': 'No valid image data provided'
                })
                continue
            
            # Process single image
            input_array = preprocess_image(image_data)
            input_name = model.get_inputs()[0].name
            predictions = model.run(None, {input_name: input_array})
            detections = postprocess_predictions(predictions, confidence_threshold)
            
            chunk_results.append({
                'image_id': img_info.get('image_id', f'image_{len(chunk_results)}'),
                'status': 'success',
                'detections': detections,
                'num_detections': len(detections)
            })
            
        except Exception as e:
            print(f"DEBUG: Error in process_batch_images: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    return chunk_results



def lambda_handler(event: dict, context: dict) -> dict:
    """Main Lambda handler"""
    try:
        # Handle health check
        if event.get('httpMethod') == 'GET' and event.get('path') == '/health':
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'status': 'healthy',
                    'message': 'YOLO inference service is running'
                })
            }
        
        # 1. Load model (happens once per container)
        model = load_model(job_name)
        
        # Parse request body for POST requests
        body = None
        if event.get('body'):
            try:
                body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
            except json.JSONDecodeError:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'Invalid JSON in request body'})
                }
        
        if not body:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Request body is required for inference',
                    'usage': {
                        'image_base64': 'Send base64 encoded image',
                        's3_bucket + s3_key': 'Reference image in S3',
                        'images': 'Array of images for batch processing'
                    }
                })
            }
        
        # 2. Batch processing
        results = process_batch_images(body, model)

        return results
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'type': type(e).__name__
            })
        }