"""
SageMaker Inference Script for YOLO Object Detection Models.
"""

import os
import json
import sys
import argparse
from typing import Dict, Any, Union

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils_config import (
    load_inference_config
)
from utils.utils_pipelines import (
    get_model_from_registry,
)
from utils.utils_metrics import (
    list_all_training_jobs_with_metrics,
)
from utils.utils_exceptions import (
    ModelLoadError,
    InferenceError
)
from utils.utils_inference import (
    visualize_detections,
    list_local_models
)



# global configuration
INFERENCE_CONFIG = load_inference_config()
DEFAULT_CONFIDENCE = INFERENCE_CONFIG['confidence_threshold']
DEFAULT_IOU_THRESHOLD = INFERENCE_CONFIG['iou_threshold']


def model_fn(model_dir: str, training_job_name: str = None) -> YOLO:
    """
    Load the YOLO model for inference.
    
    This function is called by SageMaker to load the model. It first tries to load
    a model from the provided model_dir, and if that fails, it attempts to download
    a model from the SageMaker Model Registry.
    
    Args:
        model_dir: Directory containing the model artifacts
        training_job_name: Optional specific training job name to load.
                          If None, loads the best performing model by metric.
        
    Returns:
        Loaded YOLO model instance
        
    Raises:
        ModelLoadError: If model loading fails
    """
    print(f"Loading model from directory: {model_dir}")
    
    try:
        # First, try to find ONNX model in the provided model directory
        onnx_path = None
        if os.path.exists(model_dir):
            # If training_job_name is provided, check in job-specific subdirectory first
            if training_job_name:
                job_specific_path = os.path.join(model_dir, training_job_name)
                if os.path.exists(job_specific_path):
                    for root, dirs, files in os.walk(job_specific_path):
                        for file in files:
                            if file.endswith(".onnx"):
                                onnx_path = os.path.join(root, file)
                                break
                        if onnx_path:
                            break
        
        # If no model found in model_dir, try to get from registry
        if not onnx_path:
            if training_job_name:
                print(f"No ONNX model found in model_dir, searching for training job: {training_job_name}")
            else:
                print("No ONNX model found in model_dir, searching registry...")
            onnx_path = get_model_from_registry(training_job_name)
        
        # Verify the model file exists
        if not onnx_path or not os.path.exists(onnx_path):
            if training_job_name:
                raise ModelLoadError(f"Training job '{training_job_name}' model not found locally and could not be downloaded from registry")
            else:
                raise ModelLoadError(f"ONNX model file not found at: {onnx_path}")
        
        # Additional verification: if training_job_name is specified, ensure we're using the right model
        if training_job_name and onnx_path:
            # Check if the model path contains the requested training job name
            if training_job_name not in onnx_path:
                raise ModelLoadError(f"Model found at {onnx_path} does not match requested training job: {training_job_name}")
        
        # Load the YOLO model
        print(f"Loading YOLO model from: {onnx_path}")
        model = YOLO(onnx_path, task="detect")
        
        print("Model loaded successfully")
        return model
        
    except ModelLoadError:
        # Re-raise ModelLoadError as-is
        raise
    except Exception as e:
        print(f"Error loading model: {e}")
        raise ModelLoadError(f"Failed to load model: {e}")


def input_fn(request_body: Union[str, bytes], content_type: str) -> Dict[str, Any]:
    """
    Parse and preprocess the input data.
    
    Args:
        request_body: The request body (image data or JSON)
        content_type: The content type of the request
        
    Returns:
        Dictionary containing processed input data
        
    Raises:
        ValueError: If input format is not supported
    """
    print(f"Processing input with content type: {content_type}")
    
    try:
        if content_type.startswith('image/'):
            # Handle direct image upload
            image_array = np.frombuffer(request_body, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            return {
                "image": image,
                "confidence": DEFAULT_CONFIDENCE,
                "iou_threshold": DEFAULT_IOU_THRESHOLD,
                "return_format": "json"
            }
            
        elif content_type == 'application/json':
            # Handle JSON request with parameters
            if isinstance(request_body, bytes):
                request_body = request_body.decode('utf-8')
            
            data = json.loads(request_body)
            
            # Handle base64 encoded image
            if "image_base64" in data:
                import base64
                image_data = base64.b64decode(data["image_base64"])
                image_array = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if image is None:
                    raise ValueError("Failed to decode base64 image")
                
                return {
                    "image": image,
                    "confidence": data.get("confidence", DEFAULT_CONFIDENCE),
                    "iou_threshold": data.get("iou_threshold", DEFAULT_IOU_THRESHOLD),
                    "return_format": data.get("return_format", "json")
                }
            
            # Handle image URL
            elif "image_url" in data:
                # For production, you might want to download from URL
                raise ValueError("Image URL input not implemented")
            
            else:
                raise ValueError("No valid image data found in JSON request")
        
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
            
    except Exception as e:
        print(f"Error processing input: {e}")
        raise ValueError(f"Input processing failed: {e}")


def predict_fn(input_data: Dict[str, Any], model: YOLO) -> Dict[str, Any]:
    """
    Run inference using the loaded model.
    
    Args:
        input_data: Preprocessed input data from input_fn
        model: Loaded YOLO model
        
    Returns:
        Dictionary containing prediction results
        
    Raises:
        InferenceError: If inference fails
    """
    print("Running inference...")
    
    try:
        image = input_data["image"]
        confidence = input_data["confidence"]
        iou_threshold = input_data["iou_threshold"]
        
        # Run YOLO inference
        results = model(
            image,
            conf=confidence,
            iou=iou_threshold,
            verbose=False
        )
        
        # Extract results from the first (and only) result
        result = results[0]
        
        # Process detections
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
            
            # Get class names
            class_names = model.names
            
            for i in range(len(boxes)):
                detection = {
                    "bbox": {
                        "x1": float(boxes[i][0]),
                        "y1": float(boxes[i][1]), 
                        "x2": float(boxes[i][2]),
                        "y2": float(boxes[i][3])
                    },
                    "confidence": float(confidences[i]),
                    "class_id": int(class_ids[i]),
                    "class_name": class_names[int(class_ids[i])]
                }
                detections.append(detection)
        
        # Prepare response
        response = {
            "detections": detections,
            "num_detections": len(detections),
            "image_shape": {
                "height": image.shape[0],
                "width": image.shape[1],
                "channels": image.shape[2]
            },
            "inference_params": {
                "confidence_threshold": confidence,
                "iou_threshold": iou_threshold
            }
        }
        
        print(f"Inference completed. Found {len(detections)} detections")
        return response
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise InferenceError(f"Inference failed: {e}")


def output_fn(prediction: Dict[str, Any], accept: str) -> Union[str, bytes]:
    """
    Format the prediction output based on the requested format.
    
    Args:
        prediction: Prediction results from predict_fn
        accept: Requested output format
        
    Returns:
        Formatted prediction output
        
    Raises:
        ValueError: If output format is not supported
    """
    print(f"Formatting output for accept type: {accept}")
    
    try:
        if accept == 'application/json':
            return json.dumps(prediction, indent=2)
        
        elif accept == 'text/plain':
            # Simple text format
            output_lines = [
                f"Found {prediction['num_detections']} detections:",
                f"Image size: {prediction['image_shape']['width']}x{prediction['image_shape']['height']}"
            ]
            
            for i, detection in enumerate(prediction['detections']):
                bbox = detection['bbox']
                output_lines.append(
                    f"Detection {i+1}: {detection['class_name']} "
                    f"(confidence: {detection['confidence']:.3f}, "
                    f"bbox: [{bbox['x1']:.0f}, {bbox['y1']:.0f}, {bbox['x2']:.0f}, {bbox['y2']:.0f}])"
                )
            
            return '\n'.join(output_lines)
        
        else:
            # Default to JSON for unknown accept types
            print(f"Unknown accept type '{accept}', defaulting to JSON")
            return json.dumps(prediction, indent=2)
            
    except Exception as e:
        print(f"Error formatting output: {e}")
        raise ValueError(f"Output formatting failed: {e}")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SageMaker YOLO Inference Script')
    parser.add_argument('--list-jobs', action='store_true', 
                       help='List all recent training jobs with Rich table formatting')
    parser.add_argument('--list-local', action='store_true',
                       help='List all locally available models')
    parser.add_argument('training_job_name', nargs='?', 
                       help='Specific training job name to load (optional)')

    args = parser.parse_args()
    
    if args.list_jobs:
        list_all_training_jobs_with_metrics()
        print("\nUsage: python sagemaker_inference.py <training_job_name>")
        exit(0)
    
    if args.list_local:
        list_local_models()
        print("\nUsage: python sagemaker_inference.py <training_job_name>")
        exit(0)

    print("Starting SageMaker YOLO inference script...")
    training_job_name = args.training_job_name
    
    if not training_job_name:
        print("Error: Training job name is required!")
        print("Usage: python sagemaker_inference.py <training_job_name>")
        print("Use --list-jobs to see available training jobs")
        print("Use --list-local to see locally available models")
        exit(1)
    
    print(f"Loading model from specific training job: {training_job_name}")
    
    try:
        # Try to load model from registry
        model = model_fn("./models", training_job_name)
        print("Model loaded successfully from registry")
        
    except ModelLoadError as e:
        print(f"Model loading failed: {e}")
        print(f"The model for training job '{training_job_name}' could not be found or loaded.")
        exit(1)
        
    except Exception as e:
        print(f"Unexpected error during model loading: {e}")
        exit(1)
    
    # Test with a sample image from config
    test_image_path = INFERENCE_CONFIG.get('test_image_path', '')
    
    if not test_image_path:
        print("Warning: test_image_path not configured in inference config")
        print("Skipping inference test - model loaded successfully")
        print("You can use the loaded model for inference with your own images")
        exit(0)
    
    # Handle both S3 paths and local paths
    if test_image_path.startswith('s3://'):
        print("Warning: S3 test images not yet supported for local testing")
        print("Please configure a local test image path in config.yaml")
        print("You can use the loaded model for inference with other images")
        exit(0)
    
    
    if not os.path.exists(test_image_path):
        print(f"Warning: Test image not found at {test_image_path}")
        print("Please ensure the dataset directory is properly configured")
        print("You can still use the loaded model for inference with other images")
        print(f"Expected path: {os.path.abspath(test_image_path)}")
        exit(0)
    
    try:
        print(f"Running inference on test image: {test_image_path}")
        
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            raise ValueError("Failed to load test image")
        
        # Load image as bytes for input_fn testing
        with open(test_image_path, 'rb') as f:
            image_data = f.read()
        
        # Run inference pipeline
        input_data = input_fn(image_data, 'image/jpeg')
        prediction = predict_fn(input_data, model)
        output = output_fn(prediction, 'application/json')
        
        print("Inference completed successfully!")
        print(f"Found {prediction['num_detections']} objects")
        
        # Save prediction results
        output_dir = INFERENCE_CONFIG.get('output_dir', './outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract image name from path for unique file naming
        image_name = os.path.splitext(os.path.basename(test_image_path))[0]
        
        # Save JSON results with image name
        json_path = os.path.join(output_dir, f"inference_results_{training_job_name}_{image_name}.json")
        with open(json_path, 'w') as f:
            f.write(output)
        print(f"JSON results saved to: {json_path}")
        
        # Visualize and save predictions
        print("Generating visualization...")
        save_path = os.path.join(output_dir, f"inference_visualization_{training_job_name}_{image_name}.jpg")
        
        # Convert detection format for visualization function
        # visualize_detections expects: (class_id, score, x0, y0, x1, y1) normalized coordinates
        converted_detections = []
        img_height, img_width = test_image.shape[:2]
        
        for det in prediction['detections']:
            # Convert absolute coordinates to normalized coordinates
            x0 = det['bbox']['x1'] / img_width
            y0 = det['bbox']['y1'] / img_height 
            x1 = det['bbox']['x2'] / img_width
            y1 = det['bbox']['y2'] / img_height
            
            converted_detections.append((
                det['class_id'],
                det['confidence'], 
                x0, y0, x1, y1
            ))
        
        visualize_detections(
            test_image_path, 
            converted_detections, 
            threshold=INFERENCE_CONFIG['confidence_threshold'],  # Use threshold from config
            save_path=save_path,
        )
        print(f"Visualization saved to: {save_path}")
        
        # Show sample output
        print("\nSample detection results:")
        print(output[:500] + "..." if len(output) > 500 else output)
        
    except Exception as e:
        print(f"Inference test failed: {e}")
        print("The model loaded successfully, but inference failed.")
        print("This might be due to image processing issues or model compatibility.")
        exit(1)
    
    print("\nLocal testing completed successfully!")
    print(f"All outputs saved to: {output_dir}")


    """
    Example Command:

    Run inference on a specific training job:

    # List the training jobs with metrics
    python inference_local.py --list-jobs

    # List the locally available models
    python inference_local.py --list-local

    # Run inference on a specific training job (This will check if the model is available locally, if not, it will download from the registry)
    python inference_local.py <training_job_name>
    """
