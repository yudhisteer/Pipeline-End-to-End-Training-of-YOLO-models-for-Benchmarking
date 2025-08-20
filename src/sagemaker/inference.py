"""
SageMaker inference handler for YOLO models.
This file should be placed in code/inference.py
"""

import torch
import json
import logging
import sys
import os
from PIL import Image
import io
import numpy as np
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

def model_fn(model_dir):
    """Load the YOLO model for inference."""
    global model
    
    if model is not None:
        logger.info("Model already loaded, returning existing model")
        return model
    
    try:
        logger.info(f"Loading model from directory: {model_dir}")
        
        # List all files in model directory for debugging
        try:
            available_files = os.listdir(model_dir)
            logger.info(f"Available files in model directory: {available_files}")
        except Exception as e:
            logger.error(f"Cannot list model directory: {e}")
            available_files = []
        
        # Import ultralytics
        try:
            from ultralytics import YOLO
            logger.info("Successfully imported ultralytics YOLO")
        except ImportError as e:
            logger.error(f"Failed to import ultralytics: {e}")
            raise RuntimeError("ultralytics package is required but not available")
        
        # Look for YOLO model files
        model_files = ['best.pt', 'last.pt', 'yolo.pt', 'model.pt']
        model_path = None
        
        for model_file in model_files:
            potential_path = os.path.join(model_dir, model_file)
            if os.path.exists(potential_path):
                model_path = potential_path
                logger.info(f"Found model file: {model_path}")
                break
        
        if not model_path:
            error_msg = f"No YOLO model found in {model_dir}. Available files: {available_files}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Loading YOLO model from: {model_path}")
        
        # Load model with specific settings
        try:
            model = YOLO(model_path)
            # Ensure model is in eval mode
            if hasattr(model, 'model'):
                model.model.eval()
            logger.info("YOLO model loaded and set to eval mode")
            return model
        except Exception as load_error:
            logger.error(f"Failed to load YOLO model: {load_error}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Model loading failed: {load_error}")
        
    except Exception as e:
        logger.error(f"Error in model_fn: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

def input_fn(request_body, request_content_type):
    """Parse input data for inference."""
    try:
        logger.info(f"Processing input with content type: {request_content_type}")
        
        if request_body is None:
            raise ValueError("Request body is None")
        
        logger.info(f"Request body type: {type(request_body)}, size: {len(request_body) if hasattr(request_body, '__len__') else 'unknown'}")
        
        if request_content_type in ['image/jpeg', 'image/png', 'image/jpg', 'application/x-image']:
            # Handle direct image input
            try:
                image = Image.open(io.BytesIO(request_body))
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                logger.info(f"Loaded image with size: {image.size}, mode: {image.mode}")
                return image
            except Exception as img_error:
                logger.error(f"Failed to process image: {img_error}")
                raise ValueError(f"Invalid image data: {img_error}")
                
        elif request_content_type == 'application/json':
            # Handle JSON input (base64 encoded image)
            try:
                if isinstance(request_body, bytes):
                    request_body = request_body.decode('utf-8')
                
                data = json.loads(request_body)
                logger.info(f"JSON data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
                
                if 'image' in data:
                    import base64
                    image_data = base64.b64decode(data['image'])
                    image = Image.open(io.BytesIO(image_data))
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    logger.info(f"Loaded base64 image with size: {image.size}, mode: {image.mode}")
                    return image
                else:
                    available_keys = list(data.keys()) if isinstance(data, dict) else str(data)
                    raise ValueError(f"JSON input must contain 'image' field. Available keys: {available_keys}")
                    
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON decode error: {json_error}")
                raise ValueError(f"Invalid JSON format: {json_error}")
            except Exception as json_proc_error:
                logger.error(f"JSON processing error: {json_proc_error}")
                raise ValueError(f"Failed to process JSON: {json_proc_error}")
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
            
    except Exception as e:
        logger.error(f"Error in input_fn: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

def predict_fn(input_data, model):
    """Run inference on the input data."""
    try:
        logger.info("Starting prediction")
        
        if model is None:
            raise ValueError("Model is None")
        
        if input_data is None:
            raise ValueError("Input data is None")
        
        logger.info(f"Input image size: {input_data.size if hasattr(input_data, 'size') else 'unknown'}")
        
        # Run YOLO inference with proper error handling
        try:
            # Run inference with appropriate settings
            results = model(input_data, verbose=False, conf=0.25, iou=0.45)
            logger.info(f"Inference completed, got {len(results)} results")
        except Exception as inference_error:
            logger.error(f"Model inference failed: {inference_error}")
            logger.error(f"Inference traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Inference failed: {inference_error}")
        
        # Extract predictions with robust error handling
        predictions = []
        
        try:
            # Handle Ultralytics YOLO format
            for i, result in enumerate(results):
                logger.info(f"Processing result {i}")
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    logger.info(f"Found {len(boxes)} boxes in result {i}")
                    
                    for j, box in enumerate(boxes):
                        try:
                            # Safely extract box data
                            if hasattr(box, 'xyxy') and hasattr(box, 'conf') and hasattr(box, 'cls'):
                                # Get tensors and convert to numpy
                                xyxy_tensor = box.xyxy[0]
                                conf_tensor = box.conf[0] 
                                cls_tensor = box.cls[0]
                                
                                # Convert to CPU and numpy
                                x1, y1, x2, y2 = xyxy_tensor.cpu().numpy()
                                conf = float(conf_tensor.cpu().numpy())
                                cls = int(cls_tensor.cpu().numpy())
                                
                                # Get image dimensions
                                img_width, img_height = input_data.size
                                
                                # Create prediction with normalized coordinates
                                prediction = {
                                    "class": cls,
                                    "confidence": conf,
                                    "bbox": [
                                        float(x1/img_width),   # x1 normalized
                                        float(y1/img_height),  # y1 normalized  
                                        float(x2/img_width),   # x2 normalized
                                        float(y2/img_height)   # y2 normalized
                                    ]
                                }
                                predictions.append(prediction)
                                
                        except Exception as box_error:
                            logger.warning(f"Error processing box {j}: {box_error}")
                            continue
                else:
                    logger.info(f"No boxes found in result {i}")
                    
        except Exception as extraction_error:
            logger.error(f"Error extracting predictions: {extraction_error}")
            logger.error(f"Extraction traceback: {traceback.format_exc()}")
            # Return empty predictions rather than failing
            predictions = []
        
        logger.info(f"Successfully extracted {len(predictions)} predictions")
        return {"predictions": predictions}
        
    except Exception as e:
        logger.error(f"Error in predict_fn: {e}")
        logger.error(f"Prediction traceback: {traceback.format_exc()}")
        raise

def output_fn(prediction, content_type):
    """Format the prediction output."""
    try:
        logger.info(f"Formatting output with content type: {content_type}")
        logger.info(f"Prediction type: {type(prediction)}")
        
        if content_type == 'application/json':
            result = json.dumps(prediction)
            logger.info(f"Formatted output successfully, length: {len(result)}")
            return result
        else:
            raise ValueError(f"Unsupported output content type: {content_type}")
            
    except Exception as e:
        logger.error(f"Error in output_fn: {e}")
        logger.error(f"Output traceback: {traceback.format_exc()}")
        raise