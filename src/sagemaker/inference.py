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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load the YOLO model for inference."""
    try:
        logger.info(f"Loading model from directory: {model_dir}")
        
        # List all files in model directory for debugging
        available_files = os.listdir(model_dir)
        logger.info(f"Available files in model directory: {available_files}")
        
        # Try to import ultralytics first
        try:
            from ultralytics import YOLO
            logger.info("Successfully imported ultralytics YOLO")
            
            # Look for YOLO model files
            model_files = ['best.pt', 'last.pt', 'yolo.pt', 'model.pt']
            model_path = None
            
            for model_file in model_files:
                potential_path = os.path.join(model_dir, model_file)
                if os.path.exists(potential_path):
                    model_path = potential_path
                    break
            
            if not model_path:
                logger.error(f"No YOLO model found in {model_dir}. Available files: {available_files}")
                raise FileNotFoundError("No YOLO model file found")
            
            logger.info(f"Loading YOLO model from: {model_path}")
            model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")
            return model
            
        except ImportError:
            logger.warning("ultralytics not available, trying torch.hub")
            try:
                # Fallback to torch.hub yolov5
                import torch
                model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                     path=os.path.join(model_dir, 'best.pt'),
                                     force_reload=True)
                logger.info("Loaded YOLOv5 model via torch.hub")
                return model
            except Exception as hub_error:
                logger.error(f"torch.hub failed: {hub_error}")
                
                # Last resort: try loading as pure PyTorch model
                try:
                    model_path = os.path.join(model_dir, 'model.pth')
                    if not os.path.exists(model_path):
                        model_path = os.path.join(model_dir, 'best.pt')
                    
                    model = torch.load(model_path, map_location='cpu')
                    logger.info(f"Loaded PyTorch model from {model_path}")
                    return model
                except Exception as torch_error:
                    logger.error(f"PyTorch load failed: {torch_error}")
                    raise
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

def input_fn(request_body, request_content_type):
    """Parse input data for inference."""
    try:
        logger.info(f"Processing input with content type: {request_content_type}")
        logger.info(f"Request body type: {type(request_body)}, size: {len(request_body) if hasattr(request_body, '__len__') else 'unknown'}")
        
        if request_content_type in ['image/jpeg', 'image/png', 'image/jpg']:
            # Handle image input
            image = Image.open(io.BytesIO(request_body))
            logger.info(f"Loaded image with size: {image.size}, mode: {image.mode}")
            return image
        elif request_content_type == 'application/json':
            # Handle JSON input (base64 encoded image)
            data = json.loads(request_body)
            if 'image' in data:
                import base64
                image_data = base64.b64decode(data['image'])
                image = Image.open(io.BytesIO(image_data))
                logger.info(f"Loaded base64 image with size: {image.size}, mode: {image.mode}")
                return image
            else:
                raise ValueError("JSON input must contain 'image' field with base64 encoded image")
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

def predict_fn(input_data, model):
    """Run inference on the input data."""
    try:
        logger.info("Starting prediction")
        
        # Run YOLO inference
        results = model(input_data, verbose=False)
        logger.info(f"Got {len(results)} results from model")
        
        # Extract predictions
        predictions = []
        
        # Handle different YOLO result formats
        if hasattr(results, '__iter__'):
            # Ultralytics YOLO format
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for box in boxes:
                        # Extract box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Get image dimensions
                        img_width, img_height = input_data.size
                        
                        predictions.append([
                            cls,           # class
                            conf,          # confidence
                            x1/img_width,  # x1 normalized
                            y1/img_height, # y1 normalized  
                            x2/img_width,  # x2 normalized
                            y2/img_height  # y2 normalized
                        ])
        else:
            # YOLOv5 hub format or other formats
            if hasattr(results, 'pandas'):
                # YOLOv5 format
                df = results.pandas().xyxy[0]
                img_width, img_height = input_data.size
                
                for _, row in df.iterrows():
                    predictions.append([
                        int(row['class']),        # class
                        float(row['confidence']), # confidence
                        row['xmin']/img_width,    # x1 normalized
                        row['ymin']/img_height,   # y1 normalized
                        row['xmax']/img_width,    # x2 normalized
                        row['ymax']/img_height    # y2 normalized
                    ])
        
        logger.info(f"Extracted {len(predictions)} predictions")
        return {"prediction": predictions}
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

def output_fn(prediction, content_type):
    """Format the prediction output."""
    try:
        if content_type == 'application/json':
            return json.dumps(prediction)
        else:
            raise ValueError(f"Unsupported output content type: {content_type}")
    except Exception as e:
        logger.error(f"Error formatting output: {e}")
        raise