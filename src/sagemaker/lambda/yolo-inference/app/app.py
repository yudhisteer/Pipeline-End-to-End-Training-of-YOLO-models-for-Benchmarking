import json
import boto3
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import base64
import os
import tempfile

# Initialize S3 client
s3_client = boto3.client('s3')

# Global variables for model (loaded once per container)
model_session = None
MODEL_S3_BUCKET = "cyudhist-pipeline-yolo-503561429929"
MODEL_S3_KEY = "yolo-pipeline/onnx-models/best.onnx"

def load_model():
    """Load ONNX model from S3"""
    global model_session
    
    if model_session is None:
        print("Loading ONNX model from S3...")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
            # Download model from S3
            s3_client.download_fileobj(
                MODEL_S3_BUCKET,
                MODEL_S3_KEY, 
                tmp_file
            )
            tmp_file.flush()
            
            # Load ONNX model
            model_session = ort.InferenceSession(
                tmp_file.name,
                providers=['CPUExecutionProvider']  # Lambda only supports CPU
            )
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
        print("Model loaded successfully!")
        print(f"Model inputs: {[input.name for input in model_session.get_inputs()]}")
        print(f"Model outputs: {[output.name for output in model_session.get_outputs()]}")
    
    return model_session

def preprocess_image(image_data, target_size=(640, 640)):
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

def postprocess_predictions(predictions, confidence_threshold=0.5):
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

def lambda_handler(event, context):
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
        
        # Load model (happens once per container)
        model = load_model()
        
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
                        's3_bucket + s3_key': 'Reference image in S3'
                    }
                })
            }
        
        # Handle different input methods
        if 'image_base64' in body:
            # Base64 encoded image
            image_data = body['image_base64']
        elif 's3_bucket' in body and 's3_key' in body:
            # Image from S3
            response = s3_client.get_object(Bucket=body['s3_bucket'], Key=body['s3_key'])
            image_data = response['Body'].read()
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'No image provided. Send image_base64 or s3_bucket/s3_key'
                })
            }
        
        # Preprocess image
        input_array = preprocess_image(image_data)
        
        # Get input name from model
        input_name = model.get_inputs()[0].name
        
        # Run inference
        predictions = model.run(None, {input_name: input_array})
        
        # Debug: print prediction info
        print(f"Number of outputs: {len(predictions)}")
        for i, pred in enumerate(predictions):
            print(f"Output {i} shape: {pred.shape}")
        
        # Post-process predictions
        confidence_threshold = body.get('confidence_threshold', 0.5)
        detections = postprocess_predictions(predictions, confidence_threshold)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'detections': detections,
                'num_detections': len(detections),
                'model_info': {
                    'input_shape': [input.shape for input in model.get_inputs()],
                    'output_shape': [output.shape for output in model.get_outputs()]
                }
            })
        }
        
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