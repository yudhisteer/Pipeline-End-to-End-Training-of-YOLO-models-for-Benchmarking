#!/usr/bin/env python3
"""
Script to test YOLO Lambda inference locally
"""
import requests
import base64
import json
import os
from pathlib import Path
import sys
import os
    
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from src.sagemaker.utils.utils_config import load_config, get_lambda_config


def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def test_health_check(base_url="http://127.0.0.3000", health_endpoint="/health"):
    """Test the health check endpoint"""
    print(f"Testing health check at {base_url}{health_endpoint}...")
    try:
        response = requests.get(f"{base_url}{health_endpoint}", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False

def test_yolo_inference(image_path, base_url="http://127.0.0.1:3000", confidence_threshold=0.3, predict_endpoint="/predict") -> bool:
    """Test YOLO inference with a local image"""
    print(f"\nTesting YOLO inference with image: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return False
    
    # Get image info
    file_size = os.path.getsize(image_path) / 1024  # KB
    print(f"Image size: {file_size:.1f} KB")
    
    # Encode image
    print("Encoding image to base64...")
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return False
    
    # Prepare request
    payload = {
        "image_base64": base64_image,
        "confidence_threshold": confidence_threshold
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"Sending inference request (confidence threshold: {confidence_threshold})...")
    
    try:
        # Send request
        response = requests.post(
            f"{base_url}{predict_endpoint}", 
            json=payload, 
            headers=headers,
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Inference successful!")
            print(f"Number of detections: {result.get('num_detections', 0)}")
            
            # Print model info
            if 'model_info' in result:
                print(f"Model input shape: {result['model_info']['input_shape']}")
                print(f"Model output shape: {result['model_info']['output_shape']}")
            
            # Print detections
            if result.get('detections'):
                print("\nDetections found:")
                for i, detection in enumerate(result['detections']):
                    # Use the correct field names from our Lambda response
                    bbox_center = detection.get('bbox_center', [])
                    bbox_corners = detection.get('bbox_corners', [])
                    final_confidence = detection.get('final_confidence', 0)
                    objectness = detection.get('objectness', 0)
                    class_confidence = detection.get('class_confidence', 0)
                    class_id = detection.get('class_id', -1)
                    
                    print(f"  Detection {i+1}:")
                    print(f"    Final Confidence: {final_confidence:.3f}")
                    print(f"    Objectness: {objectness:.3f}")
                    print(f"    Class Confidence: {class_confidence:.3f}")
                    print(f"    Class ID: {class_id}")
                    print(f"    Bounding Box (center): [{bbox_center[0]:.1f}, {bbox_center[1]:.1f}, {bbox_center[2]:.1f}, {bbox_center[3]:.1f}]")
                    print(f"    Bounding Box (corners): [{bbox_corners[0]:.1f}, {bbox_corners[1]:.1f}, {bbox_corners[2]:.1f}, {bbox_corners[3]:.1f}]")
                    
                    # Show a few class scores if available
                    class_scores = detection.get('class_scores', [])
                    if len(class_scores) > 1:
                        print(f"    Class Scores (first 5): {[f'{score:.3f}' for score in class_scores[:5]]}")
                    print("\n")
            else:
                print("No detections found")
            
            return True
        else:
            print("Inference failed!")
            try:
                error_response = response.json()
                print(f"Error: {error_response}")
            except:
                print(f"Response text: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("Request timed out - inference took too long")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False

def main():
    """Main test function"""
    print("YOLO Lambda Inference Test")
    print("=" * 40)
    
    # Load configuration from config.yaml
    config = load_config("config.yaml")
    lambda_config = get_lambda_config(config)

    print(f"   Loaded configuration from config.yaml")
    print(f"   Base URL: {lambda_config['local_base_url']}")
    print(f"   Image Path: {lambda_config['image_path']}")
    print(f"   Confidence Threshold: {lambda_config['confidence_threshold']}")
    print(f"   Model Bucket: {lambda_config['model_s3_bucket']}")
    print(f"   Model Key: {lambda_config['model_s3_key']}")

    # Use configuration values
    base_url = lambda_config['local_base_url']
    image_path = lambda_config['image_path']
    confidence_threshold = lambda_config['confidence_threshold']
    
    # Test 1: Health check
    health_endpoint = lambda_config.get('health_endpoint', "/health") if lambda_config else "/health"
    if not test_health_check(base_url, health_endpoint):
        print("Health check failed. Make sure SAM local is running.")
        return
    
    print("Health check passed!")
    
    # Test 2: YOLO inference
    predict_endpoint = lambda_config.get('predict_endpoint', "/predict") if lambda_config else "/predict"
    success = test_yolo_inference(image_path, base_url, confidence_threshold, predict_endpoint)
    
    if success:
        print("\nAll tests passed!")
    else:
        print("\nInference test failed")

if __name__ == "__main__":
    main()