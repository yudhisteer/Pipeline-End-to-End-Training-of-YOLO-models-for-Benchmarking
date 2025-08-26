#!/usr/bin/env python3
"""
Script to test YOLO Lambda inference locally
"""
import requests
import base64
import json
import os
import boto3
from pathlib import Path
import sys
    
from utils.utils_config import load_config, get_lambda_config


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

def get_s3_images(bucket, prefix, max_images=10):
    """Get list of actual image files from S3 bucket"""
    s3_client = boto3.client('s3')
    
    try:
        print(f"Discovering images in s3://{bucket}/{prefix}...")
        
        # List objects with the given prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        images = []
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Check if it's an image file
                    if any(key.lower().endswith(ext) for ext in image_extensions):
                        images.append({
                            'key': key,
                            'size': obj['Size'],
                            'last_modified': obj['LastModified']
                        })
        
        # Sort by name and limit
        images.sort(key=lambda x: x['key'])
        images = images[:max_images]
        
        print(f"Found {len(images)} image files:")
        for i, img in enumerate(images):
            filename = img['key'].split('/')[-1]  # Get just the filename
            size_kb = img['size'] / 1024
            print(f"  {i+1:2d}. {filename} ({size_kb:.1f} KB)")
        
        return images
        
    except Exception as e:
        print(f"Error listing S3 images: {e}")
        return []

def test_batch_inference(s3_bucket, s3_prefix, base_url, confidence_threshold, predict_endpoint, num_images=5):
    """Test YOLO batch inference with real S3 images"""
    print(f"\nTesting YOLO batch inference with up to {num_images} real images from S3...")
    
    # Get real image files from S3
    image_files = get_s3_images(s3_bucket, s3_prefix, num_images)
    
    if not image_files:
        print("No image files found in S3 bucket")
        return False
    
    # Create batch payload with real S3 keys
    images = []
    for img_info in image_files:
        filename = img_info['key'].split('/')[-1]  # Get just the filename for ID
        images.append({
            "image_id": filename,
            "s3_bucket": s3_bucket,
            "s3_key": img_info['key']
        })
    
    payload = {
        "images": images,
        "confidence_threshold": confidence_threshold,
        "chunk_size": 3  # Process 3 images per chunk
    }
    
    headers = {"Content-Type": "application/json"}
    
    print(f"Sending batch request for {len(images)} images...")
    
    try:
        response = requests.post(
            f"{base_url}{predict_endpoint}",
            json=payload,
            headers=headers,
            timeout=300  # 5 minutes for batch processing
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Batch inference successful!")
            print(f"Batch size: {result.get('batch_size', 0)}")
            print(f"Chunks processed: {result.get('chunks_processed', 0)}")
            print(f"Total detections: {result.get('total_detections', 0)}")
            
            # Print results for each image
            for img_result in result.get('results', []):
                print(f"\n{img_result['image_id']}:")
                print(f"  Status: {img_result['status']}")
                if img_result['status'] == 'success':
                    num_det = img_result['num_detections']
                    print(f"  Detections: {num_det}")
                    
                    # Show first few detections if available
                    if num_det > 0 and 'detections' in img_result:
                        for i, det in enumerate(img_result['detections'][:2]):  # Show first 2
                            conf = det.get('final_confidence', 0)
                            bbox = det.get('bbox_corners', [])
                            print(f"    Det {i+1}: conf={conf:.3f}, bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]")
                        if num_det > 2:
                            print(f"    ... and {num_det - 2} more detections")
                else:
                    print(f"  Error: {img_result.get('error', 'Unknown error')}")
            
            return True
        else:
            print("Batch inference failed!")
            try:
                error_response = response.json()
                print(f"Error: {error_response}")
            except:
                print(f"Response text: {response.text}")
            return False
            
    except Exception as e:
        print(f"Batch request failed: {e}")
        return False

def main():
    """Main test function"""
    print("YOLO Lambda Inference Test")
    print("=" * 40)
    
    # Load configuration
    config = load_config("config.yaml")
    lambda_config = get_lambda_config(config)
    
    base_url = lambda_config['base_url']
    confidence_threshold = lambda_config['confidence_threshold']
    
    # Test health check first
    if not test_health_check(base_url):
        return
    
    print("Health check passed!")
    
    # Test single image
    success_single = test_yolo_inference(
        lambda_config['image_path'], 
        base_url, 
        confidence_threshold
    )
    
    # Test batch processing with real S3 files
    test_config = config.get('inference', {}).get('test', {})
    success_batch = test_batch_inference(
        test_config.get('s3_bucket'),
        test_config.get('s3_prefix'), 
        base_url,
        confidence_threshold,
        lambda_config.get('predict_endpoint', '/predict'),
        test_config.get('batch_size', 5)
    )
    
    print(f"\nResults: Single={success_single}, Batch={success_batch}")

if __name__ == "__main__":
    main()