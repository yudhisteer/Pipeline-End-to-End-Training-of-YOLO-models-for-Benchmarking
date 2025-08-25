#!/usr/bin/env python3
"""
Script to test YOLO Lambda inference locally
"""
import requests
import base64
import json
import os
from pathlib import Path

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

def test_health_check(base_url="http://127.0.0.1:3000"):
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False

def test_yolo_inference(image_path, base_url="http://127.0.0.1:3000", confidence_threshold=0.5):
    """Test YOLO inference with a local image"""
    print(f"\n🖼️  Testing YOLO inference with image: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Get image info
    file_size = os.path.getsize(image_path) / 1024  # KB
    print(f"Image size: {file_size:.1f} KB")
    
    # Encode image
    print("📦 Encoding image to base64...")
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return False
    
    print(f"Base64 length: {len(base64_image)} characters")
    
    # Prepare request
    payload = {
        "image_base64": base64_image,
        "confidence_threshold": confidence_threshold
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"🚀 Sending inference request (confidence threshold: {confidence_threshold})...")
    
    try:
        # Send request
        response = requests.post(
            f"{base_url}/predict", 
            json=payload, 
            headers=headers,
            timeout=60  # Give it time for inference
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Inference successful!")
            print(f"Number of detections: {result.get('num_detections', 0)}")
            
            # Print model info
            if 'model_info' in result:
                print(f"Model input shape: {result['model_info']['input_shape']}")
                print(f"Model output shape: {result['model_info']['output_shape']}")
            
            # Print detections
            if result.get('detections'):
                print("\n🎯 Detections found:")
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
                    print()  # Empty line between detections
            else:
                print("No detections found")
            
            return True
        else:
            print("❌ Inference failed!")
            try:
                error_response = response.json()
                print(f"Error: {error_response}")
            except:
                print(f"Response text: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out - inference took too long")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 YOLO Lambda Inference Test")
    print("=" * 40)
    
    # Configuration
    base_url = "http://127.0.0.1:3000"
    image_path = "dataset/yolo-dataset/train/images/000000000790.jpg"
    confidence_threshold = 0.3  # Lower threshold to see more detections
    
    # Test 1: Health check
    if not test_health_check(base_url):
        print("❌ Health check failed. Make sure SAM local is running.")
        return
    
    print("✅ Health check passed!")
    
    # Test 2: YOLO inference
    success = test_yolo_inference(image_path, base_url, confidence_threshold)
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Inference test failed")
    
    print("\nTip: Try different confidence thresholds (0.1 - 0.9) to see more/fewer detections")

if __name__ == "__main__":
    main()